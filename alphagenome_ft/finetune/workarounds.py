"""Paper-parity model-construction workarounds for the splice heads.

These operate on the model object returned by ``create_model_with_heads``
and exist purely to match alphagenome-pytorch's finetuning behavior; they are
not part of the core model/head implementation (that lives in
``custom_model.py``, alongside ``set_usage_num_segments``, which the same
callers also use).
"""

from __future__ import annotations

import numpy as np

_JUNCTION_ROPE_SUBMODULES = (
    "pos_donor_logits", "pos_acceptor_logits", "neg_donor_logits", "neg_acceptor_logits",
)


def reinit_junction_rope_embeddings(model, head_id: str, std: float, seed: int) -> None:
    """Overwrite a SpliceSitesJunctionHead's RoPE "embeddings" parameter with
    a fresh truncated-normal(std) sample (std=0.0 gives exact zeros).

    model._params is a plain flat {module_path: {param_name: array}} dict
    (confirmed by direct inspection, not a Haiku FlatMapping requiring
    hk.data_structures round-tripping — that round-trip was tried first and
    silently restructured the tree in a way parameter_utils.get_head_parameter_paths
    no longer recognized, breaking --heads-only optimizer masking entirely).

    Only meant to be called for explicit ``--rope-init`` ablation
    (truncated_normal or zeros) — the default ``--rope-init none`` never
    calls this, since a fresh model construction with a current
    alphagenome_research already gives this parameter a real
    TruncatedNormal(0.1)-ish init with normal nonzero gradients (the zero-init
    dead-gradient bug this function originally existed to work around,
    present in alphagenome_research 0.1.0, was fixed upstream and no longer
    reproduces without this explicit override).
    """
    import jax

    target_module_paths = {f"head/{head_id}/{sm}" for sm in _JUNCTION_ROPE_SUBMODULES}
    missing = target_module_paths - set(model._params)
    if missing:
        raise KeyError(
            f"Expected RoPE submodule(s) {sorted(missing)} not found in model "
            f"params — alphagenome_ft/alphagenome_research's "
            f"SpliceSitesJunctionHead parameter naming may have changed; "
            f"update _JUNCTION_ROPE_SUBMODULES."
        )

    key = jax.random.PRNGKey(seed)
    for module_path in sorted(target_module_paths):
        key, subkey = jax.random.split(key)
        old = model._params[module_path]["embeddings"]
        model._params[module_path] = dict(model._params[module_path])
        model._params[module_path]["embeddings"] = std * jax.random.truncated_normal(
            subkey, lower=-2.0, upper=2.0, shape=old.shape, dtype=old.dtype,
        )


_PRETRAINED_SPLICE_SITE_KEY = "alphagenome/head/splice_sites_classification/multi_organism_linear"


def init_splice_site_from_pretrained(model, head_id: str, organism_index: int = 0) -> None:
    """Initialize a custom splice_site head from the pretrained model's own
    standard splice-site classification head, matching alphagenome-pytorch's
    ``--pretrained-head-samples "splice_site:0"``.

    alphagenome-pytorch's transfer.py comment for this modality: "Fixed
    5-class output: copy full pretrained weight matrix directly" — unlike
    other modalities' per-track slicing, splice_site's classification output
    isn't per-tissue, so there is nothing to select a track of; PyTorch's
    ``:0`` there is an *organism* index (``sd[pt_key][organism_idx:organism_idx+1]``),
    not a tissue/track index, and this mirrors exactly that.

    ``create_model_with_heads``'s param-merging keeps the pretrained model's
    full param tree in ``model._params`` even for standard heads never used
    by our forward pass (see ``merge_params`` in ``custom_model.py`` — it
    appends "any keys only in pretrained" after merging our custom heads'
    keys), so the pretrained splice_sites_classification head's weights are
    already sitting in ``model._params`` unused, under a different module
    path than our own custom-named head.

    Both are the same predefined head kind (`splice_sites_classification`),
    but NOT the same shape: the pretrained model's own head is multi-organism
    ({'b': (2, 5), 'w': (2, 1536, 5)}), while our custom head — built for a
    single --organism — is single-organism ({'b': (1, 5), 'w': (1, 1536, 5)}).
    Slice the pretrained tensor down to `organism_index` (0 = human) before
    copying, matching PyTorch's own organism_idx slice exactly rather than
    assuming the shapes already match.
    """
    dst_key = f"head/{head_id}/multi_organism_linear"
    _copy_whole_organism_slice(model, dst_key, _PRETRAINED_SPLICE_SITE_KEY, organism_index)


def _copy_whole_organism_slice(model, dst_key: str, src_key: str, organism_index: int) -> None:
    """Organism-slice ``src_key``'s params and copy the whole tensor(s) into
    ``dst_key``, with no per-track indexing.

    For submodules whose output isn't per-tissue/track (e.g. splice_site's
    fixed 5-class classification head, or splice_junctions' main linear,
    which projects trunk embeddings into a shared hidden representation, not
    per-tissue tracks) — the only meaningful choice is which organism's
    weights to copy, matching the shapes exactly after that slice.
    """
    if src_key not in model._params:
        raise KeyError(f"Expected pretrained head at '{src_key}' not found in model params.")
    if dst_key not in model._params:
        raise KeyError(f"Expected custom head at '{dst_key}' not found in model params.")
    src_full = model._params[src_key]
    sliced = {
        k: v[organism_index:organism_index + 1] for k, v in src_full.items()
    }
    sliced_shapes = {k: v.shape for k, v in sliced.items()}
    dst_shapes = {k: v.shape for k, v in model._params[dst_key].items()}
    if sliced_shapes != dst_shapes:
        raise ValueError(
            f"Shape mismatch initializing '{dst_key}' from pretrained "
            f"'{src_key}'[organism_index={organism_index}]: "
            f"{dst_shapes} vs {sliced_shapes}."
        )
    model._params[dst_key] = sliced


# --- Flexible per-head pretrained/random init (--pretrained-head-samples) ---
#
# Generalizes init_splice_site_from_pretrained above to all four heads, with
# per-track granularity, mirroring alphagenome-pytorch's
# --pretrained-head-samples "modality[@resolution]:idx" flag. "idx" is either
# a single int (broadcast to every output track) or a "|"-separated list of
# int/NA, one entry per output track; "NA" (whole-modality or per-track)
# keeps the existing (random) initialization.

_JUNCTION_ROPE_KIND = "track_with_rope"
_TRACK_KIND = "track"
_ORGANISM_ONLY_KIND = "organism_only"
_GENOME_TRACK_KIND = "genome_track"

# (head_id, pretrained_kind, kind) for each user-facing modality name. Head
# IDs are fixed by this repo's runner.py (not user-configurable), so this
# mapping needs no plumbing from the caller.
_MODALITY_INFO: dict[str, tuple[str, str, str]] = {
    "rna_seq": ("rna_seq", "rna_seq", _GENOME_TRACK_KIND),
    "splice_site": ("splice_site", "splice_sites_classification", _ORGANISM_ONLY_KIND),
    "splice_usage": ("splice_usage", "splice_sites_usage", _TRACK_KIND),
    "splice_junctions": ("splice_junctions", "splice_sites_junction", _JUNCTION_ROPE_KIND),
}


def parse_pretrained_head_samples(raw: str) -> dict[str, int | list[int | None] | None]:
    """Parse a ``--pretrained-head-samples`` string into ``{modality: idx}``.

    Mirrors alphagenome-pytorch's parser (its ``args.py`` ``postprocess_args``):
    comma-separated ``modality[@resolution]:idx`` entries, where ``idx`` is a
    single int (broadcast), a ``|``-separated list of int/``NA`` (one entry
    per output track), or ``NA`` (keep the whole modality random). The
    ``modality`` key may carry an ``@resolution`` suffix (e.g. ``"rna_seq@128"``)
    verbatim, for the caller to split.
    """

    def _parse_idx(x: str) -> int | None:
        x = x.strip()
        return None if x.upper() == "NA" else int(x)

    result: dict[str, int | list[int | None] | None] = {}
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(
                f"Malformed --pretrained-head-samples entry {item!r}: "
                f"expected 'modality[@resolution]:idx'."
            )
        modality, idx_str = item.rsplit(":", 1)
        modality = modality.strip()
        idx_str = idx_str.strip()
        if "|" in idx_str:
            result[modality] = [_parse_idx(x) for x in idx_str.split("|")]
        else:
            result[modality] = _parse_idx(idx_str)
    return result


def _copy_multi_organism_linear(
    model, dst_key: str, src_key: str, organism_index: int,
    indices: int | list[int | None] | None,
) -> None:
    """Copy pretrained track(s) into a custom head's ``_MultiOrganismLinear``.

    ``indices`` semantics match alphagenome-pytorch: a single int broadcasts
    that pretrained track to every output track; a list scatters one
    pretrained (or ``None`` = keep-random) track per output track, and must
    match the head's actual track count. Out-of-range pretrained indices
    print a warning and leave that output track at its existing (random)
    init, rather than raising — matching PyTorch's lenience there.
    """
    if indices is None:
        return
    if src_key not in model._params:
        raise KeyError(f"Expected pretrained head at '{src_key}' not found in model params.")
    if dst_key not in model._params:
        raise KeyError(f"Expected custom head at '{dst_key}' not found in model params.")

    src = model._params[src_key]
    dst = model._params[dst_key]
    w_src = np.asarray(src["w"][organism_index])  # (embed, pretrained_tracks)
    b_src = np.asarray(src["b"][organism_index])  # (pretrained_tracks,)
    w_dst = np.array(dst["w"][0])  # (embed, num_tracks), mutable copy
    b_dst = np.array(dst["b"][0])  # (num_tracks,)
    num_tracks = w_dst.shape[-1]
    num_pretrained_tracks = w_src.shape[-1]

    def _in_range(idx: int) -> bool:
        if idx >= num_pretrained_tracks:
            print(
                f"  Warning: pretrained track index {idx} out of range for "
                f"'{src_key}' ({num_pretrained_tracks} tracks) — keeping "
                f"random init for '{dst_key}'."
            )
            return False
        return True

    if isinstance(indices, int):
        if _in_range(indices):
            w_dst[:, :] = w_src[:, indices:indices + 1]
            b_dst[:] = b_src[indices]
    else:
        if len(indices) != num_tracks:
            raise ValueError(
                f"--pretrained-head-samples index list for '{dst_key}' has "
                f"{len(indices)} entries but the head has {num_tracks} tracks."
            )
        for out_i, src_i in enumerate(indices):
            if src_i is None:
                continue
            if _in_range(src_i):
                w_dst[:, out_i] = w_src[:, src_i]
                b_dst[out_i] = b_src[src_i]

    model._params[dst_key] = {"w": w_dst[None], "b": b_dst[None]}


def _copy_junction_rope(
    model, head_id: str, pretrained_kind: str, organism_index: int,
    indices: int | list[int | None], hidden_dim: int,
) -> None:
    """Copy pretrained RoPE ``embeddings`` track(s) for a splice_junctions head.

    Each RoPE submodule's ``embeddings`` param is stored flattened from
    ``(num_organisms, 2, num_tissues, hidden_dim)`` (see
    ``reinit_junction_rope_embeddings`` above for the same submodule set).
    This is the *only* place a splice_junctions head is actually indexed by
    tissue/track — its main ``multi_organism_linear`` projects trunk
    embeddings into a shared ``hidden_dim``-sized representation shared
    across all tissues (see ``_copy_whole_organism_slice``), not a per-track
    layer, so ``num_tissues`` here (unlike other heads' track counts) can't
    be read from that sibling submodule and must come from these RoPE
    params' own flattened size instead: ``flat_size == 2 * num_tissues *
    hidden_dim``, with ``hidden_dim`` given (from the main linear's own
    output dim, which *is* shared/known-good on both sides).
    """
    for submodule in _JUNCTION_ROPE_SUBMODULES:
        dst_key = f"head/{head_id}/{submodule}"
        src_key = f"alphagenome/head/{pretrained_kind}/{submodule}"
        if src_key not in model._params:
            raise KeyError(f"Expected pretrained RoPE submodule at '{src_key}' not found in model params.")
        if dst_key not in model._params:
            raise KeyError(f"Expected custom RoPE submodule at '{dst_key}' not found in model params.")

        dst_flat = np.array(model._params[dst_key]["embeddings"])
        src_flat = np.asarray(model._params[src_key]["embeddings"])
        num_tracks = dst_flat.shape[-1] // (2 * hidden_dim)
        num_pretrained_tracks = src_flat.shape[-1] // (2 * hidden_dim)
        if isinstance(indices, list) and len(indices) != num_tracks:
            raise ValueError(
                f"--pretrained-head-samples index list for '{dst_key}' has "
                f"{len(indices)} entries but the head has {num_tracks} tissues."
            )

        dst_shaped = dst_flat.reshape(1, 2, num_tracks, hidden_dim)
        src_shaped = src_flat[organism_index:organism_index + 1].reshape(
            1, 2, num_pretrained_tracks, hidden_dim
        )

        def _in_range(idx: int) -> bool:
            if idx >= num_pretrained_tracks:
                print(
                    f"  Warning: pretrained RoPE track index {idx} out of range for "
                    f"'{src_key}' ({num_pretrained_tracks} tracks) — keeping "
                    f"random init for '{dst_key}'."
                )
                return False
            return True

        if isinstance(indices, int):
            if _in_range(indices):
                dst_shaped[:, :, :, :] = src_shaped[:, :, indices:indices + 1, :]
        else:
            for out_i, src_i in enumerate(indices):
                if src_i is None:
                    continue
                if _in_range(src_i):
                    dst_shaped[:, :, out_i, :] = src_shaped[:, :, src_i, :]

        model._params[dst_key] = {"embeddings": dst_shaped.reshape(dst_flat.shape)}


def _rna_seq_resolutions(model, head_id: str) -> list[int]:
    """Resolutions the custom rna_seq head was actually built with.

    Discovered dynamically from ``model._params`` rather than hardcoded,
    since this repo's runner.py overrides rna_seq to ``resolutions: [1]``
    (dropping the pretrained checkpoint's default ``[1, 128]``) — a run that
    ever changes that override should not require a code change here too.
    """
    prefix = f"head/{head_id}/resolution_"
    suffix = "/multi_organism_linear"
    resolutions = sorted(
        int(k[len(prefix):-len(suffix)])
        for k in model._params
        if k.startswith(prefix) and k.endswith(suffix)
    )
    if not resolutions:
        raise KeyError(f"No resolution-scoped multi_organism_linear params found for head '{head_id}'.")
    return resolutions


def apply_pretrained_head_samples(model, raw_spec: str, organism_index: int) -> None:
    """Apply a ``--pretrained-head-samples`` spec to a freshly-built model.

    See ``parse_pretrained_head_samples`` for the string format. Modalities
    not mentioned in ``raw_spec`` stay fully randomly initialized.
    """
    spec = parse_pretrained_head_samples(raw_spec)
    for modality_key, indices in spec.items():
        if "@" in modality_key:
            modality, resolution_str = modality_key.split("@", 1)
            requested_resolution = int(resolution_str)
        else:
            modality, requested_resolution = modality_key, None

        if modality not in _MODALITY_INFO:
            raise ValueError(
                f"Unrecognized --pretrained-head-samples modality '{modality}'; "
                f"expected one of {sorted(_MODALITY_INFO)}."
            )
        head_id, pretrained_kind, kind = _MODALITY_INFO[modality]

        if requested_resolution is not None and kind != _GENOME_TRACK_KIND:
            raise ValueError(
                f"'@resolution' is only meaningful for 'rna_seq', got '{modality_key}'."
            )

        if kind == _ORGANISM_ONLY_KIND:
            if indices is None:
                continue
            print(f"Initializing {modality} head from pretrained (organism_index={organism_index}).")
            init_splice_site_from_pretrained(model, head_id, organism_index)
        elif kind == _TRACK_KIND:
            print(f"Initializing {modality} head from pretrained tracks {indices!r}.")
            _copy_multi_organism_linear(
                model, f"head/{head_id}/multi_organism_linear",
                f"alphagenome/head/{pretrained_kind}/multi_organism_linear",
                organism_index, indices,
            )
        elif kind == _JUNCTION_ROPE_KIND:
            if indices is None:
                continue
            print(f"Initializing {modality} head (+ RoPE) from pretrained tissues {indices!r}.")
            dst_key = f"head/{head_id}/multi_organism_linear"
            src_key = f"alphagenome/head/{pretrained_kind}/multi_organism_linear"
            # The main linear projects trunk embeddings into a shared
            # hidden_dim-sized representation (not per-tissue), so it's
            # copied whole (organism-slice only) like splice_site, never
            # per-track sliced; only the RoPE embeddings are tissue-indexed.
            hidden_dim = model._params[dst_key]["w"].shape[-1]
            _copy_whole_organism_slice(model, dst_key, src_key, organism_index)
            _copy_junction_rope(model, head_id, pretrained_kind, organism_index, indices, hidden_dim)
        elif kind == _GENOME_TRACK_KIND:
            if indices is None:
                continue
            available = _rna_seq_resolutions(model, head_id)
            if requested_resolution is not None:
                if requested_resolution not in available:
                    raise ValueError(
                        f"rna_seq@{requested_resolution} requested, but the custom "
                        f"head only has resolutions {available}."
                    )
                resolutions = [requested_resolution]
            else:
                resolutions = available
            print(f"Initializing rna_seq head (resolutions={resolutions}) from pretrained tracks {indices!r}.")
            for resolution in resolutions:
                suffix = f"resolution_{resolution}/multi_organism_linear"
                _copy_multi_organism_linear(
                    model, f"head/{head_id}/{suffix}", f"alphagenome/head/{pretrained_kind}/{suffix}",
                    organism_index, indices,
                )
        else:
            raise AssertionError(f"Unhandled modality kind: {kind}")
