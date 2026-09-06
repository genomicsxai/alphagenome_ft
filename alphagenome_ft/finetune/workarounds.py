"""Paper-parity model-construction workarounds for the splice heads.

These operate on the model object returned by ``create_model_with_heads``
and exist purely to match alphagenome-pytorch's finetuning behavior; they are
not part of the core model/head implementation (that lives in
``custom_model.py``, alongside ``set_usage_num_segments``, which the same
callers also use).
"""

from __future__ import annotations

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
    if _PRETRAINED_SPLICE_SITE_KEY not in model._params:
        raise KeyError(
            f"Expected pretrained standard head at "
            f"'{_PRETRAINED_SPLICE_SITE_KEY}' not found in model params — "
            f"alphagenome_research's splice_sites_classification head "
            f"parameter naming may have changed."
        )
    if dst_key not in model._params:
        raise KeyError(
            f"Expected custom head at '{dst_key}' not found in model params."
        )
    src_full = model._params[_PRETRAINED_SPLICE_SITE_KEY]
    sliced = {
        k: v[organism_index:organism_index + 1] for k, v in src_full.items()
    }
    sliced_shapes = {k: v.shape for k, v in sliced.items()}
    dst_shapes = {k: v.shape for k, v in model._params[dst_key].items()}
    if sliced_shapes != dst_shapes:
        raise ValueError(
            f"Shape mismatch initializing '{dst_key}' from pretrained "
            f"'{_PRETRAINED_SPLICE_SITE_KEY}'[organism_index={organism_index}]: "
            f"{dst_shapes} vs {sliced_shapes}."
        )
    model._params[dst_key] = sliced
