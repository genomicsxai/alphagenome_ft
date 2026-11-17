"""Pure-JAX ops for ``junction_position_source="predicted"`` splice training.

Unlike the annotated-position path (where the junction target matrix is
precomputed once at data-loading time with pandas, see
``alphagenome_ft.finetune.star_junctions.junctions_to_junction_matrix``),
predicted mode needs positions derived from the classification head's
*current* logits at every training step. Rebuilding the junction target
matrix from those positions therefore has to happen inside the same JAX
trace as the rest of the training step — no pandas, no host round-trip —
so gradients through the (possibly non-frozen) backbone stay correct
regardless of ``heads_only``. These two functions are that trace-safe,
vectorized replacement.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int


# alphagenome_research.model.model.DEFAULT_NUM_SPLICE_SITES / DEFAULT_SPLICE_SITE_THRESHOLD
DEFAULT_JUNCTION_TOP_K = 512
DEFAULT_SPLICE_SITE_THRESHOLD = 0.1


def top_k_splice_positions(
    logits: Float[Array, "B S 5"],
    top_k: int,
    threshold: float = DEFAULT_SPLICE_SITE_THRESHOLD,
) -> Int[Array, "B 4 {top_k}"]:
    """Derive junction-head splice-site positions from classification logits.

    Mirrors ``alphagenome_research.model.splicing._top_k_splice_sites`` (and
    ``alphagenome_pytorch``'s ``_top_k_positions_from_logits``): for each of
    the 4 role channels (0=Donor+, 1=Acceptor+, 2=Donor-, 3=Acceptor-;
    channel 4 is background), take the ``top_k`` highest softmax
    *probabilities*, drop sites below ``threshold``, and sort the rest
    ascending with ``-1`` padding at the end.

    Uses exact ``jax.lax.top_k`` (alphagenome_research uses
    ``approx_max_k``, which is exact off-TPU).

    Args:
        logits: Classification head logits, ``(B, S, 5)``, NLC layout.
        top_k: Number of positions to keep per role. If ``top_k > S`` the
            remaining columns are padded with -1.
        threshold: Minimum class probability for a selected site; 0 disables.

    Returns:
        Int32 array ``(B, 4, top_k)`` of 0-based positions, -1 padded,
        wrapped in ``stop_gradient``.
    """
    batch_size, seq_len, _ = logits.shape
    k = min(top_k, seq_len)
    probs = jax.nn.softmax(logits.astype(jnp.float32), axis=-1)

    def _per_role(channel_idx: int) -> Int[Array, "B {k}"]:
        values, idx = jax.lax.top_k(probs[:, :, channel_idx], k)
        if threshold > 0:
            # Push below-threshold sites past every valid index so they sort last.
            idx = jnp.where(values < threshold, seq_len, idx)
        idx = jnp.sort(idx, axis=-1)
        if threshold > 0:
            idx = jnp.where(idx == seq_len, -1, idx)
        return idx

    positions = jnp.stack([_per_role(c) for c in range(4)], axis=1)
    if k < top_k:
        pad = jnp.full((batch_size, 4, top_k - k), -1, dtype=positions.dtype)
        positions = jnp.concatenate([positions, pad], axis=-1)
    return jax.lax.stop_gradient(positions.astype(jnp.int32))


def _match_index(
    query: Int[Array, "M"],
    position_row: Int[Array, "K"],
    valid: Bool[Array, "M"],
) -> Int[Array, "M"]:
    """First index in ``position_row`` equal to each valid ``query`` entry, else -1."""
    eq = (query[:, None] == position_row[None, :]) & valid[:, None] & (position_row[None, :] >= 0)
    any_match = jnp.any(eq, axis=-1)
    idx = jnp.argmax(eq, axis=-1)
    return jnp.where(any_match, idx, -1)


def _build_junction_matrix_one(
    positions: Int[Array, "4 K"],
    junction_d_rel: Int[Array, "M"],
    junction_a_rel: Int[Array, "M"],
    junction_is_pos_strand: Bool[Array, "M"],
    junction_counts: Float[Array, "M C"],
    max_splice_sites: int,
) -> Float[Array, "K K {2*C}"]:
    n_samples = junction_counts.shape[-1]
    valid = (junction_d_rel >= 0) & (junction_a_rel >= 0)

    def _strand_matrix(donor_row, acceptor_row, strand_mask):
        d_idx = _match_index(junction_d_rel, donor_row, valid & strand_mask)
        a_idx = _match_index(junction_a_rel, acceptor_row, valid & strand_mask)
        ok = (d_idx >= 0) & (a_idx >= 0)
        d_idx = jnp.where(ok, d_idx, 0)
        a_idx = jnp.where(ok, a_idx, 0)
        values = jnp.where(ok[:, None], junction_counts, 0.0)
        matrix = jnp.zeros((max_splice_sites, max_splice_sites, n_samples), dtype=jnp.float32)
        return matrix.at[d_idx, a_idx, :].add(values)

    pos_matrix = _strand_matrix(positions[0], positions[1], junction_is_pos_strand)
    neg_matrix = _strand_matrix(positions[2], positions[3], ~junction_is_pos_strand)
    return jnp.concatenate([pos_matrix, neg_matrix], axis=-1)


def build_junction_matrix(
    positions: Int[Array, "B 4 K"],
    junction_d_rel: Int[Array, "B M"],
    junction_a_rel: Int[Array, "B M"],
    junction_is_pos_strand: Bool[Array, "B M"],
    junction_counts: Float[Array, "B M C"],
    max_splice_sites: int,
) -> Float[Array, "B K K {2*C}"]:
    """Vectorized, jit/pmap-safe replacement for the pandas junction-matrix build.

    Rebuilds the ``(B, K, K, 2*n_samples)`` donor x acceptor read-count
    matrix for a batch of windows, given per-window predicted (or
    annotated) ``positions`` and a padded list of raw junction events. This
    is the trace-safe equivalent of
    ``star_junctions.junctions_to_junction_matrix(..., positions=...)``
    (predicted-mode branch) — same semantics, but pure JAX so it can run
    inside the same ``value_and_grad`` trace as the rest of training.

    Args:
        positions: ``(B, 4, K)`` int, -1 padded — [pos_donor, pos_acceptor,
            neg_donor, neg_acceptor], 0-based relative to the window.
        junction_d_rel: ``(B, M)`` int, -1 padded — donor position of each
            raw junction event.
        junction_a_rel: ``(B, M)`` int, -1 padded — acceptor position.
        junction_is_pos_strand: ``(B, M)`` bool strand flag per event.
        junction_counts: ``(B, M, n_samples)`` float per-sample read count.
        max_splice_sites: Static ``K``, must match ``positions.shape[-1]``.

    Returns:
        Float32 array ``(B, K, K, 2*n_samples)`` — same layout as the
        annotated-mode matrix (positive-strand sample channels first, then
        negative-strand).
    """
    return jax.vmap(_build_junction_matrix_one, in_axes=(0, 0, 0, 0, 0, None))(
        positions,
        junction_d_rel,
        junction_a_rel,
        junction_is_pos_strand,
        junction_counts,
        max_splice_sites,
    )
