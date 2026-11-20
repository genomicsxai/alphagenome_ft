"""Data loading for splice fine-tuning (classification, usage, junctions).

Produces training batches for the three splice modalities supported by the
real ``alphagenome_research`` heads (``SpliceSitesClassificationHead``,
``SpliceSitesUsageHead``, ``SpliceSitesJunctionHead``): splice site
classification, splice site usage (SSU), and splice junction read counts.

Reuses the STAR ``SJ.out.tab`` / SSU parquet / GTF preprocessing pipeline
from ``alphagenome-pytorch``'s finetuning extension: run
``scripts/get_star_junctions.py`` and ``scripts/compute_ssu.py`` from that
repo to produce the junction/SSU files consumed here (the file formats are
identical; only the JAX-side windowing/batching differs). See
``alphagenome_ft/finetune/star_junctions.py`` for the format parsers, ported
from ``alphagenome_pytorch/extensions/finetuning/star_junctions.py``.

A single :class:`SpliceDataModule` instance backs up to three head kinds
sharing one STAR-junction (+ optional SSU/GTF) source: ``splice_sites``
(classification), ``splice_site_usage``, and ``splice_junctions``. Which
head kinds are produced, and under which head id, is controlled by
``head_kinds``.
"""

from __future__ import annotations

from typing import Iterator, Mapping, Sequence

import numpy as np
import pandas as pd

from alphagenome.data import genome
from alphagenome.io import fasta as fasta_lib
from alphagenome_research.model import one_hot_encoder

from alphagenome_ft.finetune.star_junctions import (
    donor_acceptor_positions,
    gtf_splice_sites_to_junctions,
    junctions_to_site_table,
    junctions_to_classification_array,
    junctions_to_junction_matrix,
    junctions_to_ssu_approx_arrays_by_strand,
    normalize_junctions_per_sample,
    read_star_junctions,
    read_ssu_parquet,
    splice_sites_to_classification_array,
    ssu_to_arrays_by_strand,
)



class SpliceDataModule:
    """Creates training batches with splice classification/usage/junction targets."""

    def __init__(
        self,
        *,
        intervals: Mapping[str, Sequence[genome.Interval]],
        fasta_path,
        star_junction_files: Sequence[str],
        head_kinds: Mapping[str, str],
        batch_size: int,
        shuffle: bool,
        ssu_files: Sequence[str] | None = None,
        gtf_file: str | None = None,
        min_unique_reads: int = 1,
        filter_to_junctions: bool = True,
        max_splice_sites: int = 256,
        drop_last: bool = False,
        emit_raw_junction_events: bool = False,
        junction_event_pad_size: int = 1024,
    ) -> None:
        """Initialize streaming sequence/splice-target batch generation.

        Args:
            intervals: Split-to-interval mapping (``train``/``valid``/``test``).
                Intervals are used as-is (no resizing) — build them at the
                desired sequence length beforehand.
            fasta_path: Reference FASTA used to extract sequence windows.
            star_junction_files: STAR ``SJ.out.tab`` files, one per sample.
                Always drive the ``splice_junctions`` target; also drive
                ``splice_sites``/``splice_site_usage`` when ``ssu_files`` is
                not provided.
            head_kinds: Mapping from splice kind (``splice_sites``,
                ``splice_site_usage``, ``splice_junctions``) to the head id
                that should receive that target under ``targets_{head_id}``.
                Omit a kind to skip producing that target.
            batch_size: Number of windows per yielded batch.
            shuffle: Whether to shuffle window order in ``iter_batches``.
            ssu_files: Optional per-sample SSU parquet paths (same order as
                ``star_junction_files``). When provided, drives
                ``splice_sites``/``splice_site_usage`` instead of the
                junction-only fallback.
            gtf_file: Optional GTF/parquet of canonical splice sites, added
                to the classification target only (annotation-only, zero
                usage).
            min_unique_reads: Minimum uniquely-mapped reads for a junction
                to be kept.
            filter_to_junctions: If True (default), discard intervals with no
                complete splice junction across all junction files.
            max_splice_sites: Max sites per donor/acceptor role kept for the
                junction-position/matrix targets (padded with -1).
            drop_last: If True, drop incomplete final batches.
            emit_raw_junction_events: If True, also emit padded per-window
                raw junction event lists (``junction_d_rel``/``junction_a_rel``/
                ``junction_is_pos_strand``/``junction_counts``) needed by
                ``junction_position_source="predicted"`` training, where the
                target matrix must be rebuilt from positions that aren't
                known until forward time (see
                ``alphagenome_ft.finetune.splice_positions.build_junction_matrix``).
                Only meaningful together with a ``splice_junctions`` head kind.
            junction_event_pad_size: Minimum padded row count for the raw
                junction event lists (one row per (sample, junction) pair
                inside the window); a batch with more events grows it by
                powers of two, so no event is ever dropped. Only used when
                ``emit_raw_junction_events=True``.
        """
        if not head_kinds:
            raise ValueError("head_kinds must include at least one splice kind.")
        valid_kinds = {"splice_sites", "splice_site_usage", "splice_junctions"}
        unknown = set(head_kinds) - valid_kinds
        if unknown:
            raise ValueError(f"Unknown splice kind(s) {unknown}; expected {valid_kinds}.")

        self._fasta_path = fasta_path
        self._head_kinds = dict(head_kinds)
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._drop_last = drop_last
        self._max_splice_sites = max_splice_sites
        self._emit_raw_junction_events = emit_raw_junction_events
        if junction_event_pad_size < 1:
            raise ValueError("junction_event_pad_size must be >= 1.")
        self._junction_event_pad_size = junction_event_pad_size
        self._encoder = one_hot_encoder.DNAOneHotEncoder(dtype=np.float32)

        self.star_junction_files = [str(p) for p in star_junction_files]
        self.ssu_files = [str(p) for p in ssu_files] if ssu_files is not None else None

        self._all_juncs: list[pd.DataFrame] = []
        for path in self.star_junction_files:
            junc = read_star_junctions(path)
            junc = junc.loc[junc["n_uniquely_mapped_reads"] >= min_unique_reads].copy()
            junc = junc.loc[
                junc["chrom"].str.contains("chr", na=False)
                & junc["strand"].isin(["+", "-"])
            ].drop_duplicates()
            junc["exon_start"] = junc["intron_start"] - 1
            junc["exon_end"] = junc["intron_end"] + 1
            junc["count"] = junc["n_uniquely_mapped_reads"]
            junc = normalize_junctions_per_sample(junc)
            self._all_juncs.append(junc)

        # Per-chrom slices of self._all_juncs, used by the per-window hot path
        # (_window_targets/_raw_junction_events) so those calls filter a
        # ~1/24th-size chrom slice instead of re-scanning the full per-sample
        # table (hundreds of thousands of rows) on every single window.
        self._all_juncs_by_chrom: list[dict[str, pd.DataFrame]] = [
            {chrom: group for chrom, group in junc.groupby("chrom", sort=False)}
            for junc in self._all_juncs
        ]
        self._empty_juncs = [junc.iloc[0:0] for junc in self._all_juncs]

        self._gtf_sites: pd.DataFrame | None = None
        self._gtf_juncs: pd.DataFrame | None = None
        if gtf_file is not None:
            gtf_juncs = gtf_splice_sites_to_junctions(gtf_file)
            self._gtf_juncs = gtf_juncs
            self._gtf_sites = junctions_to_site_table(gtf_juncs)

        if self._all_juncs:
            union_juncs = pd.concat(
                [j[["chrom", "exon_start", "exon_end", "strand"]] for j in self._all_juncs],
                ignore_index=True,
            ).drop_duplicates()
        else:
            union_juncs = pd.DataFrame(columns=["chrom", "exon_start", "exon_end", "strand"])

        self._intervals: dict[str, list[genome.Interval]] = {}
        for split, windows in intervals.items():
            windows = list(windows)
            if filter_to_junctions and windows and not union_juncs.empty:
                windows = [
                    w
                    for w in windows
                    if not union_juncs.loc[
                        (union_juncs["chrom"] == w.chromosome)
                        & (union_juncs["exon_start"] > w.start)
                        & (union_juncs["exon_end"] <= w.end)
                    ].empty
                ]
            if windows:
                self._intervals[split] = windows

    def iter_batches(
        self, split: str, *, seed: int | None = None, skip_batches: int = 0
    ) -> Iterator[dict[str, np.ndarray]]:
        windows = list(self._intervals.get(split, ()))
        if not windows:
            return

        order = np.arange(len(windows))
        # Only the train split is shuffled (validation keeps BED order), matching
        # BigWigDataModule.iter_batches; both must agree for CombinedDataModule.
        if self._shuffle and split == "train":
            rng = np.random.default_rng(seed)
            rng.shuffle(order)
        if split != "train" and self._drop_last:
            # Evaluation never drops windows: pad the final partial batch by
            # wrapping around to the start, as torch's DistributedSampler does
            # for alphagenome-pytorch's validation loader.
            order = _pad_order_to_batch_multiple(order, self._batch_size)
        if skip_batches:
            # Resuming mid-epoch: the shuffle above is a deterministic
            # function of (len(windows), seed), so slicing off the already
            # -consumed batches here reproduces the exact same batch order a
            # full re-iteration would skip past — without paying to extract
            # (decode FASTA + bigwig lookups for) any of the discarded
            # windows.
            order = order[skip_batches * self._batch_size:]

        extractor = fasta_lib.FastaExtractor(str(self._fasta_path))

        batch_indices: list[int] = []
        for idx in order:
            batch_indices.append(int(idx))
            if len(batch_indices) == self._batch_size:
                yield self._make_batch(batch_indices, windows, extractor)
                batch_indices = []

        if batch_indices and not self._drop_last:
            yield self._make_batch(batch_indices, windows, extractor)

    def _juncs_for_chrom(self, chrom: str) -> list[pd.DataFrame]:
        """Per-sample junction DataFrames restricted to one chromosome.

        Precomputed once in __init__ via groupby; looking this up per window
        avoids re-scanning the full (hundreds-of-thousands-row) per-sample
        table with a fresh boolean mask on every call.
        """
        return [
            by_chrom.get(chrom, empty)
            for by_chrom, empty in zip(self._all_juncs_by_chrom, self._empty_juncs)
        ]

    def _raw_junction_events(
        self, chrom: str, start: int, seq_len: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Unpadded, per-(sample, junction) event list for predicted-mode rebuild.

        One row per (sample, junction-inside-window) pair — samples are
        *not* merged into a shared event, matching how
        ``junctions_to_junction_matrix`` accumulates each sample into its
        own channel independently. Junctions whose strand is neither "+" nor
        "-" are skipped, as in ``junctions_to_junction_matrix``. Every event
        is kept (no cap); ``_make_batch`` pads the rows to a shared length.
        See ``alphagenome_ft.finetune.splice_positions.build_junction_matrix``,
        which consumes this layout.
        """
        n_samples = len(self._all_juncs)
        end = start + seq_len
        d_rel: list[int] = []
        a_rel: list[int] = []
        is_pos_strand: list[bool] = []
        sample_rows: list[int] = []
        sample_counts: list[float] = []
        for sample_idx, junc_df in enumerate(self._juncs_for_chrom(chrom)):
            mask = (
                (junc_df["exon_start"] > start)
                & (junc_df["exon_start"] <= end)
                & (junc_df["exon_end"] > start)
                & (junc_df["exon_end"] <= end)
                & junc_df["strand"].isin(("+", "-"))
            )
            local = junc_df.loc[mask]
            donor, acceptor = donor_acceptor_positions(local)
            d_rel.extend((donor - 1 - start).tolist())
            a_rel.extend((acceptor - 1 - start).tolist())
            is_pos_strand.extend((local["strand"] == "+").tolist())
            sample_rows.extend([sample_idx] * len(local))
            sample_counts.extend(local["count"].astype(float).tolist())

        m = len(d_rel)
        counts = np.zeros((m, n_samples), dtype=np.float32)
        counts[np.arange(m), np.asarray(sample_rows, dtype=np.int64)] = sample_counts
        return (
            np.asarray(d_rel, dtype=np.int32).reshape(m),
            np.asarray(a_rel, dtype=np.int32).reshape(m),
            np.asarray(is_pos_strand, dtype=bool).reshape(m),
            counts,
        )

    def _junction_event_rows(self, max_events: int) -> int:
        """Padded event-row count for a batch: the smallest power-of-two
        multiple of ``junction_event_pad_size`` that holds ``max_events``.

        Never truncates; bucketing keeps the number of distinct shapes (and
        hence jit retraces) logarithmic in the largest window's event count.
        """
        rows = self._junction_event_pad_size
        while rows < max_events:
            rows *= 2
        return rows

    def _window_targets(self, chrom: str, start: int, seq_len: int) -> dict[str, np.ndarray]:
        """Compute the shared per-window splice arrays used by all head kinds."""
        if self.ssu_files is not None:
            ssu_dfs = [
                read_ssu_parquet(p, chrom, start, start + seq_len) for p in self.ssu_files
            ]
            cls_sources = []
            for ssu_df in ssu_dfs:
                if "ssu_spliser" not in ssu_df.columns:
                    raise ValueError("ssu_df is missing required column ssu_spliser")
                cls_sources.append(ssu_df[ssu_df["ssu_spliser"].notna()])
            if self._gtf_sites is not None:
                cls_sources.append(self._gtf_sites)
            cls_arr = splice_sites_to_classification_array(cls_sources, chrom, start, seq_len)

            usage_tracks, alpha_tracks = [], []
            for ssu_df in ssu_dfs:
                pos_arr, neg_arr, pos_alpha, neg_alpha = ssu_to_arrays_by_strand(
                    ssu_df, chrom, start, seq_len
                )
                usage_tracks.extend([pos_arr, neg_arr])
                alpha_tracks.extend([pos_alpha, neg_alpha])
        else:
            chrom_juncs = self._juncs_for_chrom(chrom)
            cls_juncs = chrom_juncs + ([self._gtf_juncs] if self._gtf_juncs is not None else [])
            cls_arr = junctions_to_classification_array(cls_juncs, chrom, start, seq_len)

            usage_tracks, alpha_tracks = [], []
            for junc_df in chrom_juncs:
                pos_arr, neg_arr = junctions_to_ssu_approx_arrays_by_strand(
                    junc_df, chrom, start, seq_len
                )
                usage_tracks.extend([pos_arr, neg_arr])
                alpha_tracks.extend(
                    [np.full(seq_len, -1.0, dtype=np.float32), np.full(seq_len, -1.0, dtype=np.float32)]
                )
        usage_arr = np.stack(usage_tracks, axis=-1)
        usage_alpha_arr = np.stack(alpha_tracks, axis=-1)

        junc_positions, junc_matrix = junctions_to_junction_matrix(
            self._juncs_for_chrom(chrom),
            max_splice_sites=self._max_splice_sites,
            cls_arr=cls_arr,
            chrom=chrom,
            start=start,
            seq_len=seq_len,
        )

        result = {
            "probs": cls_arr,
            "usage": usage_arr,
            "usage_alpha": usage_alpha_arr,
            "junction_positions": junc_positions,
            "junction_matrix": junc_matrix,
        }
        if self._emit_raw_junction_events:
            d_rel, a_rel, is_pos_strand, counts = self._raw_junction_events(chrom, start, seq_len)
            result["junction_d_rel"] = d_rel
            result["junction_a_rel"] = a_rel
            result["junction_is_pos_strand"] = is_pos_strand
            result["junction_counts"] = counts
        return result

    def _make_batch(
        self,
        batch_indices: Sequence[int],
        windows: Sequence[genome.Interval],
        extractor: fasta_lib.FastaExtractor,
    ) -> dict[str, np.ndarray]:
        sequences = []
        per_window: list[dict[str, np.ndarray]] = []

        for idx in batch_indices:
            window = windows[idx]
            seq = extractor.extract(window)
            encoded = self._encoder.encode(seq)
            sequences.append(encoded)
            seq_len = encoded.shape[0]
            per_window.append(self._window_targets(window.chromosome, window.start, seq_len))

        batch: dict[str, np.ndarray] = {
            "sequences": np.stack(sequences, axis=0).astype(np.float32),
            "negative_strand_mask": np.zeros((len(batch_indices),), dtype=bool),
        }

        if "splice_sites" in self._head_kinds:
            head_id = self._head_kinds["splice_sites"]
            batch[f"targets_{head_id}"] = np.stack(
                [w["probs"] for w in per_window], axis=0
            ).astype(np.float32)

        if "splice_site_usage" in self._head_kinds:
            head_id = self._head_kinds["splice_site_usage"]
            batch[f"targets_{head_id}"] = np.stack(
                [w["usage"] for w in per_window], axis=0
            ).astype(np.float32)
            batch[f"usage_alpha_{head_id}"] = np.stack(
                [w["usage_alpha"] for w in per_window], axis=0
            ).astype(np.float32)

        if "splice_junctions" in self._head_kinds:
            head_id = self._head_kinds["splice_junctions"]
            batch[f"targets_{head_id}"] = np.stack(
                [w["junction_matrix"] for w in per_window], axis=0
            ).astype(np.float32)
            batch["splice_site_positions"] = np.stack(
                [w["junction_positions"] for w in per_window], axis=0
            ).astype(np.int32)

        if self._emit_raw_junction_events:
            rows = self._junction_event_rows(
                max(len(w["junction_d_rel"]) for w in per_window)
            )
            n_samples = len(self._all_juncs)
            d_rel = np.full((len(per_window), rows), -1, dtype=np.int32)
            a_rel = np.full((len(per_window), rows), -1, dtype=np.int32)
            is_pos_strand = np.zeros((len(per_window), rows), dtype=bool)
            counts = np.zeros((len(per_window), rows, n_samples), dtype=np.float32)
            for i, w in enumerate(per_window):
                m = len(w["junction_d_rel"])
                d_rel[i, :m] = w["junction_d_rel"]
                a_rel[i, :m] = w["junction_a_rel"]
                is_pos_strand[i, :m] = w["junction_is_pos_strand"]
                counts[i, :m] = w["junction_counts"]
            batch["junction_d_rel"] = d_rel
            batch["junction_a_rel"] = a_rel
            batch["junction_is_pos_strand"] = is_pos_strand
            batch["junction_counts"] = counts

        return batch


def _pad_order_to_batch_multiple(order: np.ndarray, batch_size: int) -> np.ndarray:
    """Wrap-pad ``order`` to a multiple of ``batch_size`` (DistributedSampler-style)."""
    remainder = len(order) % batch_size
    if remainder == 0:
        return order
    return np.concatenate([order, np.resize(order, batch_size - remainder)])
