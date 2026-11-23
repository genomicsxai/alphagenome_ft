"""Splice targets (classification, usage, junctions) for the finetuning data module.

Builds per-window targets for the three splice heads supported by the real
``alphagenome_research`` heads (``SpliceSitesClassificationHead``,
``SpliceSitesUsageHead``, ``SpliceSitesJunctionHead``) from STAR
``SJ.out.tab`` files, optional SSU parquets and an optional GTF/parquet of
annotated sites. Windowing, shuffling and batching live in
:class:`alphagenome_ft.finetune.data.FinetuneDataModule`, which calls into
:class:`SpliceTargets` for every head whose spec carries a ``splice_source``.

Input formats are shared with ``alphagenome-pytorch`` (its
``agt preprocess`` tools produce the junction/SSU files consumed here); see
``alphagenome_ft/finetune/star_junctions.py`` for the parsers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Mapping, Sequence

import numpy as np
import pandas as pd

from alphagenome.data import genome

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

from alphagenome_ft.finetune.config import SPLICE_KINDS

if TYPE_CHECKING:
    from alphagenome_ft.finetune.config import HeadSpec, SpliceSourceSpec

CLASSIFICATION_KIND, USAGE_KIND, JUNCTION_KIND = SPLICE_KINDS


def _source_key(source: "SpliceSourceSpec") -> tuple:
    """Fields that define the underlying data; heads sharing them share one loader."""
    return (
        tuple(str(p) for p in source.star_junctions),
        tuple(str(p) for p in source.ssu) if source.ssu is not None else None,
        str(source.gtf) if source.gtf is not None else None,
        int(source.min_unique_reads),
    )


def build_splice_targets(
    head_specs: Sequence["HeadSpec"], *, junction_event_pad_size: int = 1024,
) -> list["SpliceTargets"]:
    """Group splice head specs by data source into :class:`SpliceTargets`.

    Heads that read the same STAR/SSU/GTF files share one loader, so the
    files are parsed once and each window's targets are computed once.
    """
    groups: dict[tuple, list["HeadSpec"]] = {}
    for spec in head_specs:
        if spec.splice_source is None:
            continue
        if spec.kind not in SPLICE_KINDS:
            raise ValueError(f"Head {spec.head_id!r} has a splice_source but kind {spec.kind!r}.")
        groups.setdefault(_source_key(spec.splice_source), []).append(spec)

    n_junction = sum(1 for specs in groups.values() for s in specs if s.kind == JUNCTION_KIND)
    if n_junction > 1:
        # The model reads junction positions from one batch-level key.
        raise ValueError("At most one splice_sites_junction head is supported per model.")

    targets = []
    for specs in groups.values():
        head_ids: dict[str, str] = {}
        for spec in specs:
            if spec.kind in head_ids:
                raise ValueError(
                    f"Heads {head_ids[spec.kind]!r} and {spec.head_id!r} have the same kind "
                    f"{spec.kind!r} and data source; use one head."
                )
            head_ids[spec.kind] = spec.head_id
        junction_spec = next((s for s in specs if s.kind == JUNCTION_KIND), None)
        source = (junction_spec or specs[0]).splice_source
        targets.append(
            SpliceTargets(
                source,
                head_ids,
                filter_to_junctions=any(s.splice_source.filter_to_junctions for s in specs),
                emit_raw_junction_events=(
                    junction_spec is not None
                    and junction_spec.splice_source.junction_position_source == "predicted"
                ),
                junction_event_pad_size=junction_event_pad_size,
            )
        )
    return targets


class SpliceTargets:
    """Per-window splice targets for the heads that share one data source.

    Args:
        source: Data source (STAR junction files, optional SSU/GTF, filters).
        head_ids: Splice head kind -> head id for the heads fed by ``source``.
        filter_to_junctions: Whether windows without a complete junction
            should be dropped (see :meth:`has_junction`).
        emit_raw_junction_events: Also emit padded raw junction events
            (``junction_d_rel``/``junction_a_rel``/``junction_is_pos_strand``/
            ``junction_counts``) for ``junction_position_source="predicted"``,
            where the target matrix is rebuilt from positions only known at
            forward time (``splice_positions.build_junction_matrix``).
        junction_event_pad_size: Minimum padded row count for the raw events;
            a batch with more events grows it by powers of two (never drops).
    """

    def __init__(
        self,
        source: "SpliceSourceSpec",
        head_ids: Mapping[str, str],
        *,
        filter_to_junctions: bool = True,
        emit_raw_junction_events: bool = False,
        junction_event_pad_size: int = 1024,
    ) -> None:
        unknown = set(head_ids) - set(SPLICE_KINDS)
        if not head_ids or unknown:
            raise ValueError(f"head_ids must map splice kinds {SPLICE_KINDS}; got {dict(head_ids)}.")
        if junction_event_pad_size < 1:
            raise ValueError("junction_event_pad_size must be >= 1.")
        self.head_ids = dict(head_ids)
        self.filter_to_junctions = filter_to_junctions
        self._max_splice_sites = int(source.max_splice_sites)
        self._emit_raw_junction_events = emit_raw_junction_events
        self._junction_event_pad_size = junction_event_pad_size

        self.star_junction_files = [str(p) for p in source.star_junctions]
        self.ssu_files = [str(p) for p in source.ssu] if source.ssu is not None else None

        self._all_juncs: list[pd.DataFrame] = []
        for path in self.star_junction_files:
            junc = read_star_junctions(path)
            junc = junc.loc[junc["n_uniquely_mapped_reads"] >= source.min_unique_reads].copy()
            junc = junc.loc[
                junc["chrom"].str.contains("chr", na=False)
                & junc["strand"].isin(["+", "-"])
            ].drop_duplicates()
            junc["exon_start"] = junc["intron_start"] - 1
            junc["exon_end"] = junc["intron_end"] + 1
            junc["count"] = junc["n_uniquely_mapped_reads"]
            self._all_juncs.append(normalize_junctions_per_sample(junc))

        # Per-chromosome slices for the per-window hot path, so each window
        # filters a small chromosome table instead of the full sample table.
        self._all_juncs_by_chrom: list[dict[str, pd.DataFrame]] = [
            {chrom: group for chrom, group in junc.groupby("chrom", sort=False)}
            for junc in self._all_juncs
        ]
        self._empty_juncs = [junc.iloc[0:0] for junc in self._all_juncs]

        self._gtf_sites: pd.DataFrame | None = None
        self._gtf_juncs: pd.DataFrame | None = None
        if source.gtf is not None:
            self._gtf_juncs = gtf_splice_sites_to_junctions(str(source.gtf))
            self._gtf_sites = junctions_to_site_table(self._gtf_juncs)

        cols = ["chrom", "exon_start", "exon_end", "strand"]
        self._union_juncs = (
            pd.concat([j[cols] for j in self._all_juncs], ignore_index=True).drop_duplicates()
            if self._all_juncs else pd.DataFrame(columns=cols)
        )

    def has_junction(self, window: genome.Interval) -> bool:
        """Whether ``window`` contains at least one complete junction (any sample)."""
        juncs = self._union_juncs
        if juncs.empty:
            return True
        return not juncs.loc[
            (juncs["chrom"] == window.chromosome)
            & (juncs["exon_start"] > window.start)
            & (juncs["exon_end"] <= window.end)
        ].empty

    def _juncs_for_chrom(self, chrom: str) -> list[pd.DataFrame]:
        return [
            by_chrom.get(chrom, empty)
            for by_chrom, empty in zip(self._all_juncs_by_chrom, self._empty_juncs)
        ]

    def _raw_junction_events(
        self, chrom: str, start: int, seq_len: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Unpadded per-(sample, junction) events for the predicted-mode rebuild.

        One row per (sample, junction inside the window); each sample keeps
        its own channel, as in ``junctions_to_junction_matrix``. Donor and
        acceptor are strand-aware (``donor_acceptor_positions``).
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

    def junction_event_rows(self, max_events: int) -> int:
        """Padded event-row count: smallest power-of-two multiple of the pad size
        that holds ``max_events`` (never truncates; few distinct jit shapes)."""
        rows = self._junction_event_pad_size
        while rows < max_events:
            rows *= 2
        return rows

    def window_targets(self, chrom: str, start: int, seq_len: int) -> dict[str, np.ndarray]:
        """Per-window splice arrays shared by all head kinds of this source."""
        if self.ssu_files is not None:
            ssu_dfs = [read_ssu_parquet(p, chrom, start, start + seq_len) for p in self.ssu_files]
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
                alpha_tracks.extend([np.full(seq_len, -1.0, dtype=np.float32)] * 2)

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
            "usage": np.stack(usage_tracks, axis=-1),
            "usage_alpha": np.stack(alpha_tracks, axis=-1),
            "junction_positions": junc_positions,
            "junction_matrix": junc_matrix,
        }
        if self._emit_raw_junction_events:
            d_rel, a_rel, is_pos_strand, counts = self._raw_junction_events(chrom, start, seq_len)
            result.update(
                junction_d_rel=d_rel, junction_a_rel=a_rel,
                junction_is_pos_strand=is_pos_strand, junction_counts=counts,
            )
        return result

    def add_to_batch(self, batch: dict[str, np.ndarray], per_window: Sequence[dict]) -> None:
        """Stack ``window_targets`` results into the batch under the model's keys."""
        if CLASSIFICATION_KIND in self.head_ids:
            batch[f"targets_{self.head_ids[CLASSIFICATION_KIND]}"] = np.stack(
                [w["probs"] for w in per_window]).astype(np.float32)
        if USAGE_KIND in self.head_ids:
            head_id = self.head_ids[USAGE_KIND]
            batch[f"targets_{head_id}"] = np.stack([w["usage"] for w in per_window]).astype(np.float32)
            batch[f"usage_alpha_{head_id}"] = np.stack(
                [w["usage_alpha"] for w in per_window]).astype(np.float32)
        if JUNCTION_KIND in self.head_ids:
            batch[f"targets_{self.head_ids[JUNCTION_KIND]}"] = np.stack(
                [w["junction_matrix"] for w in per_window]).astype(np.float32)
            batch["splice_site_positions"] = np.stack(
                [w["junction_positions"] for w in per_window]).astype(np.int32)

        if self._emit_raw_junction_events:
            rows = self.junction_event_rows(max(len(w["junction_d_rel"]) for w in per_window))
            n_windows, n_samples = len(per_window), len(self._all_juncs)
            d_rel = np.full((n_windows, rows), -1, dtype=np.int32)
            a_rel = np.full((n_windows, rows), -1, dtype=np.int32)
            is_pos_strand = np.zeros((n_windows, rows), dtype=bool)
            counts = np.zeros((n_windows, rows, n_samples), dtype=np.float32)
            for i, w in enumerate(per_window):
                m = len(w["junction_d_rel"])
                d_rel[i, :m] = w["junction_d_rel"]
                a_rel[i, :m] = w["junction_a_rel"]
                is_pos_strand[i, :m] = w["junction_is_pos_strand"]
                counts[i, :m] = w["junction_counts"]
            batch.update(
                junction_d_rel=d_rel, junction_a_rel=a_rel,
                junction_is_pos_strand=is_pos_strand, junction_counts=counts,
            )
