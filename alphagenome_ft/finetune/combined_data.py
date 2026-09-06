"""Joint rna_seq + 3-splice-head data loading for the finetuning driver.

Combines ``alphagenome_ft.finetune.data.BigWigDataModule`` (rna_seq) and
``alphagenome_ft.finetune.splice_data.SpliceDataModule`` (splice_site,
splice_usage, splice_junctions) into one joint-modality batch per step,
matching a PyTorch run's ``--modality bigwig ... --modality splicing ...``
(all 4 heads trained together).
"""

from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np


def load_interval_list(bed_path: Path, window_size: int):
    """Load a plain 3-column (chrom, start, end) BED into genome.Interval list.

    Unlike ``alphagenome_ft.finetune.data.load_intervals_from_bed`` (which
    expects one combined BED with a 4th train/valid/test split column), fold
    BEDs already split into one file per split (no split column) would
    silently drop every row (len(parts) < 4) if pointed at that loader
    directly.
    """
    from alphagenome_ft.finetune.data import build_interval

    intervals = []
    opened = gzip.open if str(bed_path).endswith(".gz") else open
    with opened(bed_path, "rt") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            chrom, start_str, end_str = line.split()[:3]
            intervals.append(build_interval(
                chromosome=chrom,
                start=int(float(start_str)),
                end=int(float(end_str)),
                window_size=window_size,
            ))
    if not intervals:
        raise ValueError(f"No intervals parsed from {bed_path}.")
    return intervals


def compute_track_means(
    bigwig_files, bed_path: Path, sequence_length: int, max_samples: int | None,
) -> list[float]:
    """Compute nonzero_mean per rna_seq track, matching alphagenome-pytorch's
    datasets.py::compute_track_means (same centering/expansion logic, same
    deterministic every-Nth subsetting, same nonzero-mean formula,
    resolution 1) — ported line-for-line rather than approximated, since this
    directly feeds a real, active part of training dynamics (the predefined
    rna_seq head rescales predictions/targets by this on every forward pass).

    Unlike PyTorch's version this has no strand_pair_groups support: no
    caller here passes strand pairs for the rna_seq modality.
    """
    import pyBigWig

    raw_intervals = []
    opened = gzip.open if str(bed_path).endswith(".gz") else open
    with opened(bed_path, "rt") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            chrom, start_str, end_str = line.split()[:3]
            raw_intervals.append((chrom, int(float(start_str)), int(float(end_str))))

    bws = [pyBigWig.open(str(p)) for p in bigwig_files]
    n_tracks = len(bws)
    try:
        chrom_sizes = dict(bws[0].chroms())
        half_len = sequence_length // 2
        valid_positions = []
        for chrom, start, end in raw_intervals:
            if chrom not in chrom_sizes:
                continue
            center = (start + end) // 2
            final_start = center - half_len
            final_end = center + half_len
            if final_start < 0 or final_end > chrom_sizes[chrom]:
                continue
            valid_positions.append((chrom, final_start, final_end))

        if max_samples is not None and len(valid_positions) > max_samples:
            step = len(valid_positions) // max_samples
            valid_positions = valid_positions[::step][:max_samples]

        if not valid_positions:
            raise ValueError("No valid positions found for computing track means.")

        sums = np.zeros(n_tracks, dtype=np.float64)
        counts = np.zeros(n_tracks, dtype=np.int64)
        for chrom, start, end in valid_positions:
            for i, bw in enumerate(bws):
                values = bw.values(chrom, start, end, numpy=True)
                values = np.nan_to_num(np.asarray(values, dtype=np.float32), nan=0.0)
                nonzero = values[values != 0]
                sums[i] += nonzero.sum()
                counts[i] += len(nonzero)
    finally:
        for bw in bws:
            bw.close()

    means = np.where(counts > 0, sums / counts, 1.0)
    print(f"Computed nonzero_mean per rna_seq track ({len(valid_positions)} "
          f"sampled windows): {means}", flush=True)
    return means.tolist()


class CombinedDataModule:
    """Zips a BigWigDataModule (rna_seq) and a SpliceDataModule (3 splice
    heads) into one joint-modality batch per step.

    ``train()`` only ever reads ``_intervals``/``_batch_size``/``_drop_last``
    and calls ``iter_batches`` on whatever data_module it's given (see
    ``alphagenome_ft.finetune.train.train``) - this wrapper needs no changes
    to either underlying data module.

    Safety: both underlying modules must be constructed from the identical
    window list/order (identical-by-construction window lists/order/seed
    make both modules' independent index shuffles align in lock-step).
    Rather than just trust that, ``iter_batches`` asserts the two modules'
    ``sequences`` arrays are byte-identical every single batch - same
    windows extracted via the same FASTA must produce the same encoded
    sequence, so any mismatch (a future alphagenome_ft change reordering
    internally, a filtering difference introduced later, etc.) surfaces
    immediately as a loud error instead of silently training on misaligned
    targets.
    """

    def __init__(self, bigwig_module, splice_module):
        for split in ("train", "valid"):
            n_bw = len(bigwig_module._intervals.get(split, ()))
            n_sp = len(splice_module._intervals.get(split, ()))
            if n_bw != n_sp:
                raise ValueError(
                    f"CombinedDataModule: {split} window count mismatch "
                    f"(bigwig={n_bw}, splice={n_sp}) - the two modules were "
                    f"not built from the same window list, so their "
                    f"per-batch shuffles cannot be assumed to align."
                )
        self._bigwig = bigwig_module
        self._splice = splice_module
        self._intervals = splice_module._intervals
        self._batch_size = splice_module._batch_size
        self._drop_last = splice_module._drop_last

    def iter_batches(self, split: str, *, seed: int | None = None, skip_batches: int = 0):
        for bw_batch, sp_batch in zip(
            self._bigwig.iter_batches(split, seed=seed, skip_batches=skip_batches),
            self._splice.iter_batches(split, seed=seed, skip_batches=skip_batches),
        ):
            if not np.array_equal(bw_batch["sequences"], sp_batch["sequences"]):
                raise RuntimeError(
                    "CombinedDataModule: bigwig and splice batches disagree on "
                    "'sequences' for the same batch index - the two data "
                    "modules' window order has desynchronized. Refusing to "
                    "train on what would be misaligned targets."
                )
            combined = dict(sp_batch)
            combined["targets_rna_seq"] = bw_batch["targets_rna_seq"]
            yield combined
