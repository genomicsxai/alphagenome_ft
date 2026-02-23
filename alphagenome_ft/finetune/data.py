"""Data utilities for fine-tuning."""

from __future__ import annotations

import contextlib
import gzip
from collections import defaultdict
from pathlib import Path
from bisect import bisect_left
from typing import Iterator, Mapping, Sequence

import numpy as np
import pandas as pd
import pyBigWig
from alphagenome.data import genome
from alphagenome_research.io import fasta as fasta_lib
from alphagenome_research.model import one_hot_encoder


from alphagenome_ft.finetune.config import HeadSpec

WINDOW_SIZE = 1_048_576  # 1 Mbp windows

_DEFAULT_INTERVALS = {
    "HOMO_SAPIENS": (
        "https://github.com/calico/borzoi/raw/"
        "5c9358222b5026abb733ed5fb84f3f6c77239b37/data/sequences_human.bed.gz"
    ),
    "MUS_MUSCULUS": (
        "https://github.com/calico/borzoi/raw/"
        "5c9358222b5026abb733ed5fb84f3f6c77239b37/data/sequences_mouse.bed.gz"
    ),
}

_ORGANISM_ALIASES = {
    "human": "HOMO_SAPIENS",
    "homo_sapiens": "HOMO_SAPIENS",
    "homo-sapiens": "HOMO_SAPIENS",
    "homo sapiens": "HOMO_SAPIENS",
    "hg38": "HOMO_SAPIENS",

    "mouse": "MUS_MUSCULUS",
    "mus_musculus": "MUS_MUSCULUS",
    "mus-musculus": "MUS_MUSCULUS",
    "mus musculus": "MUS_MUSCULUS",
    "mm10": "MUS_MUSCULUS",
}

# https://hgdownload.cse.ucsc.edu/goldenpath/hg38/bigZips/hg38.chrom.sizes
# https://hgdownload.cse.ucsc.edu/goldenpath/mm10/bigZips/mm10.chrom.sizes
_CHROMSIZES = {
    "HOMO_SAPIENS": {
        "chr1": 248_956_422,
        "chr2": 242_193_529,
        "chr3": 198_295_559,
        "chr4": 190_214_555,
        "chr5": 181_538_259,
        "chr6": 170_805_979,
        "chr7": 159_345_973,
        "chr8": 145_138_636,
        "chr9": 138_394_717,
        "chr10": 133_797_422,
        "chr11": 135_086_622,
        "chr12": 133_275_309,
        "chr13": 114_364_328,
        "chr14": 107_043_718,
        "chr15": 101_991_189,
        "chr16": 90_338_345,
        "chr17": 83_257_441,
        "chr18": 80_373_285,
        "chr19": 58_617_616,
        "chr20": 64_444_167,
        "chr21": 46_709_983,
        "chr22": 50_818_468,
        "chrX": 156_040_895,
        "chrY": 57_227_415
    },
    "MUS_MUSCULUS": {
        "chr1": 195_471_971,
        "chr2": 182_113_224,
        "chr3": 160_039_680,
        "chr4": 156_508_116,
        "chr5": 151_834_684,
        "chr6": 149_736_546,
        "chr7": 145_441_459,
        "chr8": 129_401_213,
        "chr9": 124_595_110,
        "chr10": 130_694_993,
        "chr11": 122_082_543,
        "chr12": 120_129_022,
        "chr13": 120_421_639,
        "chr14": 124_902_244,
        "chr15": 104_043_685,
        "chr16": 98_207_768,
        "chr17": 94_987_271,
        "chr18": 90_702_639,
        "chr19": 61_431_566,
        "chrX": 171_031_299,
        "chrY": 91_744_698,
    },
}

FOLD_MAPPING = {
    "0": {
        "train": ["fold2", "fold3", "fold4", "fold5", "fold6", "fold7"],
        "valid": ["fold1"],
        "test": ["fold0"],
    },
    "1": {
        "train": ["fold0", "fold3", "fold4", "fold5", "fold6", "fold7"],
        "valid": ["fold2"],
        "test": ["fold1"],
    },
    "2": {
        "train": ["fold0", "fold1", "fold4", "fold5", "fold6", "fold7"],
        "valid": ["fold3"],
        "test": ["fold2"],
    },
    "3": {
        "train": ["fold0", "fold1", "fold2", "fold5", "fold6", "fold7"],
        "valid": ["fold4"],
        "test": ["fold3"],
    },
}


def build_split_lookup(fold_key: str) -> dict[str, str]:
    split_lookup: dict[str, str] = {}
    mapping = FOLD_MAPPING[fold_key]
    for split_name, fold_list in mapping.items():
        for fold_label in fold_list:
            split_lookup[fold_label] = split_name
    return split_lookup


def expand_interval(
    start: int,
    end: int,
    *,
    window_size: int,
    chrom_size: int | None,
) -> tuple[int, int]:
    length = end - start
    if length <= 0:
        raise ValueError(f"Invalid interval with non-positive length: start={start}, end={end}")

    midpoint = start + length // 2
    half_window = window_size // 2
    new_start = midpoint - half_window
    new_end = new_start + window_size

    if new_start < 0:
        new_start = 0
        new_end = window_size

    if chrom_size is not None and new_end > chrom_size:
        new_end = chrom_size
        new_start = new_end - window_size
        if new_start < 0:
            raise ValueError(
                "Chromosome length is shorter than the requested window size; cannot place interval."
            )

    return new_start, new_end


def _normalize_fold_label(fold_label: object) -> str:
    label = str(fold_label).strip()
    if label.startswith("fold"):
        return label
    return f"fold{label}"


def _normalize_organism(organism: str) -> str:
    raw = str(organism).strip()
    if raw in _DEFAULT_INTERVALS:
        return raw
    key = raw.lower()
    return _ORGANISM_ALIASES.get(key, raw)


def _has_training_overlap(
    chrom: str,
    start: int,
    end: int,
    training_intervals: dict[str, tuple[Sequence[tuple[int, int]], Sequence[int]]],
) -> bool:
    record = training_intervals.get(chrom)
    if not record:
        return False

    intervals, starts = record
    idx = bisect_left(starts, start)

    def overlaps(interval: tuple[int, int]) -> bool:
        return interval[0] < end and interval[1] > start

    if idx < len(intervals) and overlaps(intervals[idx]):
        return True
    if idx > 0 and overlaps(intervals[idx - 1]):
        return True
    return False


def get_fold_intervals(
    fold: str | int,
    window_size: int = WINDOW_SIZE,
    organism: str = "HOMO_SAPIENS",
    bed_path: str | None = None,
) -> pd.DataFrame:
    """Create train/valid/test windows for a Borzoi-style fold split.

    Args:
        fold: Fold identifier (``0``-``3``) used with ``FOLD_MAPPING``.
        window_size: Final window length centered on each source interval.
        organism: Organism key or alias (for example ``HOMO_SAPIENS`` or ``hg38``).
        bed_path: Optional BED/BED.GZ path containing ``chrom start end fold``.
            If omitted, uses a built-in default BED for the selected organism.

    Returns:
        DataFrame with columns ``chromosome``, ``start``, ``end``, ``split``.
        Validation/test windows that overlap any training window are removed.

    Raises:
        ValueError: If fold/organism labels are invalid, BED content is empty,
            or fold labels in the BED do not match the selected mapping.
    """
    organism = _normalize_organism(organism)

    fold_key = str(fold)
    if fold_key not in FOLD_MAPPING:
        valid = ", ".join(sorted(FOLD_MAPPING))
        raise ValueError(f"Invalid fold '{fold}'. Valid folds: {valid}")

    if bed_path is None:
        if organism not in _DEFAULT_INTERVALS:
            valid = ", ".join(sorted(_DEFAULT_INTERVALS))
            raise ValueError(f"Unknown organism '{organism}'. Valid values: {valid}")
        bed_path = _DEFAULT_INTERVALS[organism]

    regions = pd.read_csv(
        bed_path,
        sep="\t",
        names=["chromosome", "start", "end", "fold"],
        comment="#",
    )
    if regions.empty:
        raise ValueError("Input BED did not contain any intervals.")

    split_lookup = build_split_lookup(fold_key)

    windows: list[tuple[str, int, int, str]] = []
    for _, row in regions.iterrows():
        chrom = str(row["chromosome"])
        start = int(row["start"])
        end = int(row["end"])
        fold_label = _normalize_fold_label(row["fold"])

        split = split_lookup.get(fold_label)
        if split is None:
            raise ValueError(
                f"Input fold label '{fold_label}' is not recognized for selected fold {fold_key}."
            )

        chrom_size = _CHROMSIZES[organism].get(chrom)
        new_start, new_end = expand_interval(
            start,
            end,
            window_size=window_size,
            chrom_size=chrom_size,
        )
        windows.append((chrom, new_start, new_end, split))

    training_by_chrom: dict[str, list[tuple[int, int]]] = {}
    for chrom, start, end, split in windows:
        if split == "train":
            training_by_chrom.setdefault(chrom, []).append((start, end))

    training_lookup: dict[str, tuple[Sequence[tuple[int, int]], Sequence[int]]] = {}
    for chrom, intervals in training_by_chrom.items():
        intervals.sort(key=lambda iv: iv[0])
        starts = [iv[0] for iv in intervals]
        training_lookup[chrom] = (intervals, starts)

    filtered: list[tuple[str, int, int, str]] = []
    for chrom, start, end, split in windows:
        if split in {"valid", "test"} and _has_training_overlap(chrom, start, end, training_lookup):
            continue
        filtered.append((chrom, start, end, split))

    num_filtered = len(windows) - len(filtered)
    if num_filtered > 0:
        print(f"Filtered out {num_filtered} intervals from valid/test sets due to training overlap.")

    return pd.DataFrame(filtered, columns=["chromosome", "start", "end", "split"])


class BigWigDataModule:
    """Creates training batches by streaming sequences + BigWig targets."""

    def __init__(
        self,
        *,
        intervals: Mapping[str, Sequence[genome.Interval]],
        fasta_path: Path,
        head_specs: Sequence[HeadSpec],
        batch_size: int,
        shuffle: bool,
        drop_last: bool = False,
    ) -> None:
        """Initialize streaming sequence/BigWig batch generation.

        Args:
            intervals: Split-to-interval mapping (``train``/``valid``/``test``).
            fasta_path: Reference FASTA used to extract sequence windows.
            head_specs: Head definitions including target BigWig track paths.
            batch_size: Number of windows per yielded batch.
            shuffle: Whether to shuffle window order in ``iter_batches``.
            drop_last: If True, drop incomplete final batches.

        Raises:
            ValueError: If no shared chromosomes exist across configured BigWigs,
                or no training intervals remain after chromosome filtering.
        """
        filtered_intervals = self._filter_intervals_by_bigwig_chromosomes(intervals, head_specs)
        if not filtered_intervals.get("train"):
            raise ValueError(
                "No training intervals remain after filtering to chromosomes present in all BigWigs."
            )
        self._intervals = filtered_intervals
        self._fasta_path = fasta_path
        self._head_specs = head_specs
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._drop_last = drop_last
        self._encoder = one_hot_encoder.DNAOneHotEncoder(dtype=np.float32)

    @staticmethod
    def _get_common_bigwig_chromosomes(
        head_specs: Sequence[HeadSpec],
    ) -> set[str]:
        common_chroms: set[str] | None = None
        for spec in head_specs:
            for track in spec.tracks:
                with pyBigWig.open(str(track.path)) as handle:
                    chrom_dict = handle.chroms()
                if not chrom_dict:
                    raise ValueError(f"BigWig has no chromosome index: {track.path}")
                track_chroms = set(chrom_dict.keys())
                if common_chroms is None:
                    common_chroms = track_chroms
                else:
                    common_chroms &= track_chroms
                if not common_chroms:
                    return set()
        return common_chroms or set()

    @classmethod
    def _filter_intervals_by_bigwig_chromosomes(
        cls,
        intervals: Mapping[str, Sequence[genome.Interval]],
        head_specs: Sequence[HeadSpec],
    ) -> dict[str, list[genome.Interval]]:
        common_chroms = cls._get_common_bigwig_chromosomes(head_specs)
        if not common_chroms:
            raise ValueError("No shared chromosomes were found across configured BigWig tracks.")

        filtered: dict[str, list[genome.Interval]] = {}
        removed_chroms: set[str] = set()
        removed_counts: dict[str, int] = defaultdict(int)
        total_removed = 0

        for split, split_intervals in intervals.items():
            kept: list[genome.Interval] = []
            for interval in split_intervals:
                if interval.chromosome in common_chroms:
                    kept.append(interval)
                else:
                    removed_chroms.add(interval.chromosome)
                    removed_counts[split] += 1
                    total_removed += 1
            if kept:
                filtered[split] = kept

        if total_removed > 0:
            removed = ", ".join(sorted(removed_chroms))
            split_counts = ", ".join(
                f"{split}={removed_counts[split]}"
                for split in ("train", "valid", "test")
                if removed_counts.get(split, 0) > 0
            )
            print(
                "Filtered out "
                f"{total_removed} intervals to match BigWig chromosome coverage "
                f"(removed chromosomes: {removed}; removed intervals: {split_counts})."
            )

        return filtered

    def iter_batches(
        self, split: str, *, seed: int | None = None
    ) -> Iterator[dict[str, np.ndarray]]:
        windows = list(self._intervals.get(split, ()))
        if not windows:
            return

        order = np.arange(len(windows))
        if self._shuffle:
            rng = np.random.default_rng(seed)
            rng.shuffle(order)

        extractor = fasta_lib.FastaExtractor(str(self._fasta_path))
        with contextlib.ExitStack() as stack:
            head_handles: dict[str, list[pyBigWig.pyBigWig]] = {}
            for spec in self._head_specs:
                handles = []
                for track in spec.tracks:
                    handles.append(stack.enter_context(pyBigWig.open(str(track.path))))
                head_handles[spec.head_id] = handles

            batch_indices: list[int] = []
            for idx in order:
                batch_indices.append(int(idx))
                if len(batch_indices) == self._batch_size:
                    yield self._make_batch(batch_indices, windows, extractor, head_handles)
                    batch_indices = []

            if batch_indices and not self._drop_last:
                yield self._make_batch(batch_indices, windows, extractor, head_handles)

    def _make_batch(
        self,
        batch_indices: Sequence[int],
        windows: Sequence[genome.Interval],
        extractor: fasta_lib.FastaExtractor,
        head_handles: Mapping[str, Sequence[pyBigWig.pyBigWig]],
    ) -> dict[str, np.ndarray]:
        sequences = []
        targets: dict[str, list[np.ndarray]] = {spec.head_id: [] for spec in self._head_specs}

        for idx in batch_indices:
            window = windows[idx]
            seq = extractor.extract(window)
            encoded = self._encoder.encode(seq)
            sequences.append(encoded)

            seq_len = encoded.shape[0]
            for spec in self._head_specs:
                channel_arrays = []
                for handle in head_handles[spec.head_id]:
                    values = handle.values(window.chromosome, window.start, window.end)
                    track = self._prepare_track(values, seq_len)
                    channel_arrays.append(track)
                targets[spec.head_id].append(np.stack(channel_arrays, axis=-1))

        batch = {
            'sequences': np.stack(sequences, axis=0).astype(np.float32),
            'negative_strand_mask': np.zeros((len(batch_indices),), dtype=bool),
        }
        for head_name, arrays in targets.items():
            batch[f'targets_{head_name}'] = np.stack(arrays, axis=0).astype(np.float32)
        return batch

    @staticmethod
    def _prepare_track(values: Sequence[float] | None, target_len: int) -> np.ndarray:
        if values is None:
            padded = np.zeros((target_len,), dtype=np.float32)
            return padded
        arr = np.nan_to_num(np.asarray(values, dtype=np.float32))
        if arr.shape[0] == target_len:
            return arr
        padded = np.zeros((target_len,), dtype=np.float32)
        limit = min(target_len, arr.shape[0])
        padded[:limit] = arr[:limit]
        return padded


def load_intervals_from_bed(
    bed: Path,
    window_size: int | None = None,
    *,
    limit_train: int | None = None,
    limit_valid: int | None = None,
    limit_test: int | None = None,
) -> Mapping[str, list[genome.Interval]]:
    """Load ``train``/``valid``/``test`` intervals from a BED or BED.GZ file.

    The BED is expected to provide at least four fields per row:
    ``chrom start end split``. Rows with unknown split labels are skipped.

    Args:
        bed: Path to input BED/BED.GZ.
        window_size: Optional target window size; if set, intervals are recentered
            and resized via ``build_interval``.
        limit_train: Optional maximum number of train intervals to keep.
        limit_valid: Optional maximum number of valid intervals to keep.
        limit_test: Optional maximum number of test intervals to keep.

    Returns:
        Mapping from split name to list of ``genome.Interval`` objects.

    Raises:
        ValueError: If no training intervals are present after parsing/limits.
    """
    splits: dict[str, list[genome.Interval]] = defaultdict(list)
    opened = gzip.open if bed.suffix == '.gz' else open
    mode = 'rt'
    with opened(bed, mode) as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            chrom, start_str, end_str, split = parts[:4]
            label = split.lower()
            if label not in {'train', 'valid', 'test'}:
                continue
            interval = build_interval(
                chromosome=chrom,
                start=int(float(start_str)),
                end=int(float(end_str)),
                window_size=window_size,
            )
            splits[label].append(interval)

    return _finalize_splits(
        splits,
        limit_train=limit_train,
        limit_valid=limit_valid,
        limit_test=limit_test,
        empty_train_error='No training intervals found in --bed file.',
    )


def load_intervals_from_dataframe(
    intervals_df: pd.DataFrame,
    window_size: int | None = None,
    *,
    limit_train: int | None = None,
    limit_valid: int | None = None,
    limit_test: int | None = None,
) -> Mapping[str, list[genome.Interval]]:
    """Load ``train``/``valid``/``test`` intervals from a DataFrame.

    Args:
        intervals_df: Input table. If named columns ``chromosome``, ``start``,
            ``end``, ``split`` are present, they are used. Otherwise, the first
            four columns are interpreted in that order.
        window_size: Optional target window size; if set, intervals are recentered
            and resized via ``build_interval``.
        limit_train: Optional maximum number of train intervals to keep.
        limit_valid: Optional maximum number of valid intervals to keep.
        limit_test: Optional maximum number of test intervals to keep.

    Returns:
        Mapping from split name to list of ``genome.Interval`` objects.

    Raises:
        ValueError: If fewer than four columns are provided or no training
            intervals remain.
    """
    splits: dict[str, list[genome.Interval]] = defaultdict(list)
    required_cols = {"chromosome", "start", "end", "split"}
    has_named_cols = required_cols.issubset(set(intervals_df.columns))
    if not has_named_cols and intervals_df.shape[1] < 4:
        raise ValueError(
            "intervals_df must include columns {chromosome,start,end,split} "
            "or have at least 4 columns in that order."
        )

    if has_named_cols:
        row_iter = (
            (
                str(getattr(row, "chromosome")),
                int(float(getattr(row, "start"))),
                int(float(getattr(row, "end"))),
                str(getattr(row, "split")).lower(),
            )
            for row in intervals_df.itertuples(index=False)
        )
    else:
        row_iter = (
            (
                str(row[0]),
                int(float(row[1])),
                int(float(row[2])),
                str(row[3]).lower(),
            )
            for row in intervals_df.iloc[:, :4].itertuples(index=False, name=None)
        )

    for chrom, start, end, label in row_iter:
        if label not in {'train', 'valid', 'test'}:
            continue
        interval = build_interval(
            chromosome=chrom,
            start=start,
            end=end,
            window_size=window_size,
        )
        splits[label].append(interval)

    return _finalize_splits(
        splits,
        limit_train=limit_train,
        limit_valid=limit_valid,
        limit_test=limit_test,
        empty_train_error='No training intervals found in generated fold intervals.',
    )


def build_interval(
    *, chromosome: str, start: int, end: int, window_size: int | None = None
) -> genome.Interval:
    if start >= end:
        raise ValueError(f'Invalid interval ({chromosome}, {start}, {end}).')
    if window_size is not None:
        center = (start + end) // 2
        half = window_size // 2
        start = max(0, center - half)
        end = start + window_size
    return genome.Interval(start=start, end=end, chromosome=chromosome)


def _maybe_limit(intervals: list[genome.Interval], limit: int | None) -> None:
    if limit is not None and len(intervals) > limit:
        del intervals[limit:]


def _finalize_splits(
    splits: dict[str, list[genome.Interval]],
    *,
    limit_train: int | None,
    limit_valid: int | None,
    limit_test: int | None,
    empty_train_error: str,
) -> Mapping[str, list[genome.Interval]]:
    if not splits['train']:
        raise ValueError(empty_train_error)

    _maybe_limit(splits['train'], limit_train)
    _maybe_limit(splits['valid'], limit_valid)
    _maybe_limit(splits['test'], limit_test)

    for key in list(splits.keys()):
        if not splits[key]:
            splits.pop(key)
    return splits


def prepare_batch(
    batch: Mapping[str, np.ndarray],
    organism_index_value: int,
    head_names: Sequence[str],
):
    """Convert a numpy batch to JAX arrays and attach organism/head fields.

    Args:
        batch: Batch mapping containing ``sequences``, ``negative_strand_mask``,
            and per-head arrays under ``targets_{head_name}``.
        organism_index_value: Integer organism index to broadcast across batch.
        head_names: Head names to extract from ``batch`` target keys.

    Returns:
        Mapping ready for model calls with JAX arrays:
        ``sequences``, ``organism_index``, ``negative_strand_mask``, and
        ``targets_{head_name}`` for each head.
    """
    import jax.numpy as jnp

    prepared = {
        'sequences': jnp.asarray(batch['sequences']),
        'organism_index': jnp.full(
            (batch['sequences'].shape[0],), organism_index_value, dtype=jnp.int32
        ),
        'negative_strand_mask': jnp.asarray(batch['negative_strand_mask']),
    }
    for head_name in head_names:
        prepared[f'targets_{head_name}'] = jnp.asarray(batch[f'targets_{head_name}'])
    return prepared
