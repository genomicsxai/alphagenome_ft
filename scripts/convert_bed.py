#!/usr/bin/env python3
"""Convert Borzoi BED windows into AlphaGenome-compatible 1 Mb intervals."""

from __future__ import annotations

import argparse
import gzip
from bisect import bisect_left
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

WINDOW_SIZE = 1_048_576  # 1 Mbp windows expected by AlphaGenome

# https://hgdownload.cse.ucsc.edu/goldenpath/hg38/bigZips/hg38.chrom.sizes
HG38_CHROM2SIZE = {
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
    "chrX":  156_040_895,
}

FOLD_MAPPING = {
    '0': {
        'train': ['fold2', 'fold3', 'fold4', 'fold5', 'fold6', 'fold7'],
        'valid': ['fold1'],
        'test': ['fold0'],
    },
    '1': {
        'train': ['fold0', 'fold3', 'fold4', 'fold5', 'fold6', 'fold7'],
        'valid': ['fold2'],
        'test': ['fold1'],
    },
    '2': {
        'train': ['fold0', 'fold1', 'fold4', 'fold5', 'fold6', 'fold7'],
        'valid': ['fold3'],
        'test': ['fold2'],
    },
    '3': {
        'train': ['fold0', 'fold1', 'fold2', 'fold5', 'fold6', 'fold7'],
        'valid': ['fold4'],
        'test': ['fold3'],
    },
}


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Expand 196,608 bp Borzoi intervals to 1,048,576 bp AlphaGenome windows"
            " and remap fold labels to train/valid/test splits."
        )
    )
    parser.add_argument("inbed", type=Path, help="Input path to the Borzoi BED file.")
    parser.add_argument("outbed", type=Path, help="Output path for the AlphaGenome BED file.")
    parser.add_argument(
        "--fold",
        required=True,
        choices=sorted(FOLD_MAPPING.keys()),
        help="Fold to build the bed file for (determines train/valid/test split mapping).",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=WINDOW_SIZE,
        help="Target window size (defaults to AlphaGenome's 1,048,576 bp).",
    )
    return parser.parse_args(argv)


def build_split_lookup(fold_key: str) -> Dict[str, str]:
    split_lookup: Dict[str, str] = {}
    mapping = FOLD_MAPPING[fold_key]
    for split_name, fold_list in mapping.items():
        for fold_label in fold_list:
            split_lookup[fold_label] = split_name
    return split_lookup


def expand_interval(start: int, end: int, *, window_size: int, chrom_size: int | None) -> Tuple[int, int]:
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


def _has_training_overlap(
    chrom: str,
    start: int,
    end: int,
    training_intervals: Dict[str, Tuple[Sequence[Tuple[int, int]], Sequence[int]]],
) -> bool:
    record = training_intervals.get(chrom)
    if not record:
        return False

    intervals, starts = record
    idx = bisect_left(starts, start)

    def overlaps(interval: Tuple[int, int]) -> bool:
        return interval[0] < end and interval[1] > start

    if idx < len(intervals) and overlaps(intervals[idx]):
        return True
    if idx > 0 and overlaps(intervals[idx - 1]):
        return True
    return False


def convert_bed(in_bed: Path, out_bed: Path, fold: str, window_size: int) -> None:
    split_lookup = build_split_lookup(fold)
    windows: List[Tuple[str, int, int, str]] = []

    def _open(path: Path, mode: str):
        if path.suffix == ".gz":
            return gzip.open(path, mode + "t")
        return path.open(mode)

    with _open(in_bed, "r") as src:
        for line_no, raw_line in enumerate(src, start=1):
            stripped = raw_line.strip()
            if not stripped or stripped.startswith("#"):
                continue

            parts = stripped.split()
            if len(parts) < 4:
                raise ValueError(f"Line {line_no}: expected at least 4 columns, got {len(parts)}")

            chrom, start_str, end_str, fold_label = parts[:4]
            try:
                start = int(start_str)
                end = int(end_str)
            except ValueError as exc:
                raise ValueError(f"Line {line_no}: start/end columns must be integers.") from exc

            split = split_lookup.get(fold_label)
            if split is None:
                raise ValueError(f"Line {line_no}: fold label '{fold_label}' is not recognized for fold {fold}.")

            chrom_size = HG38_CHROM2SIZE.get(chrom)
            new_start, new_end = expand_interval(start, end, window_size=window_size, chrom_size=chrom_size)
            windows.append((chrom, new_start, new_end, split))

    training_by_chrom: Dict[str, List[Tuple[int, int]]] = {}
    for chrom, start, end, split in windows:
        if split == "train":
            training_by_chrom.setdefault(chrom, []).append((start, end))

    training_lookup: Dict[str, Tuple[Sequence[Tuple[int, int]], Sequence[int]]] = {}
    for chrom, intervals in training_by_chrom.items():
        intervals.sort(key=lambda iv: iv[0])
        starts = [iv[0] for iv in intervals]
        training_lookup[chrom] = (intervals, starts)

    with _open(out_bed, "w") as dst:
        for chrom, start, end, split in windows:
            if split in {"valid", "test"} and _has_training_overlap(chrom, start, end, training_lookup):
                continue
            dst.write(f"{chrom}\t{start}\t{end}\t{split}\n")


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    convert_bed(args.inbed, args.outbed, args.fold, args.window_size)


if __name__ == "__main__":
    main()
