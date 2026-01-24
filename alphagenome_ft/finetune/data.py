"""Data utilities for fine-tuning."""

from __future__ import annotations

import contextlib
import gzip
from collections import defaultdict
from pathlib import Path
from typing import Iterator, Mapping, Sequence

import numpy as np
import pyBigWig
from alphagenome.data import genome
from alphagenome_research.io import fasta as fasta_lib
from alphagenome_research.model import one_hot_encoder

from alphagenome_ft.finetune.config import HeadSpec


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
        window_size: int | None,
        drop_last: bool = False,
    ) -> None:
        self._intervals = intervals
        self._fasta_path = fasta_path
        self._head_specs = head_specs
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._window_size = window_size
        self._drop_last = drop_last
        self._encoder = one_hot_encoder.DNAOneHotEncoder(dtype=np.float32)

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
                head_handles[spec.head_name] = handles

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
        targets: dict[str, list[np.ndarray]] = {spec.head_name: [] for spec in self._head_specs}

        for idx in batch_indices:
            window = windows[idx]
            seq = extractor.extract(window)
            encoded = self._encoder.encode(seq)
            sequences.append(encoded)

            seq_len = encoded.shape[0]
            for spec in self._head_specs:
                channel_arrays = []
                for handle in head_handles[spec.head_name]:
                    values = handle.values(window.chromosome, window.start, window.end)
                    track = self._prepare_track(values, seq_len)
                    channel_arrays.append(track)
                targets[spec.head_name].append(np.stack(channel_arrays, axis=-1))

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


def load_intervals(args) -> Mapping[str, list[genome.Interval]]:
    splits: dict[str, list[genome.Interval]] = defaultdict(list)
    opened = gzip.open if args.bed.suffix == '.gz' else open
    mode = 'rt'
    with opened(args.bed, mode) as handle:
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
                window_size=args.window_size,
            )
            splits[label].append(interval)

    if not splits['train']:
        raise ValueError('No training intervals found in --bed file.')

    _maybe_limit(splits['train'], args.limit_train)
    _maybe_limit(splits['valid'], args.limit_valid)
    _maybe_limit(splits['test'], args.limit_test)

    for key in list(splits.keys()):
        if not splits[key]:
            splits.pop(key)
    return splits


def build_interval(
    *, chromosome: str, start: int, end: int, window_size: int | None
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


def prepare_batch_for_jax(
    batch: Mapping[str, np.ndarray],
    organism_index_value: int,
    head_names: Sequence[str],
):
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
