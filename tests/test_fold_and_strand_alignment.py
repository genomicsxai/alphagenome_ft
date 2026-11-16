"""Regression coverage for model folds, held-out partitions, and track strands."""
import ast
from pathlib import Path

import numpy as np
import pytest
from alphagenome.models import dna_client, dna_output
from alphagenome_ft.finetune.config import TrackInfo, _build_track_metadata, _parse_targets
from alphagenome_ft.finetune.data import FOLD_MAPPING, build_split_lookup, get_fold_split

EXPECTED = [(0, 1), (3, 4), (2, 5), (6, 7)]


@pytest.mark.parametrize('model_fold,held_out', enumerate(EXPECTED))
def test_each_model_uses_the_expected_partitions(tmp_path, model_fold, held_out):
    valid, test = (f'fold{i}' for i in held_out)
    lookup = build_split_lookup(f'FOLD_{model_fold}')
    assert lookup[valid] == 'valid'
    assert lookup[test] == 'test'
    assert len(lookup) == 8
    assert sum(split == 'train' for split in lookup.values()) == 6
    bed = tmp_path / 'partitions.bed'
    bed.write_text(''.join(f'chr1\t{1000 + i * 1000}\t{1100 + i * 1000}\tfold{i}\n' for i in range(8)))
    rows = get_fold_split(f'fold_{model_fold}', window_size=100, bed_path=str(bed))
    assert len(rows) == 8
    assert rows['split'].tolist() == [lookup[f'fold{i}'] for i in range(8)]


def test_mapping_matches_workspace_borzoi_converter():
    converter = Path(__file__).resolve().parents[2] / 'alphagenome-pytorch/scripts/convert_borzoi_folds.py'
    if not converter.exists():
        pytest.skip('Sibling Borzoi converter is not installed')
    node = next(n for n in ast.parse(converter.read_text()).body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == 'ALPHAGENOME_FOLDS' for t in n.targets))
    reference = ast.literal_eval(node.value)
    for fold, mapping in FOLD_MAPPING.items():
        assert {s: mapping[s][0] for s in ('valid', 'test')} == reference[f'FOLD_{fold}']


def test_invalid_model_fold_rejected():
    with pytest.raises(ValueError, match='Invalid model fold'):
        build_split_lookup('fold_4')


def test_explicit_and_inferred_strands_reindex_correctly(tmp_path):
    path = tmp_path / 'track.bw'
    path.touch()
    tracks = _parse_targets([
        {'path': path, 'label': 'override_reverse', 'strand': '+'},
        {'path': path, 'label': 'override_forward', 'strand': '-'},
        {'path': path, 'label': 'LCL.100M.+'},
        {'path': path, 'label': 'LCL.100M.-'},
        {'path': path, 'label': 'SRR17111303+'},
        {'path': path, 'label': 'SRR17111303-'},
        {'path': path, 'label': 'ATAC'},
        {'path': path, 'label': 'unstranded_forward', 'strand': '.'},
    ])
    meta = _build_track_metadata(tracks, dna_client.Organism.HOMO_SAPIENS,
                                 dna_output.OutputType.RNA_SEQ)[dna_client.Organism.HOMO_SAPIENS]
    assert meta.rna_seq.strand.tolist() == ['+', '-', '+', '-', '+', '-', '.', '.']
    np.testing.assert_array_equal(
        meta.strand_reindexing[dna_output.OutputType.RNA_SEQ], [1, 0, 3, 2, 5, 4, 6, 7])


def test_unstranded_tracks_are_not_labeled_positive(tmp_path):
    path = tmp_path / 'track.bw'
    path.touch()
    tracks = _parse_targets([{'path': path, 'label': 'K562_ATAC'}, {'path': path, 'label': 'GM12878_ATAC'}])
    meta = _build_track_metadata(tracks, dna_client.Organism.HOMO_SAPIENS,
                                 dna_output.OutputType.ATAC)[dna_client.Organism.HOMO_SAPIENS]
    assert meta.atac.strand.tolist() == ['.', '.']
    np.testing.assert_array_equal(meta.strand_reindexing[dna_output.OutputType.ATAC], [0, 1])


def test_invalid_and_unpaired_strands_fail_early():
    with pytest.raises(ValueError, match='Invalid strand'):
        TrackInfo('track', Path('track.bw'), strand='unknown')
    with pytest.raises(ValueError, match='equal numbers'):
        _build_track_metadata([TrackInfo('sample_forward', Path('track.bw'))],
                              dna_client.Organism.HOMO_SAPIENS, dna_output.OutputType.RNA_SEQ)
