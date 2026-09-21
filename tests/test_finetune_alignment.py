"""Parity checks for JAX/PyTorch fine-tuning data semantics."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from alphagenome.models import dna_client, dna_output

from alphagenome_ft.finetune import config as config_module
from alphagenome_ft.finetune import data as data_module
from alphagenome_ft.finetune.config import TrackInfo


def test_track_metadata_keeps_means_and_infers_strands():
    tracks = [
        TrackInfo("sample_forward", Path("forward.bw"), 2.5),
        TrackInfo("sample_reverse", Path("reverse.bw"), 2.5),
        TrackInfo("unstranded", Path("atac.bw"), 4.0),
    ]
    metadata = config_module._build_track_metadata(
        tracks,
        dna_client.Organism.HOMO_SAPIENS,
        dna_output.OutputType.RNA_SEQ,
    )
    frame = metadata[dna_client.Organism.HOMO_SAPIENS].get(
        dna_output.OutputType.RNA_SEQ
    )
    assert frame["strand"].tolist() == ["+", "-", "."]
    assert frame["nonzero_mean"].tolist() == [2.5, 2.5, 4.0]


def test_nonzero_means_and_strand_pair_averaging(monkeypatch):
    values_by_path = {
        "forward.bw": np.array([0.0, 2.0, np.nan, 4.0], dtype=np.float32),
        "reverse.bw": np.array([0.0, 1.0, 3.0, 0.0], dtype=np.float32),
    }

    class FakeBigWig:
        def __init__(self, path):
            self.path = path

        def values(self, *_args, **_kwargs):
            return values_by_path[self.path]

        def close(self):
            pass

    monkeypatch.setattr(data_module.pyBigWig, "open", FakeBigWig)
    intervals = [SimpleNamespace(chromosome="chr1", start=0, end=4)]
    means = data_module.compute_track_nonzero_means(
        [Path("forward.bw"), Path("reverse.bw")],
        intervals,
        strand_pair_groups=[(0, 1)],
    )
    # Individual means are 3 and 2; paired strands share their average.
    np.testing.assert_allclose(means, [2.5, 2.5])


def test_validation_keeps_bed_order_with_partial_batch(monkeypatch):
    """The final singleton affects a mean-of-batch-means validation loss."""
    module = object.__new__(data_module.BigWigDataModule)
    module._intervals = {"valid": list(range(5)), "train": list(range(5))}
    module._shuffle = True
    module._batch_size = 2
    module._drop_last = False
    module._head_specs = []
    module._fasta_path = Path("unused.fa")
    monkeypatch.setattr(data_module.fasta_lib, "FastaExtractor", lambda _: None)
    module._make_batch = lambda indices, *_: list(indices)
    assert list(module.iter_batches("valid", seed=7)) == [[0, 1], [2, 3], [4]]
    assert list(module.iter_batches("train", seed=7)) != [[0, 1], [2, 3], [4]]


def test_local_driver_skips_boundary_windows(tmp_path):
    import importlib.util

    path = Path(__file__).parents[1] / "scripts" / "finetune_1mb_lora_head.py"
    spec = importlib.util.spec_from_file_location("lora_driver", path)
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    fasta = tmp_path / "genome.fa"
    Path(f"{fasta}.fai").write_text("chr1\t100\t0\t100\t101\n")
    for split in ("train", "valid", "test"):
        (tmp_path / f"{split}.bed").write_text(
            "chr1\t0\t2\nchr1\t40\t42\nchr1\t98\t100\nchrMissing\t40\t42\n"
        )
    args = SimpleNamespace(
        fasta=fasta, split_dir=tmp_path, window_size=10,
        limit_train=None, limit_valid=None, limit_test=None,
    )
    splits = driver.load_local_splits(args)
    for rows in splits.values():
        assert [(iv.chromosome, iv.start, iv.end) for iv in rows] == [("chr1", 36, 46)]


def test_pytorch_head_initialization_uses_absolute_truncation_bounds():
    import haiku as hk
    import jax
    import jax.numpy as jnp
    from alphagenome_research.model import heads
    from alphagenome_ft.custom_model import _pytorch_head_parameter_creator

    def forward(x):
        with hk.name_scope("head/task/resolution_1"):
            with hk.custom_creator(_pytorch_head_parameter_creator):
                return heads._MultiOrganismLinear(16, 1)(x, jnp.array([0]))

    params = hk.transform(forward).init(jax.random.PRNGKey(42), jnp.ones((1, 2, 1536)))
    weights = next(v["w"] for v in params.values() if "w" in v)
    std = 1 / np.sqrt(1536)
    # Native Haiku initialization cannot produce any |w| > 2 * std.
    assert np.abs(weights).max() > 3 * std
    np.testing.assert_allclose(np.std(weights), std, rtol=0.03)
