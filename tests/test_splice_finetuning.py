"""Tests for splice fine-tuning: parsers, config wiring, and head construction.

Covers the three splice modalities (``splice_sites_classification``,
``splice_sites_usage``, ``splice_sites_junction``) added on top of the
already-working ``alphagenome_research`` splice heads. Parser/config tests
run without any pretrained checkpoint; the end-to-end ``train()`` smoke test
additionally requires Kaggle credentials (see ``tests/kaggle_util.py``) since
it downloads the real pretrained weights.
"""

from __future__ import annotations

from pathlib import Path

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
from alphagenome.data import genome

from alphagenome_ft import custom_heads as custom_heads_module
from alphagenome_ft.finetune import config as ft_config
from alphagenome_ft.finetune import star_junctions as sj
from alphagenome_ft.finetune.data import FinetuneDataModule
from alphagenome_ft.finetune.splice_data import SpliceTargets, build_splice_targets
from alphagenome_ft.finetune.splice_positions import build_junction_matrix, top_k_splice_positions

from tests.conftest import require_kaggle_credentials


SAMPLE1_SJ = "sample1.SJ.out.tab"
SAMPLE2_SJ = "sample2.SJ.out.tab"

# Two junctions, both on chr1 + strand: (1000, 2000) and (3000, 4000).
_SJ_ROWS = {
    SAMPLE1_SJ: [
        "chr1\t1000\t2000\t1\t1\t1\t10\t0\t20",
        "chr1\t3000\t4000\t1\t1\t1\t5\t0\t20",
    ],
    SAMPLE2_SJ: [
        "chr1\t1000\t2000\t1\t1\t1\t8\t0\t20",
        "chr1\t3000\t4000\t1\t1\t1\t3\t0\t20",
    ],
}


@pytest.fixture()
def star_junction_files(tmp_path: Path) -> list[str]:
    paths = []
    for name, rows in _SJ_ROWS.items():
        path = tmp_path / name
        path.write_text("\n".join(rows) + "\n")
        paths.append(str(path))
    return paths


@pytest.fixture()
def mock_fasta(tmp_path: Path) -> str:
    """A tiny single-chromosome FASTA covering the fixture junction coordinates."""
    import pyfaidx

    seq = "ACGT" * 2000  # 8000bp of chr1, comfortably covers positions up to ~4000.
    path = tmp_path / "mock_genome.fa"
    lines = [">chr1"]
    for i in range(0, len(seq), 70):
        lines.append(seq[i : i + 70])
    path.write_text("\n".join(lines) + "\n")
    pyfaidx.Faidx(str(path))  # builds mock_genome.fa.fai, as FastaExtractor requires
    return str(path)


def _splice_head_cfg(star_junction_files, kinds_to_ids, **extra):
    """Config entries for splice heads sharing one source (no junction filter)."""
    heads = []
    for kind, head_id in kinds_to_ids.items():
        entry = {
            "id": head_id, "source": "predefined", "kind": kind,
            "star_junctions": star_junction_files, "filter_to_junctions": False,
            "max_splice_sites": 16,
        }
        if kind == "splice_sites_junction":
            entry.update(extra)
        heads.append(entry)
    return {"heads": heads}


ALL_SPLICE_IDS = {
    "splice_sites_classification": "cls_head",
    "splice_sites_usage": "usage_head",
    "splice_sites_junction": "junction_head",
}


# Splice heads need real 1bp decoder embeddings, which use_encoder_output=True
# (skips the transformer/decoder) doesn't provide — so the end-to-end smoke
# tests need the full model, which in turn needs a real supported sequence
# length (16384, matching create_model_with_heads's own init_seq_len default).
FULL_MODEL_SEQUENCE_LENGTH = 16384


@pytest.fixture()
def mock_fasta_full(tmp_path: Path) -> str:
    """A single-chromosome FASTA long enough for a real (non-encoder-only) forward pass."""
    import pyfaidx

    seq = "ACGT" * ((FULL_MODEL_SEQUENCE_LENGTH // 4) + 10)
    path = tmp_path / "mock_genome_full.fa"
    lines = [">chr1"]
    for i in range(0, len(seq), 70):
        lines.append(seq[i : i + 70])
    path.write_text("\n".join(lines) + "\n")
    pyfaidx.Faidx(str(path))
    return str(path)


class TestStarJunctionsParser:
    def test_read_star_junctions(self, star_junction_files):
        df = sj.read_star_junctions(star_junction_files[0])
        assert list(df["chrom"]) == ["chr1", "chr1"]
        assert list(df["intron_start"]) == [1000, 3000]
        assert list(df["strand"]) == ["+", "+"]
        assert list(df["n_uniquely_mapped_reads"]) == [10, 5]

    def test_junctions_to_classification_array(self, star_junction_files):
        juncs = [sj.read_star_junctions(p) for p in star_junction_files]
        for j in juncs:
            j["exon_start"] = j["intron_start"] - 1
            j["exon_end"] = j["intron_end"] + 1
            j["count"] = j["n_uniquely_mapped_reads"]
        arr = sj.junctions_to_classification_array(juncs, "chr1", 0, 5000)
        assert arr.shape == (5000, 5)
        # donor at 1-based exon_start=999 -> 0-based relative idx 999-1-0=998, class 0 (Donor+)
        assert arr[998, 0] == 1.0
        # background positions default to class 4 (None)
        assert arr[0, 4] == 1.0

    def test_junctions_to_junction_matrix(self, star_junction_files):
        juncs = []
        for p in star_junction_files:
            j = sj.read_star_junctions(p)
            j["exon_start"] = j["intron_start"] - 1
            j["exon_end"] = j["intron_end"] + 1
            j["count"] = j["n_uniquely_mapped_reads"]
            juncs.append(j)
        cls_arr = sj.junctions_to_classification_array(juncs, "chr1", 0, 5000)
        positions, matrix = sj.junctions_to_junction_matrix(
            juncs, max_splice_sites=16, cls_arr=cls_arr, chrom="chr1", start=0, seq_len=5000
        )
        assert positions.shape == (4, 16)
        assert matrix.shape == (16, 16, 2 * len(juncs))
        assert matrix.sum() > 0


class TestFinetuneDataModuleSplice:
    def test_batches_have_expected_shapes(self, star_junction_files, mock_fasta):
        specs = ft_config.prepare_head_specs(
            _splice_head_cfg(star_junction_files, ALL_SPLICE_IDS), organism="HOMO_SAPIENS")
        windows = [genome.Interval(chromosome="chr1", start=0, end=5000)]
        module = FinetuneDataModule(
            intervals={"train": windows},
            fasta_path=mock_fasta,
            head_specs=specs,
            batch_size=1,
            shuffle=False,
        )
        batches = list(module.iter_batches("train"))
        assert len(batches) == 1
        batch = batches[0]

        assert batch["sequences"].shape == (1, 5000, 4)
        assert batch["targets_cls_head"].shape == (1, 5000, 5)
        assert batch["targets_usage_head"].shape == (1, 5000, 4)  # 2 samples * 2 strands
        assert batch["usage_alpha_usage_head"].shape == (1, 5000, 4)
        assert batch["targets_junction_head"].shape == (1, 16, 16, 4)
        assert batch["splice_site_positions"].shape == (1, 4, 16)
        assert batch["splice_site_positions"].dtype == np.int32
        assert "junction_d_rel" not in batch          # annotated mode: no raw events

    def test_heads_sharing_a_source_share_one_loader(self, star_junction_files):
        specs = ft_config.prepare_head_specs(
            _splice_head_cfg(star_junction_files, ALL_SPLICE_IDS), organism="HOMO_SAPIENS")
        targets = build_splice_targets(specs)
        assert len(targets) == 1
        assert targets[0].head_ids == dict(ALL_SPLICE_IDS)

    def test_predicted_junction_head_emits_raw_events(self, star_junction_files, mock_fasta):
        cfg = _splice_head_cfg(
            star_junction_files,
            {"splice_sites_classification": "cls_head", "splice_sites_junction": "junction_head"},
            junction_position_source="predicted", junction_top_k=16,
            classification_head_id="cls_head",
        )
        specs = ft_config.prepare_head_specs(cfg, organism="HOMO_SAPIENS")
        module = FinetuneDataModule(
            intervals={"train": [genome.Interval(chromosome="chr1", start=0, end=5000)]},
            fasta_path=mock_fasta, head_specs=specs, batch_size=1, shuffle=False,
            junction_event_pad_size=32,
        )
        batch = next(module.iter_batches("train"))
        assert batch["junction_d_rel"].shape == (1, 32)
        assert batch["junction_counts"].shape == (1, 32, 2)

    def test_filter_to_junctions_drops_windows_without_junctions(self, star_junction_files, mock_fasta):
        cfg = _splice_head_cfg(star_junction_files, {"splice_sites_classification": "cls_head"})
        cfg["heads"][0]["filter_to_junctions"] = True
        specs = ft_config.prepare_head_specs(cfg, organism="HOMO_SAPIENS")
        module = FinetuneDataModule(
            intervals={"train": [genome.Interval(chromosome="chr1", start=0, end=5000),
                                 genome.Interval(chromosome="chr1", start=6000, end=8000)]},
            fasta_path=mock_fasta, head_specs=specs, batch_size=1, shuffle=False,
        )
        assert [(w.start, w.end) for w in module._intervals["train"]] == [(0, 5000)]

    def test_rejects_splice_source_on_non_splice_kind(self, star_junction_files):
        import dataclasses

        spec = ft_config.prepare_head_specs(
            _splice_head_cfg(star_junction_files, {"splice_sites_classification": "cls_head"}),
            organism="HOMO_SAPIENS")[0]
        with pytest.raises(ValueError, match="has a splice_source but kind"):
            build_splice_targets([dataclasses.replace(spec, kind="rna_seq")])

    def test_rejects_two_junction_heads(self, star_junction_files):
        cfg = _splice_head_cfg(star_junction_files, {"splice_sites_junction": "junc_a"})
        cfg["heads"].append({**cfg["heads"][0], "id": "junc_b", "min_unique_reads": 2})
        specs = ft_config.prepare_head_specs(cfg, organism="HOMO_SAPIENS")
        with pytest.raises(ValueError, match="At most one splice_sites_junction head"):
            build_splice_targets(specs)


class TestSpliceHeadSpecs:
    """Config wiring: prepare_head_specs / validate_head_specs / head construction."""

    def _cfg(self, star_junction_files, kind):
        return {
            "heads": [
                {
                    "id": f"my_{kind}",
                    "source": "predefined",
                    "kind": kind,
                    "star_junctions": star_junction_files,
                }
            ]
        }

    @pytest.mark.parametrize(
        "kind,expected_num_tracks",
        [
            ("splice_sites_classification", 5),
            ("splice_sites_usage", 4),
            ("splice_sites_junction", 4),
        ],
    )
    def test_prepare_and_construct_head(
        self, star_junction_files, kind, expected_num_tracks
    ):
        specs = ft_config.prepare_head_specs(
            self._cfg(star_junction_files, kind), organism="HOMO_SAPIENS"
        )
        ft_config.validate_head_specs(specs)
        assert len(specs) == 1
        spec = specs[0]
        assert spec.kind == kind
        assert spec.splice_source is not None

        def fwd():
            head = custom_heads_module.create_predefined_head_from_config(
                spec.config, metadata=spec.metadata
            )
            return head.num_tracks

        transformed = hk.transform(fwd)
        params = transformed.init(jax.random.PRNGKey(0))
        num_tracks = transformed.apply(params, None)
        assert int(num_tracks) == expected_num_tracks

    def test_missing_star_junctions_raises(self):
        cfg = {
            "heads": [
                {"id": "bad", "source": "predefined", "kind": "splice_sites_classification"}
            ]
        }
        with pytest.raises(ValueError, match="star_junctions"):
            ft_config.prepare_head_specs(cfg, organism="HOMO_SAPIENS")


class TestPredictedJunctionPositionConfig:
    """Config validation for junction_position_source="predicted"."""

    def _junction_entry(self, star_junction_files, **overrides):
        entry = {
            "id": "junc_head",
            "source": "predefined",
            "kind": "splice_sites_junction",
            "star_junctions": star_junction_files,
            "junction_position_source": "predicted",
        }
        entry.update(overrides)
        return entry

    def test_missing_classification_head_id_raises(self, star_junction_files):
        cfg = {"heads": [self._junction_entry(star_junction_files)]}
        with pytest.raises(ValueError, match="classification_head_id"):
            ft_config.prepare_head_specs(cfg, organism="HOMO_SAPIENS")

    def test_unresolved_classification_head_raises(self, star_junction_files):
        cfg = {
            "heads": [
                self._junction_entry(star_junction_files, classification_head_id="missing_cls")
            ]
        }
        with pytest.raises(ValueError, match="no \"splice_sites_classification\" head"):
            ft_config.prepare_head_specs(cfg, organism="HOMO_SAPIENS")

    def test_valid_predicted_config_accepted(self, star_junction_files):
        cfg = {
            "heads": [
                {
                    "id": "cls_head",
                    "source": "predefined",
                    "kind": "splice_sites_classification",
                    "star_junctions": star_junction_files,
                    "filter_to_junctions": False,
                },
                self._junction_entry(star_junction_files, classification_head_id="cls_head"),
            ]
        }
        specs = ft_config.prepare_head_specs(cfg, organism="HOMO_SAPIENS")
        ft_config.validate_head_specs(specs)
        junction_spec = next(s for s in specs if s.head_id == "junc_head")
        assert junction_spec.splice_source.junction_position_source == "predicted"
        assert junction_spec.splice_source.classification_head_id == "cls_head"


class TestTopKSplicePositions:
    def test_selects_top_k_per_role_and_sorts_ascending(self):
        logits = jnp.zeros((1, 10, 5))
        # donor+ channel: peaks at positions 1 (score 5) and 3 (score 3)
        logits = logits.at[0, :, 0].set(jnp.array([0, 5, 0, 3, 0, 0, 0, 0, 0, 0]))
        positions = top_k_splice_positions(logits, top_k=2)
        assert positions.shape == (1, 4, 2)
        assert positions.dtype == jnp.int32
        np.testing.assert_array_equal(np.asarray(positions[0, 0]), [1, 3])

    def test_pads_with_minus_one_when_top_k_exceeds_seq_len(self):
        logits = jnp.zeros((1, 4, 5))
        positions = top_k_splice_positions(logits, top_k=6)
        assert positions.shape == (1, 4, 6)
        np.testing.assert_array_equal(np.asarray(positions[0, 0]), [0, 1, 2, 3, -1, -1])


    def test_selects_on_probabilities_and_applies_threshold(self):
        logits = jnp.full((1, 6, 5), -10.0).at[0, :, 4].set(0.0)
        logits = logits.at[0, 1, 0].set(3.0)           # p(donor+) ~ 0.95
        logits = logits.at[0, 4, 0].set(5.0).at[0, 4, 4].set(8.0)  # high logit, p ~ 0.05
        logits = logits.at[0, 2, 0].set(0.0)           # p = 0.5
        positions = top_k_splice_positions(logits, top_k=4, threshold=0.1)
        np.testing.assert_array_equal(np.asarray(positions[0, 0]), [1, 2, -1, -1])
        assert bool(jnp.all(positions[0, 1:] == -1))

    def test_matches_alphagenome_research(self):
        from alphagenome_research.model import splicing

        logits = 3 * jax.random.normal(jax.random.PRNGKey(0), (2, 300, 5))
        ours = top_k_splice_positions(logits, top_k=64, threshold=0.1)
        ref = splicing.generate_splice_site_positions(
            jax.nn.softmax(logits, axis=-1), alt=None, splice_sites=None,
            k=64, pad_to_length=64, threshold=0.1,
        )
        np.testing.assert_array_equal(np.asarray(ours), np.asarray(ref))


class TestRawJunctionEventsNoCap:
    def test_rows_grow_instead_of_truncating(self):
        t = SpliceTargets.__new__(SpliceTargets)
        t._junction_event_pad_size = 4
        assert t.junction_event_rows(0) == 4
        assert t.junction_event_rows(4) == 4
        assert t.junction_event_rows(5) == 8
        assert t.junction_event_rows(33) == 64


def test_cli_defaults_aligned():
    from alphagenome_ft.finetune.args import parse_args

    args = parse_args([
        "--checkpoint-path", "ckpt", "--genome", "g.fa",
        "--train-bed", "t.bed", "--val-bed", "v.bed",
        "--star-junctions", "a.tab", "--bigwig", "a.bw",
        "--output-dir", "out", "--run-name", "r",
    ])
    assert args.splice_num_segments == 1
    # shared defaults (same values in alphagenome-pytorch)
    assert args.weight_decay == 0.0
    assert args.warmup_steps == 0
    assert args.lr_schedule == "constant"
    assert args.mode == "linear-probe"
    assert args.sequence_length == 1048576
    assert args.pretrained_head_samples is None   # runner applies splice_site:0 if that head exists
    assert args.targets_config is None
    assert args.rope_init == "truncated_normal"
    assert args.batch_size == 1
    assert args.lora_alpha == 16 and isinstance(args.lora_alpha, int)
    assert args.seed == 1234
    assert args.num_devices is None      # resolved to all visible devices at startup
    assert args.resume == "auto"
    assert args.max_splice_sites == 256
    assert args.junction_top_k == 512
    assert args.splice_site_threshold == 0.1


class TestBuildJunctionMatrix:
    """Cross-check the vectorized JAX rebuild against the trusted pandas reference."""

    def test_matches_pandas_reference(self):
        rows1 = pd.DataFrame({
            "chrom": ["chr1", "chr1", "chr1"],
            "exon_start": [100, 300, 300],
            "exon_end": [200, 400, 4500],
            "strand": ["+", "+", "-"],
            "count": [10.0, 5.0, 7.0],
        })
        rows2 = pd.DataFrame({
            "chrom": ["chr1", "chr1"],
            "exon_start": [100, 300],
            "exon_end": [200, 400],
            "strand": ["+", "-"],
            "count": [3.0, 2.0],
        })
        all_juncs = [rows1, rows2]

        cls_arr = np.zeros((5000, 5), dtype=np.float32)
        cls_arr[:, 4] = 1.0

        def set_class(idx, c):
            cls_arr[idx, 4] = 0.0
            cls_arr[idx, c] = 1.0

        set_class(99, 0)
        set_class(299, 0)
        set_class(199, 1)
        set_class(399, 1)
        # Minus strand: donor is the right site, acceptor the left (AlphaGenome convention).
        set_class(4499, 2)
        set_class(299, 3)

        max_splice_sites = 8
        positions_np, matrix_np = sj.junctions_to_junction_matrix(
            all_juncs, max_splice_sites=max_splice_sites, cls_arr=cls_arr,
            chrom="chr1", start=0, seq_len=5000,
        )

        n_samples = len(all_juncs)
        rows = []
        for sample_idx, df in enumerate(all_juncs):
            for _, r in df.iterrows():
                counts = [0.0] * n_samples
                counts[sample_idx] = float(r["count"])
                left, right = int(r["exon_start"]) - 1, int(r["exon_end"]) - 1
                donor, acceptor = (left, right) if r["strand"] == "+" else (right, left)
                rows.append((donor, acceptor, r["strand"] == "+", counts))
        max_junctions = 8
        d_rel = np.full((1, max_junctions), -1, dtype=np.int32)
        a_rel = np.full((1, max_junctions), -1, dtype=np.int32)
        is_pos = np.zeros((1, max_junctions), dtype=bool)
        counts_arr = np.zeros((1, max_junctions, n_samples), dtype=np.float32)
        for i, (d, a, pos_strand, c) in enumerate(rows):
            d_rel[0, i] = d
            a_rel[0, i] = a
            is_pos[0, i] = pos_strand
            counts_arr[0, i, :] = c

        matrix_jax = build_junction_matrix(
            jnp.asarray(positions_np[None]),
            jnp.asarray(d_rel), jnp.asarray(a_rel), jnp.asarray(is_pos), jnp.asarray(counts_arr),
            max_splice_sites=max_splice_sites,
        )
        np.testing.assert_allclose(np.asarray(matrix_jax)[0], matrix_np)


@pytest.mark.finetuning
class TestSpliceTrainingSmoke:
    """End-to-end smoke test: real pretrained checkpoint + one train() step."""

    def test_one_train_step_all_splice_heads(
        self, tmp_path, star_junction_files, mock_fasta_full, device
    ):
        require_kaggle_credentials()
        from alphagenome_ft import create_model_with_heads, register_predefined_head
        from alphagenome_ft.finetune.train import register_predefined_heads, train as run_train
        from alphagenome.data import genome as genome_lib

        cfg = {
            "heads": [
                {
                    "id": "cls_head",
                    "source": "predefined",
                    "kind": "splice_sites_classification",
                    "star_junctions": star_junction_files,
                    "filter_to_junctions": False,
                },
                {
                    "id": "usage_head",
                    "source": "predefined",
                    "kind": "splice_sites_usage",
                    "star_junctions": star_junction_files,
                    "filter_to_junctions": False,
                },
                {
                    "id": "junction_head",
                    "source": "predefined",
                    "kind": "splice_sites_junction",
                    "star_junctions": star_junction_files,
                    "filter_to_junctions": False,
                    "max_splice_sites": 16,
                },
            ]
        }
        specs = ft_config.prepare_head_specs(cfg, organism="HOMO_SAPIENS")
        ft_config.validate_head_specs(specs)
        register_predefined_heads(specs)

        model = create_model_with_heads(
            "all_folds",
            heads=[s.head_id for s in specs],
            device=device,
            init_seq_len=FULL_MODEL_SEQUENCE_LENGTH,
        )

        windows = [
            genome_lib.Interval(chromosome="chr1", start=0, end=FULL_MODEL_SEQUENCE_LENGTH)
        ]
        data_module = FinetuneDataModule(
            intervals={"train": windows, "valid": windows},
            fasta_path=mock_fasta_full,
            head_specs=specs,
            batch_size=1,
            shuffle=False,
        )

        run_train(
            model,
            data_module,
            specs,
            learning_rate=1e-4,
            weight_decay=0.0,
            num_epochs=1,
            max_train_steps=1,
            heads_only=True,
            organism="HOMO_SAPIENS",
        )

    @pytest.mark.parametrize("heads_only", [True, False])
    def test_one_train_step_predicted_junction_positions(
        self, star_junction_files, mock_fasta_full, device, heads_only,
    ):
        """Predicted mode must be correct with and without a frozen backbone —
        that's the whole point of rebuilding the target matrix as a pure JAX
        op instead of a host/pandas round-trip (see splice_positions.py)."""
        require_kaggle_credentials()
        from alphagenome_ft import create_model_with_heads
        from alphagenome_ft.finetune.train import register_predefined_heads, train as run_train
        from alphagenome.data import genome as genome_lib

        cfg = {
            "heads": [
                {
                    "id": "cls_head",
                    "source": "predefined",
                    "kind": "splice_sites_classification",
                    "star_junctions": star_junction_files,
                    "filter_to_junctions": False,
                },
                {
                    "id": "junction_head",
                    "source": "predefined",
                    "kind": "splice_sites_junction",
                    "star_junctions": star_junction_files,
                    "filter_to_junctions": False,
                    "max_splice_sites": 16,
                    "junction_position_source": "predicted",
                    "junction_top_k": 16,
                    "classification_head_id": "cls_head",
                },
            ]
        }
        specs = ft_config.prepare_head_specs(cfg, organism="HOMO_SAPIENS")
        ft_config.validate_head_specs(specs)
        register_predefined_heads(specs)

        model = create_model_with_heads(
            "all_folds",
            heads=[s.head_id for s in specs],
            device=device,
            init_seq_len=FULL_MODEL_SEQUENCE_LENGTH,
        )

        windows = [
            genome_lib.Interval(chromosome="chr1", start=0, end=FULL_MODEL_SEQUENCE_LENGTH)
        ]
        data_module = FinetuneDataModule(
            intervals={"train": windows, "valid": windows},
            fasta_path=mock_fasta_full,
            head_specs=specs,
            batch_size=1,
            shuffle=False,
            junction_event_pad_size=32,
        )

        run_train(
            model,
            data_module,
            specs,
            learning_rate=1e-4,
            weight_decay=0.0,
            num_epochs=1,
            max_train_steps=1,
            heads_only=heads_only,
            organism="HOMO_SAPIENS",
        )
