"""FinetuneDataModule (BigWig + splice heads in one pass) and the config-driven CLI."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from alphagenome.data import genome

from alphagenome_ft.finetune import config as ft_config
from alphagenome_ft.finetune.data import FinetuneDataModule, build_fasta_index
from alphagenome_ft.finetune.head_init import modality_head_ids

CHROM_LEN = 8000


@pytest.fixture()
def fasta(tmp_path: Path) -> Path:
    pyfaidx = pytest.importorskip("pyfaidx")
    rng = np.random.default_rng(0)
    seq = "".join(rng.choice(list("ACGT"), CHROM_LEN))
    path = tmp_path / "g.fa"
    path.write_text(">chr1\n" + "\n".join(seq[i:i + 70] for i in range(0, len(seq), 70)) + "\n")
    pyfaidx.Faidx(str(path))
    return path


@pytest.fixture()
def bigwig(tmp_path: Path) -> Path:
    pyBigWig = pytest.importorskip("pyBigWig")
    path = tmp_path / "rna.bw"
    bw = pyBigWig.open(str(path), "w")
    bw.addHeader([("chr1", CHROM_LEN)])
    starts = np.arange(0, CHROM_LEN, 10, dtype=np.int64)
    bw.addEntries(["chr1"] * len(starts), starts.tolist(), ends=(starts + 10).tolist(),
                  values=(starts % 7).astype(float).tolist())
    bw.close()
    return path


@pytest.fixture()
def sj_file(tmp_path: Path) -> Path:
    path = tmp_path / "s1.SJ.out.tab"
    path.write_text("chr1\t1000\t2000\t1\t1\t1\t10\t0\t20\nchr1\t3000\t4000\t2\t1\t1\t5\t0\t20\n")
    return path


def _config(bigwig, sj_file):
    return {"heads": [
        {"id": "rna", "source": "predefined", "kind": "rna_seq", "resolutions": [1],
         "targets": [{"path": str(bigwig), "nonzero_mean": 1.0}]},
        {"id": "cls", "source": "predefined", "kind": "splice_sites_classification",
         "star_junctions": [str(sj_file)], "filter_to_junctions": False},
        {"id": "junc", "source": "predefined", "kind": "splice_sites_junction",
         "star_junctions": [str(sj_file)], "filter_to_junctions": False, "max_splice_sites": 8},
    ]}


def test_bigwig_and_splice_heads_in_one_batch(fasta, bigwig, sj_file, monkeypatch):
    specs = ft_config.prepare_head_specs(_config(bigwig, sj_file), organism="HOMO_SAPIENS")
    module = FinetuneDataModule(
        intervals={"train": [genome.Interval("chr1", 0, 5000)]},
        fasta_path=fasta, head_specs=specs, batch_size=1, shuffle=False,
    )
    from alphagenome.io import fasta as fasta_lib
    calls = []
    real = fasta_lib.FastaExtractor.extract
    monkeypatch.setattr(fasta_lib.FastaExtractor, "extract",
                        lambda self, iv: calls.append(iv) or real(self, iv))
    batch = next(module.iter_batches("train"))
    assert len(calls) == 1                                   # one sequence read per window
    assert batch["sequences"].shape == (1, 5000, 4)
    assert batch["targets_rna"].shape == (1, 5000, 1)
    np.testing.assert_allclose(batch["targets_rna"][0, :20, 0], [0.0] * 10 + [3.0] * 10)
    assert batch["targets_cls"].shape == (1, 5000, 5)
    assert batch["targets_junc"].shape == (1, 8, 8, 2)
    assert batch["splice_site_positions"].shape == (1, 4, 8)


def test_build_fasta_index_returns_sizes(fasta):
    assert build_fasta_index(fasta) == {"chr1": CHROM_LEN}


def test_modality_head_ids(bigwig, sj_file):
    specs = ft_config.prepare_head_specs(_config(bigwig, sj_file), organism="HOMO_SAPIENS")
    assert modality_head_ids(specs) == {"rna_seq": "rna", "splice_site": "cls", "splice_junctions": "junc"}


def test_shorthand_config_matches_previous_preset(tmp_path):
    from alphagenome_ft.finetune.__main__ import shorthand_targets_config

    args = SimpleNamespace(
        star_junctions=["a.tab"], ssu=None, gtf=None, bigwig=["f.bw", "r.bw"],
        max_splice_sites=256, junction_position_source="annotated",
        junction_top_k=512, splice_site_threshold=0.1,
    )
    heads = shorthand_targets_config(args)["heads"]
    assert [(h["id"], h["kind"]) for h in heads] == [
        ("splice_site", "splice_sites_classification"),
        ("splice_usage", "splice_sites_usage"),
        ("splice_junctions", "splice_sites_junction"),
        ("rna_seq", "rna_seq"),
    ]
    assert all(h["filter_to_junctions"] is False for h in heads[:3])
    assert heads[3]["resolutions"] == [1]
    assert [t["path"] for t in heads[3]["targets"]] == ["f.bw", "r.bw"]


def test_yaml_config_resolves_splice_paths_relative_to_file(tmp_path, sj_file):
    yaml = pytest.importorskip("yaml")
    cfg_dir = tmp_path / "cfg"
    cfg_dir.mkdir()
    (cfg_dir / "s.tab").write_text(sj_file.read_text())
    path = cfg_dir / "t.yaml"
    path.write_text(yaml.safe_dump({"heads": [{
        "id": "cls", "source": "predefined", "kind": "splice_sites_classification",
        "star_junctions": ["s.tab"]}]}))
    cfg = ft_config.load_targets_config(path)
    assert cfg["heads"][0]["star_junctions"] == [str((cfg_dir / "s.tab").resolve())]


@pytest.mark.parametrize("extra,ok", [
    (["--targets-config", "t.yaml"], True),
    (["--bigwig", "a.bw", "--star-junctions", "a.tab"], True),
    (["--bigwig", "a.bw"], False),
    ([], False),
    (["--targets-config", "t.yaml", "--bigwig", "a.bw"], False),
])
def test_cli_needs_config_or_shorthand(extra, ok):
    from alphagenome_ft.finetune.args import parse_args

    base = ["--checkpoint-path", "c", "--genome", "g.fa", "--train-bed", "t.bed",
            "--val-bed", "v.bed", "--output-dir", "o", "--run-name", "r"]
    if ok:
        parse_args(base + extra)
    else:
        with pytest.raises(SystemExit):
            parse_args(base + extra)
