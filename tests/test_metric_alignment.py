"""Validation metrics match alphagenome-pytorch's definitions.

Each test computes a metric with alphagenome_ft's implementation and with
alphagenome-pytorch's own functions on the same synthetic data.
"""

from __future__ import annotations

import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

torch = pytest.importorskip("torch")

_PT_SRC = Path(__file__).resolve().parents[2] / "alphagenome-pytorch-pr57" / "src"
if not _PT_SRC.exists():
    pytest.skip("alphagenome-pytorch-pr57 checkout not found", allow_module_level=True)
sys.path.insert(0, str(_PT_SRC))

from alphagenome_pytorch.metrics import pearson_r, profile_pearson_r  # noqa: E402
from alphagenome_pytorch.extensions.finetuning import training as pt  # noqa: E402

import importlib  # noqa: E402

ft = importlib.import_module("alphagenome_ft.finetune.train")

RTOL = 1e-4


def _pt_r(pred, true):
    return pearson_r(pred.unsqueeze(0), true.unsqueeze(0), dim=1).item()


def test_genome_track_profile_and_count_pearson():
    rng = np.random.default_rng(0)
    batches = [(rng.gamma(2, size=(3, 64, 4)), rng.gamma(2, size=(3, 64, 4))) for _ in range(2)]
    acc = ft._ValidationMetrics(per_sample=False)
    for p, t in batches:
        pj, tj = jnp.asarray(p, jnp.float32), jnp.asarray(t, jnp.float32)
        acc.add_genome_tracks("rna_seq", np.asarray(ft._profile_pearson_r(pj, tj)),
                              np.asarray(pj.sum(1)), np.asarray(tj.sum(1)))
    metrics, _, hist = acc.finalize()

    all_r = torch.cat([profile_pearson_r(torch.tensor(p, dtype=torch.float32),
                                         torch.tensor(t, dtype=torch.float32)) for p, t in batches])
    pc = torch.cat([torch.tensor(p, dtype=torch.float32).sum(1) for p, _ in batches])
    tc = torch.cat([torch.tensor(t, dtype=torch.float32).sum(1) for _, t in batches])
    np.testing.assert_allclose(metrics["rna_seq_1bp_profile_pearson_r_mean"], all_r.mean().item(), rtol=RTOL)
    np.testing.assert_allclose(metrics["rna_seq_1bp_profile_pearson_r_std"], all_r.std().item(), rtol=RTOL)
    np.testing.assert_allclose(metrics["rna_seq_1bp_count_pearson_r"],
                               pearson_r(pc, tc, dim=0).mean().item(), rtol=RTOL)
    assert len(hist["rna_seq_1bp_profile_pearson_r_values"]) == all_r.numel()


def _usage_batch(rng):
    logits = rng.normal(size=(2, 50, 6)).astype(np.float32)
    target = np.where(rng.random((2, 50, 6)) < 0.15, rng.random((2, 50, 6)), 0.0).astype(np.float32)
    return logits, target


def test_splice_usage_pearson_and_per_sample():
    rng = np.random.default_rng(1)
    batches = [_usage_batch(rng) for _ in range(3)]
    acc = ft._ValidationMetrics(per_sample=True)
    head = object.__new__(pt.SpliceSitesUsageHead)
    pt_full_p, pt_full_t = [], []
    pt_ps = [([], []) for _ in range(6)]
    for logits, target in batches:
        probs = 1 / (1 + np.exp(-logits))
        stats = ft._usage_device_stats(jnp.asarray(probs), jnp.asarray(target))
        acc.add_usage("splice_usage", {k: {kk: np.asarray(vv) for kk, vv in v.items()} for k, v in stats.items()})
        preds = {1: torch.tensor(logits)}
        tdict = {"usage": torch.tensor(target)}
        full = pt._extract_splice_pearson_pairs(head, preds, tdict, "cpu")["full"]
        pt_full_p.append(full[0]); pt_full_t.append(full[1])
        for s, item in enumerate(pt._extract_usage_pearson_per_sample(preds, tdict, "cpu")):
            if item is not None:
                pt_ps[s][0].append(item[0]); pt_ps[s][1].append(item[1])
    metrics, rows, _ = acc.finalize()
    np.testing.assert_allclose(metrics["splice_usage_pearson_r"],
                               _pt_r(torch.cat(pt_full_p), torch.cat(pt_full_t)), rtol=RTOL)
    for s in range(6):
        np.testing.assert_allclose(rows[s]["splice_usage_pearson_r"],
                                   _pt_r(torch.cat(pt_ps[s][0]), torch.cat(pt_ps[s][1])), rtol=RTOL)


def _junction_batch(rng, k=6, n_t=2):
    positions = np.full((2, 4, k), -1, dtype=np.int32)
    for b in range(2):
        for role in range(4):
            n_valid = rng.integers(2, k + 1)
            positions[b, role, :n_valid] = np.sort(rng.choice(1000, n_valid, replace=False))
    pred = rng.gamma(1.0, size=(2, k, k, 2 * n_t)).astype(np.float32)
    target = np.where(rng.random((2, k, k, 2 * n_t)) < 0.3,
                      rng.integers(1, 50, size=(2, k, k, 2 * n_t)), 0).astype(np.float32)
    # Heads zero predictions/targets outside valid pairs.
    for sl, d, a in ((slice(None, n_t), 0, 1), (slice(n_t, None), 2, 3)):
        pairs = (positions[:, d] >= 0)[:, :, None] & (positions[:, a] >= 0)[:, None, :]
        pred[..., sl] *= pairs[..., None]
        target[..., sl] *= pairs[..., None]
    return pred, target, positions


def test_splice_junction_variants_auprc_and_per_sample():
    from sklearn.metrics import average_precision_score

    rng = np.random.default_rng(2)
    n_t = 2
    batches = [_junction_batch(rng, n_t=n_t) for _ in range(3)]
    acc = ft._ValidationMetrics(per_sample=True)
    head = object.__new__(pt.SpliceSitesJunctionHead)
    head._num_tissues = n_t
    pt_vars: dict = {}
    pt_ps = [{"full": ([], []), "nonzero": ([], [])} for _ in range(n_t)]
    for pred, target, positions in batches:
        acc.add_junction("splice_junctions", pred, target, positions)
        preds = {"pos_counts": torch.tensor(pred[..., :n_t]), "neg_counts": torch.tensor(pred[..., n_t:]),
                 "positions": torch.tensor(positions)}
        tdict = {"junction_matrix": torch.tensor(target), "junction_positions": torch.tensor(positions)}
        for name, data in pt._extract_splice_pearson_pairs(head, preds, tdict, "cpu").items():
            if data:
                pt_vars.setdefault(name, ([], []))
                pt_vars[name][0].append(data["pred"]); pt_vars[name][1].append(data["true"])
        for s, data in enumerate(pt._extract_junction_pearson_per_sample(preds, tdict, "cpu")):
            for v in ("full", "nonzero"):
                if data[v] is not None:
                    pt_ps[s][v][0].append(data[v][0]); pt_ps[s][v][1].append(data[v][1])
    metrics, rows, _ = acc.finalize()

    for variant, key in (("full", ""), ("nonzero", "_nonzero"), ("psi5", "_psi5"), ("psi3", "_psi3")):
        p, t = torch.cat(pt_vars[variant][0]), torch.cat(pt_vars[variant][1])
        if variant in ("full", "nonzero"):
            p, t = torch.log1p(p), torch.log1p(t)
        np.testing.assert_allclose(metrics[f"splice_junctions_pearson_r{key}"], _pt_r(p, t), rtol=RTOL)
    full_t = torch.cat(pt_vars["full"][1])
    np.testing.assert_allclose(metrics["splice_junctions_target_nonzero_frac"],
                               (full_t > 0).float().mean().item(), rtol=RTOL)
    np.testing.assert_allclose(
        metrics["splice_junctions_auprc_junction"],
        average_precision_score(torch.cat(pt_vars["binary_cls"][1]).numpy(),
                                torch.cat(pt_vars["binary_cls"][0]).numpy()),
        rtol=RTOL,
    )
    for s in range(n_t):
        for v, key in (("full", "splice_junctions_pearson_r"), ("nonzero", "splice_junctions_pearson_r_nonzero")):
            p = torch.log1p(torch.cat(pt_ps[s][v][0])); t = torch.log1p(torch.cat(pt_ps[s][v][1]))
            np.testing.assert_allclose(rows[s][key], _pt_r(p, t), rtol=RTOL)


def test_splice_site_classification_auprc():
    from sklearn.metrics import average_precision_score

    rng = np.random.default_rng(3)
    s_len = 40
    logits = rng.normal(size=(2, s_len, 5)).astype(np.float32)
    target = np.zeros((2, s_len, 5), np.float32)
    # Many active sites, few background positions -> every background position
    # is sampled (min(n_active, n_bg) == n_bg), so the sample is deterministic.
    for b in range(2):
        for i in range(s_len):
            if rng.random() < 0.8:
                target[b, i, rng.integers(0, 4)] = 1.0
    acc = ft._ValidationMetrics(per_sample=False)
    acc.add_classification("splice_site", logits, target)
    metrics, _, _ = acc.finalize()

    # alphagenome-pytorch validate_multihead's classification auPRC block.
    lf, tf = logits.reshape(-1, 5), target.reshape(-1, 5)
    active = tf.any(-1)
    lg = np.concatenate([lf[active], lf[~active]]); tg = np.concatenate([tf[active], tf[~active]])
    probs = torch.softmax(torch.tensor(lg), dim=-1).numpy()
    names = ["donor_pos", "acceptor_pos", "donor_neg", "acceptor_neg", "no_site"]
    aps = {}
    for i, n in enumerate(names):
        if tg[:, i].sum() > 0:
            aps[n] = average_precision_score(tg[:, i], probs[:, i])
            np.testing.assert_allclose(metrics[f"splice_site_auprc_{n}"], aps[n], rtol=RTOL)
    cols = [n for n in names[:4] if n in aps]
    np.testing.assert_allclose(metrics["splice_site_auprc_macro"], np.mean([aps[n] for n in cols]), rtol=RTOL)
    w = np.array([tg[:, names.index(n)].sum() for n in cols])
    np.testing.assert_allclose(metrics["splice_site_auprc_weighted"],
                               float(np.sum(np.array([aps[n] for n in cols]) * w / w.sum())), rtol=RTOL)


def test_epoch_log_rows_match_pytorch_layout(tmp_path):
    import csv

    from alphagenome_ft.logging_utils import TrainingLogger

    logger = TrainingLogger(tmp_path)
    logger.log_epoch(1, {"train_loss": 1.0, "splice_junctions_pearson_r": 0.5},
                     per_sample=[{"splice_junctions_pearson_r": 0.1}, {"splice_junctions_pearson_r": 0.2}])
    rows = list(csv.DictReader(open(tmp_path / "epoch_log.csv")))
    assert [r["sample"] for r in rows] == ["nan", "0", "1"]
    assert rows[0]["splice_junctions_pearson_r"] == "0.5"
    assert rows[2]["splice_junctions_pearson_r"] == "0.2"
