"""fp32 master head params with bf16 compute, and chromosome-bounded windows."""

from __future__ import annotations

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

from alphagenome_ft.custom_model import (
    _cast_head_params_to_float32,
    _head_compute_dtype_ctx,
)
from alphagenome_ft.finetune.data import load_split_intervals


def _forward(x):
    with hk.name_scope("head"), _head_compute_dtype_ctx("bfloat16"):
        out = hk.Linear(3, name="lin")(x)
    return out


def test_head_params_stay_fp32_and_compute_in_bf16():
    f = hk.without_apply_rng(hk.transform(_forward))
    x = jnp.ones((2, 4), jnp.bfloat16)
    # create_model_with_heads upcasts head params right after init (init
    # creates them in the bf16 input dtype).
    params = _cast_head_params_to_float32(f.init(jax.random.PRNGKey(0), x))
    assert all(v.dtype == jnp.float32 for v in jax.tree_util.tree_leaves(params))
    assert f.apply(params, x).dtype == jnp.bfloat16
    grads = jax.grad(lambda p: f.apply(p, x).astype(jnp.float32).sum())(params)
    assert all(g.dtype == jnp.float32 for g in jax.tree_util.tree_leaves(grads))
    # A tiny update survives in fp32 storage (it would round away in bf16).
    w = params["head/lin"]["w"]
    assert bool(jnp.any((w + 1e-4) != w))


def test_checkpoint_bf16_head_params_are_upcast():
    params = {"head/x/lin": {"w": jnp.ones((2, 2), jnp.bfloat16)},
              "alphagenome/trunk": {"w": jnp.ones((2, 2), jnp.bfloat16)}}
    out = _cast_head_params_to_float32(params)
    assert out["head/x/lin"]["w"].dtype == jnp.float32
    assert out["alphagenome/trunk"]["w"].dtype == jnp.bfloat16


def test_intervals_off_chromosome_are_skipped(tmp_path):
    bed = tmp_path / "a.bed"
    bed.write_text("chr1\t10\t20\nchr1\t500\t510\nchr1\t990\t1000\nchrX\t500\t510\n")
    sizes = {"chr1": 1000}
    kept = load_split_intervals({"train": bed}, 100, chrom_sizes=sizes)["train"]
    assert [(i.chromosome, i.start, i.end) for i in kept] == [("chr1", 455, 555)]
    # Without sizes: build_interval behaviour (start clamped at 0, nothing skipped).
    legacy = load_split_intervals({"train": bed}, 100)["train"]
    assert (legacy[0].start, legacy[0].end) == (0, 100)
    np.testing.assert_equal(len(legacy), 4)


def test_validation_order_is_wrap_padded_not_dropped():
    from alphagenome_ft.finetune.data import _pad_order_to_batch_multiple

    np.testing.assert_array_equal(_pad_order_to_batch_multiple(np.arange(6), 4), [0, 1, 2, 3, 4, 5, 0, 1])
    np.testing.assert_array_equal(_pad_order_to_batch_multiple(np.arange(8), 4), np.arange(8))
    np.testing.assert_array_equal(_pad_order_to_batch_multiple(np.arange(2), 5), [0, 1, 0, 1, 0])
    # torch DistributedSampler pads the same way (to a multiple of world size).
    import pytest
    torch = pytest.importorskip("torch")
    from torch.utils.data.distributed import DistributedSampler
    world = 4
    padded = []
    for r in range(world):
        padded.append(list(DistributedSampler(range(6), num_replicas=world, rank=r, shuffle=False)))
    flat = sorted(sum(padded, []))
    assert flat == sorted(_pad_order_to_batch_multiple(np.arange(6), world).tolist())


def test_epoch_log_has_pytorch_leading_columns(tmp_path):
    import csv
    import inspect

    from alphagenome_ft import logging_utils

    sig = inspect.signature(logging_utils.TrainingLogger.__init__)
    kwargs = {}
    for name, p in sig.parameters.items():
        if name == "self" or p.default is not inspect.Parameter.empty:
            continue
        kwargs[name] = tmp_path if "dir" in name or "path" in name else None
    logger = logging_utils.TrainingLogger(**kwargs)
    logger.log_epoch(1, {"val_rna_seq_loss": 0.5, "train_loss": 1.0, "val_loss": 0.7,
                         "rna_seq_1bp_count_pearson_r": 0.3},
                     learning_rate=1e-4, is_best=True)
    path = next(tmp_path.rglob("epoch_log.csv"))
    with open(path) as f:
        header = next(csv.reader(f))
    assert header[:7] == ["sample", "epoch", "train_loss", "val_loss",
                          "learning_rate", "is_best", "timestamp"]


def test_cls_auprc_subsamples_background():
    from alphagenome_ft.finetune.train import _cls_active_and_background

    target = np.zeros((100, 5), np.float32)
    target[:, 4] = 1.0
    target[[3, 50], 4] = 0.0
    target[3, 0] = target[50, 2] = 1.0
    logits = np.random.default_rng(0).normal(size=(100, 5)).astype(np.float32)
    lg, tg = _cls_active_and_background(logits, target, np.random.default_rng(1))
    assert tg.shape[0] == 4                      # 2 sites + 2 background, not all 100
    assert tg[:2, :4].any(axis=-1).all() and not tg[2:, :4].any()


def test_custom_head_shadows_pretrained_same_name():
    from alphagenome_ft.optimizer_utils import label_params_for_trainable_heads

    params = {
        "head/rna_seq/resolution_1/multi_organism_linear": {"w": jnp.ones((2, 2))},
        "alphagenome/head/rna_seq/resolution_1/multi_organism_linear": {"w": jnp.ones((2, 2))},
        "alphagenome/head/my_task": {"w": jnp.ones((2, 2))},
    }
    labels = label_params_for_trainable_heads(params, ["rna_seq", "my_task"])
    assert labels["head/rna_seq/resolution_1/multi_organism_linear"]["w"] == "head"
    assert labels["alphagenome/head/rna_seq/resolution_1/multi_organism_linear"]["w"] == "frozen"
    assert labels["alphagenome/head/my_task"]["w"] == "head"     # no top-level copy: still trainable


def test_epoch_log_widens_header_without_shifting(tmp_path):
    import csv

    from alphagenome_ft import logging_utils

    logging_utils.TrainingLogger(tmp_path).log_epoch(1, {"train_loss": 1.0, "a": 1.0})
    resumed = logging_utils.TrainingLogger(tmp_path)          # new process after a resume
    resumed.log_epoch(2, {"train_loss": 2.0, "b": 2.0})
    with open(next(tmp_path.rglob("epoch_log.csv"))) as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["a"] == "1.0" and rows[0]["b"] == ""
    assert rows[1]["a"] == "" and rows[1]["b"] == "2.0" and rows[1]["train_loss"] == "2.0"
