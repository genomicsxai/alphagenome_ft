"""CLI flags for the splice-modality probing/LoRA finetuning driver.

Kept dependency-light at module scope (only ``argparse``/``pathlib``) so
``--help`` works without a full JAX install, mirroring
``alphagenome_pytorch.extensions.finetuning.args``.

Every default here is chosen to match ``alphagenome-pytorch``'s
``extensions/finetuning/args.py`` defaults for the equivalent flag, so a JAX
run and a PyTorch run started with no overrides are paper-equivalent. See
``workarounds.py`` and ``runner.py`` docstrings for the specific bugs/defaults
this was ported to fix.

Note: there is intentionally no ``--junction-loss`` flag here, unlike
PyTorch's original/normalized/sparse switch. The real
``SpliceSitesJunctionHead.loss`` in ``alphagenome_research`` has a single
fixed formula that already matches PyTorch's "normalized" variant, so there
is nothing to select between.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-path", required=True, type=Path,
                         help="Local AlphaGenome JAX checkpoint dir (e.g. from "
                              "a Kaggle model-version download — do not point "
                              "this at Kaggle directly from a "
                              "network-isolated node).")
    parser.add_argument("--genome", required=True, type=Path, help="Reference FASTA.")
    parser.add_argument("--train-bed", required=True, type=Path)
    parser.add_argument("--val-bed", required=True, type=Path)
    parser.add_argument("--bigwig", required=True, nargs="+",
                         help="Bigwig files driving a single rna_seq head's "
                              "targets (one track per file). Matches "
                              "alphagenome-pytorch's --modality rna_seq --bigwig.")
    parser.add_argument("--track-means-samples", type=int, default=None,
                         help="Number of --train-bed windows to sample when "
                              "computing each rna_seq bigwig track's "
                              "nonzero_mean (default: all). Matches "
                              "alphagenome-pytorch's --track-means-samples — "
                              "the real predefined rna_seq head "
                              "(alphagenome_research.model.heads) rescales "
                              "predictions/targets by this on every forward "
                              "pass when present; omit only to fall back to "
                              "no scaling (all-ones).")
    parser.add_argument("--star-junctions", required=True, nargs="+",
                         help="STAR SJ.out.tab files, one per sample.")
    parser.add_argument("--ssu", nargs="+", default=None,
                         help="Optional per-sample SSU parquet files, same "
                              "order as --star-junctions.")
    parser.add_argument("--gtf", default=None,
                         help="Optional canonical splice-site GTF/parquet "
                              "(annotation-only sites, zero usage).")
    parser.add_argument("--junction-position-source", choices=["annotated", "predicted"],
                         default="annotated",
                         help="Source of splice-site positions passed to the "
                              "junction head. Matches alphagenome-pytorch's "
                              "--junction-position-source default (annotated).")
    parser.add_argument("--rope-init", choices=["none", "truncated_normal", "zeros"],
                         default="none",
                         help="How to initialize SpliceSitesJunctionHead's RoPE "
                              "scale/offset ('embeddings') parameter. 'none' "
                              "(default) skips any manual reinit and trusts "
                              "the fresh Haiku init a real model construction "
                              "already gives this parameter -- confirmed "
                              "empirically (alphagenome_research 0.3.0+) to "
                              "already be TruncatedNormal(0.1)-distributed "
                              "with normal nonzero gradients from the start; "
                              "the zero-init dead-gradient bug this flag "
                              "originally worked around (alphagenome_research "
                              "0.1.0) was fixed upstream and no longer exists "
                              "with the currently required package version. "
                              "'truncated_normal' manually reinits to a fresh "
                              "TruncatedNormal(std) sample -- statistically "
                              "indistinguishable from 'none' at "
                              "--rope-init-std 0.1, kept only for explicit "
                              "ablation. 'zeros' reproduces the old buggy "
                              "init for ablation only -- do not use otherwise.")
    parser.add_argument("--rope-init-std", type=float, default=0.1,
                         help="Stddev for --rope-init truncated_normal (ignored "
                              "when --rope-init is 'none', the default). 0.1 "
                              "matches alphagenome_research's own real "
                              "TruncatedNormal(0.1) init for this parameter "
                              "(heads.py) and alphagenome-pytorch's hardcoded "
                              "std=0.1 ('matches the JAX reference init and "
                              "the pretrained weight distribution').")
    parser.add_argument("--sequence-length", type=int, default=1048576)
    parser.add_argument("--max-splice-sites", type=int, default=256,
                         help="Max splice sites per role (donor/acceptor x "
                              "strand) fed to the junction head. Matches "
                              "alphagenome-pytorch's hardcoded default of the "
                              "same name in datasets.py (not exposed as a "
                              "PyTorch CLI flag, but the same value, 256).")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4,
                         help="Global batch size, sharded across --num-devices. "
                              "Combined with --gradient-accumulation-steps, the "
                              "effective batch size per optimizer step is "
                              "batch_size * gradient_accumulation_steps, matching "
                              "the PyTorch run's batch=1 x num_gpus x grad_accum "
                              "scheme.")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1,
                         help="Number of --batch-size batches to average grads "
                              "over per optimizer step. Use this (rather than a "
                              "larger --batch-size) to match a PyTorch DDP run's "
                              "effective batch size without risking OOM at 1Mb "
                              "sequence length on a single device: e.g. "
                              "--batch-size 1 --gradient-accumulation-steps 64 "
                              "on one GPU reproduces the same 64-example "
                              "gradient average as a PyTorch run's batch=1 x "
                              "4 GPUs x grad_accum=16.")
    parser.add_argument("--num-devices", type=int, default=4)
    parser.add_argument("--gradient-checkpointing", action="store_true",
                         help="Mirrors alphagenome-pytorch's --gradient-checkpointing: "
                              "wrap the backbone forward pass in hk.remat so its "
                              "activations are recomputed on the backward pass instead "
                              "of retained, trading compute for memory. For "
                              "--mode linear-probe this still matters even though the "
                              "backbone is detached (stop_gradient'd) — JAX/XLA does "
                              "not elide saving the backbone's forward activations "
                              "just because a downstream stop_gradient means they end "
                              "up unused. For --mode lora it is even more directly "
                              "needed: the backbone is NOT detached there (gradients "
                              "must reach the LoRA adapters through it).")
    parser.add_argument("--mode", choices=["linear-probe", "lora"], default="linear-probe",
                         help="'linear-probe' (default): freeze + detach the backbone, "
                              "train only the heads. 'lora': freeze the backbone but do "
                              "NOT detach it; instead monkeypatch alphagenome_research's "
                              "MHABlock so the --lora-targets q/v projections get a "
                              "trainable low-rank adapter (see "
                              "alphagenome_ft.lora.install_mha_backbone_lora), and train "
                              "those adapters + heads. Mirrors alphagenome-pytorch's "
                              "--mode lora.")
    parser.add_argument("--lora-rank", type=int, default=8,
                         help="LoRA rank for --mode lora. Matches alphagenome-pytorch's "
                              "--lora-rank default (8).")
    parser.add_argument("--lora-alpha", type=float, default=16.0,
                         help="LoRA alpha scaling for --mode lora; effective adapter "
                              "scale is alpha/rank. Matches alphagenome-pytorch's "
                              "--lora-alpha default (16).")
    parser.add_argument("--lora-targets", default="q_proj,v_proj",
                         help="Comma-separated backbone attention projections to adapt "
                              "for --mode lora, using alphagenome-pytorch's naming "
                              "(q_proj/k_proj/v_proj — translated internally to this "
                              "JAX model's q_layer/k_layer/v_layer). Matches "
                              "alphagenome-pytorch's --lora-targets default.")
    parser.add_argument("--max-train-steps", type=int, default=None,
                         help="Optional global cap on optimizer updates, for "
                              "quick smoke-test runs before a full finetune.")
    parser.add_argument("--save-every-steps", type=int, default=None,
                         help="Also save a 'last' checkpoint (+ opt_state, "
                              "train_state.json) every this many optimizer "
                              "steps, not just at epoch end. Mirrors "
                              "alphagenome-pytorch's --save-every-steps.")
    parser.add_argument("--max-grad-norm", type=float, default=1.0,
                         help="Clip gradients to this global norm before the "
                              "optimizer update. Matches alphagenome-pytorch's "
                              "--max-grad-norm (also hardcoded to 1.0 there). "
                              "Pass 0 or a negative value to disable clipping.")
    parser.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16",
                         help="Compute dtype for the trunk AND heads (params stay "
                              "fp32). Matches alphagenome-pytorch's --dtype "
                              "(same default, bfloat16).")
    parser.add_argument("--usage-num-segments", type=int, default=8,
                         help="Split the sequence into this many equal chunks for the "
                              "splice_site_usage BCE loss, summing per-chunk masked means "
                              "instead of one global mean (upweights sparse splice-site "
                              "regions) -- see CustomAlphaGenomeModel.set_usage_num_segments "
                              "in custom_model.py. Matches alphagenome-pytorch's "
                              "--num-segments default (8). Pass 1 to disable (single "
                              "global mean).")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--organism", default="HOMO_SAPIENS")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--resume", choices=["auto", "none"], default="auto",
                         help="'auto' (default): if <output-dir>/<run-name>/last "
                              "already has a train_state.json (from a previous, "
                              "possibly-preempted invocation with the same "
                              "--output-dir/--run-name), resume model weights "
                              "from it via alphagenome_ft.load_checkpoint and "
                              "continue epoch/global_step bookkeeping from "
                              "there instead of --rope-init reinit + fresh "
                              "pretrained weights. 'none' always starts fresh. "
                              "Also restores optimizer (Adam) state from an "
                              "opt_state sidecar there, if present. Mirrors "
                              "alphagenome-pytorch's --resume auto.")
    parser.add_argument("--log-every", type=int, default=50,
                         help="Log per-head training loss (to training_log.csv "
                              "and W&B) every this many optimizer steps. Matches "
                              "alphagenome-pytorch's --log-every default.")
    parser.add_argument("--wandb", action="store_true",
                         help="Enable Weights & Biases logging, mirroring "
                              "alphagenome-pytorch's --wandb.")
    parser.add_argument("--wandb-project", default="alphagenome-ft")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--pretrained-head-samples", type=str, default="splice_site:0",
                         help="Per-head control over pretrained-vs-random weight init, "
                              "mirroring alphagenome-pytorch's --pretrained-head-samples. "
                              "Format: 'modality[@resolution]:idx,...' where modality is "
                              "one of rna_seq/splice_site/splice_usage/splice_junctions, "
                              "and idx is either a single integer (broadcast that "
                              "pretrained track to every output track of the head) or a "
                              "'|'-separated list of integers/NA with one entry per output "
                              "track (validated against the head's actual track count). "
                              "Use 'NA' to keep random initialization for a modality or a "
                              "specific output track. '@resolution' is only meaningful for "
                              "rna_seq (this repo's custom head is built with resolutions=[1] "
                              "by default; requesting an unbuilt resolution errors). "
                              "Modalities not listed keep random initialization. The "
                              "organism is taken from --organism. For splice_site the index "
                              "is ignored (its 5-class output has no per-track structure); "
                              "for splice_junctions the index selects pretrained RoPE "
                              "tissues specifically (its main linear projects trunk "
                              "embeddings into a shared representation, not a per-tissue "
                              "one, so that part is always copied whole, organism-only). "
                              "Default 'splice_site:0' reproduces this repo's prior "
                              "hardcoded behavior.")
    return parser.parse_args(argv)
