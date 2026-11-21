"""Finetuning driver: ``python -m alphagenome_ft.finetune``.

Heads come from ``--targets-config`` (``finetune.config`` YAML/JSON format) or
the ``--bigwig``/``--star-junctions`` shorthand (one ``rna_seq`` head plus the
three splice heads). All heads are trained jointly from one
:class:`~alphagenome_ft.finetune.data.FinetuneDataModule`. JAX/
alphagenome_research analogue of alphagenome-pytorch's finetuning runner; see
``args.py`` for flags and defaults.
"""

from __future__ import annotations

import json
import random
from typing import Any

import numpy as np

from alphagenome_ft.finetune.args import parse_args, parse_modality_weights
from alphagenome_ft.finetune.head_init import (
    apply_pretrained_head_samples,
    modality_head_ids,
    reinit_junction_rope_embeddings,
)

# Head ids used by the shorthand flags.
SHORTHAND_HEAD_IDS = {
    "splice_sites_classification": "splice_site",
    "splice_sites_usage": "splice_usage",
    "splice_sites_junction": "splice_junctions",
}


def shorthand_targets_config(args) -> dict[str, Any]:
    """Head config built from the --bigwig/--star-junctions shorthand flags."""
    heads: list[dict[str, Any]] = []
    for kind, head_id in SHORTHAND_HEAD_IDS.items():
        entry: dict[str, Any] = {
            "id": head_id,
            "source": "predefined",
            "kind": kind,
            "star_junctions": list(args.star_junctions),
            "max_splice_sites": args.max_splice_sites,
            # Train over every window of the split, as a full-fold run does.
            "filter_to_junctions": False,
        }
        if args.ssu is not None:
            entry["ssu"] = list(args.ssu)
        if args.gtf is not None:
            entry["gtf"] = args.gtf
        if kind == "splice_sites_junction":
            entry["junction_position_source"] = args.junction_position_source
            if args.junction_position_source == "predicted":
                entry["classification_head_id"] = SHORTHAND_HEAD_IDS["splice_sites_classification"]
                entry["junction_top_k"] = args.junction_top_k
                entry["splice_site_threshold"] = args.splice_site_threshold
        heads.append(entry)
    heads.append({
        "id": "rna_seq",
        "source": "predefined",
        "kind": "rna_seq",
        "targets": [{"path": str(bw)} for bw in args.bigwig],
        # GenomeTracksHeadConfig defaults to resolutions [1, 128] and sums a loss
        # term per resolution; alphagenome-pytorch trains rna_seq at 1 bp only.
        "resolutions": [1],
    })
    return {"heads": heads}


def main(args=None) -> None:
    if args is None:
        args = parse_args()

    # Heavy imports deferred past argparse so --help works without a full JAX install.
    from alphagenome.models import dna_model as ag_dna_model
    from alphagenome_research.model import dna_model as research_dna_model

    from alphagenome_ft import create_model_with_heads, load_checkpoint
    from alphagenome_ft.finetune import config as ft_config
    from alphagenome_ft.finetune.data import (
        FinetuneDataModule,
        build_fasta_index,
        compute_track_nonzero_means,
        load_split_intervals,
    )
    from alphagenome_ft.finetune.train import register_predefined_heads, train as run_train

    if args.num_devices is None:
        import jax

        args.num_devices = jax.local_device_count()
        print(f"Using all {args.num_devices} visible device(s) (--num-devices not set).")

    organism_index = research_dna_model.convert_to_organism_index(
        getattr(ag_dna_model.Organism, args.organism)
    )

    random.seed(args.seed)
    np.random.seed(args.seed)

    lora_enabled = args.mode == "lora"
    detach_backbone = not lora_enabled
    backbone_lora_config = None
    if lora_enabled:
        from alphagenome_ft.backbone_lora import BackboneLoRAConfig

        backbone_lora_config = BackboneLoRAConfig(
            rank=args.lora_rank,
            alpha=args.lora_alpha,
            targets=tuple(t.strip() for t in args.lora_targets.split(",") if t.strip()),
        )
        print(f"Mode: lora - transformer LoRA adapters "
              f"(rank={backbone_lora_config.rank}, alpha={backbone_lora_config.alpha}, "
              f"targets={backbone_lora_config.targets}).")

    targets_config = ft_config.load_targets_config(
        args.targets_config if args.targets_config is not None else shorthand_targets_config(args)
    )

    # Windows: one BED per split, recentred, off-chromosome windows skipped;
    # then restricted to chromosomes every BigWig covers, before track means
    # are estimated from the training windows.
    intervals = load_split_intervals(
        {"train": args.train_bed, "valid": args.val_bed},
        args.sequence_length,
        chrom_sizes=build_fasta_index(args.genome),
    )
    bigwig_specs = [
        spec for spec in ft_config.prepare_head_specs(targets_config, organism=args.organism)
        if spec.splice_source is None
    ]
    if bigwig_specs:
        intervals = FinetuneDataModule._filter_intervals_by_bigwig_chromosomes(
            intervals, bigwig_specs,
        )

    if args.targets_config is None:
        # Shorthand: per-track means with no strand pairing, as
        # alphagenome-pytorch without --strand-pairs. (A config without
        # nonzero_mean gets prepare_head_specs' strand-aware estimate instead.)
        for head in targets_config["heads"]:
            if head.get("kind") == "rna_seq":
                means = compute_track_nonzero_means(
                    [t["path"] for t in head["targets"]], intervals["train"],
                    max_samples=args.track_means_samples,
                )
                head["targets"] = [
                    {**t, "nonzero_mean": float(m)} for t, m in zip(head["targets"], means)
                ]
                print(f"rna_seq nonzero_mean per track: {[round(float(m), 6) for m in means]}")
    specs = ft_config.prepare_head_specs(
        targets_config,
        organism=args.organism,
        nonzero_mean_intervals=intervals["train"],
        nonzero_mean_max_samples=args.track_means_samples,
    )
    ft_config.validate_head_specs(specs)
    register_predefined_heads(specs)
    head_ids = modality_head_ids(specs)

    checkpoint_dir = args.output_dir / args.run_name
    resume_dir = checkpoint_dir / "last"
    do_resume = args.resume == "auto" and (resume_dir / "train_state.json").exists()

    if not do_resume:
        # config.json next to the checkpoints records the dtype/mode they were
        # trained under, for downstream prediction/eval.
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        with open(checkpoint_dir / "config.json", "w") as f:
            json.dump({"dtype": args.dtype, "mode": args.mode}, f, indent=2)

    if do_resume:
        print(f"Found existing checkpoint at {resume_dir} — resuming from it "
              f"(skipping pretrained-weight load and head re-initialization).")
        model = load_checkpoint(
            resume_dir,
            base_checkpoint_path=args.checkpoint_path,
            init_seq_len=args.sequence_length,
            detach_backbone=detach_backbone,
            gradient_checkpointing=args.gradient_checkpointing,
            backbone_lora_config=backbone_lora_config,
            dtype=args.dtype,
        )
    else:
        print("Loading pretrained AlphaGenome JAX model from local checkpoint "
              f"cache: {args.checkpoint_path}")
        model = create_model_with_heads(
            heads=[spec.head_id for spec in specs],
            checkpoint_path=args.checkpoint_path,
            init_seq_len=args.sequence_length,
            # linear-probe: detach the backbone at its output embeddings so no
            # gradient reaches it. lora: gradients must reach the adapters
            # through the frozen backbone, so it stays attached.
            detach_backbone=detach_backbone,
            gradient_checkpointing=args.gradient_checkpointing,
            backbone_lora_config=backbone_lora_config,
            dtype=args.dtype,
        )

        junction_head = head_ids.get("splice_junctions")
        if junction_head is not None and args.rope_init != "none":
            std = args.rope_init_std if args.rope_init == "truncated_normal" else 0.0
            print(f"Re-initializing junction head RoPE embeddings ({args.rope_init}, std={std}).")
            reinit_junction_rope_embeddings(model, junction_head, std=std, seed=args.seed)

        pretrained_head_samples = args.pretrained_head_samples
        if pretrained_head_samples is None and "splice_site" in head_ids:
            pretrained_head_samples = "splice_site:0"
        if pretrained_head_samples:
            apply_pretrained_head_samples(
                model, pretrained_head_samples, organism_index, head_ids,
            )

    if args.splice_num_segments > 1:
        # As alphagenome-pytorch's --splice-num-segments: classification + usage.
        for modality in ("splice_site", "splice_usage"):
            if modality in head_ids:
                model.set_splice_num_segments(head_ids[modality], args.splice_num_segments)

    # --batch-size is per device (as alphagenome-pytorch's per-GPU batch).
    global_batch_size = args.batch_size * args.num_devices
    data_module = FinetuneDataModule(
        intervals=intervals,
        fasta_path=args.genome,
        head_specs=specs,
        batch_size=global_batch_size,
        shuffle=True,
        drop_last=args.num_devices > 1,
    )

    modality_weights = {}
    for modality, weight in parse_modality_weights(args.modality_weights).items():
        if modality not in head_ids:
            raise ValueError(f"--modality-weights names {modality!r}, which no configured head serves.")
        modality_weights[head_ids[modality]] = weight

    run_train(
        model,
        data_module,
        specs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        num_epochs=args.epochs,
        seed=args.seed,
        max_train_steps=args.max_train_steps,
        heads_only=True,
        train_lora=lora_enabled,
        checkpoint_dir=checkpoint_dir,
        organism=args.organism,
        num_devices=args.num_devices,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        resume_from=resume_dir if do_resume else None,
        save_every_steps=args.save_every_steps,
        max_grad_norm=args.max_grad_norm if args.max_grad_norm > 0 else None,
        verbose=True,
        log_every=args.log_every,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=args.run_name,
        modality_weights=modality_weights,
        warmup_steps=args.warmup_steps,
        lr_schedule=args.lr_schedule,
        compute_pearson=not args.no_val_pearson,
        metrics_per_sample=args.metrics_per_sample,
    )

    # Marker for workflow managers: declare this as the rule output, never
    # last/ itself (a rerun-incomplete would wipe resumable progress).
    (checkpoint_dir / "training_complete.marker").touch()
    print(f"Done! Checkpoints written under {checkpoint_dir}")


if __name__ == "__main__":
    main()
