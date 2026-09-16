"""Splice-modality probing/LoRA finetuning driver, orchestration layer.

Trains all 4 predefined heads jointly — rna_seq (bigwig-derived) plus the
three splice heads (STAR/SSU-derived) — by combining
``alphagenome_ft.finetune.data.BigWigDataModule`` and
``finetune.splice_data.SpliceDataModule`` via ``combined_data.CombinedDataModule``.
JAX/alphagenome_research analogue of alphagenome-pytorch's finetuning
``runner.py``; see ``args.py`` for flag definitions and defaults.

Gradient accumulation: ``alphagenome_ft.finetune.train.train()`` originally
took one full optimizer step per data_module batch with no accumulation
loop; it now accepts ``gradient_accumulation_steps`` so a small per-device
batch size can still reach a large effective batch (e.g. batch=1,
gradient-accumulation-steps=64) without risking OOM at 1Mb sequence length
on a single device.
"""

from __future__ import annotations

import json
import random

import numpy as np

from alphagenome_ft.finetune.args import parse_args, parse_modality_weights
from alphagenome_ft.finetune.combined_data import (
    CombinedDataModule,
    compute_track_means,
    load_interval_list,
)
from alphagenome_ft.finetune.workarounds import (
    apply_pretrained_head_samples,
    reinit_junction_rope_embeddings,
)


def main(args=None) -> None:
    if args is None:
        args = parse_args()

    # Heavy imports deferred past argparse so --help works without a full JAX install.
    from alphagenome.models import dna_model as ag_dna_model
    from alphagenome_research.model import dna_model as research_dna_model

    from alphagenome_ft import create_model_with_heads, load_checkpoint
    from alphagenome_ft import lora as lora_lib
    from alphagenome_ft.finetune import config as ft_config
    from alphagenome_ft.finetune.data import BigWigDataModule
    from alphagenome_ft.finetune.splice_data import SpliceDataModule
    from alphagenome_ft.finetune.train import register_predefined_heads, train as run_train

    organism_index = research_dna_model.convert_to_organism_index(
        getattr(ag_dna_model.Organism, args.organism)
    )

    random.seed(args.seed)
    np.random.seed(args.seed)

    lora_enabled = args.mode == "lora"
    detach_backbone = not lora_enabled
    install_backbone_patches = None
    if lora_enabled:
        lora_cfg = lora_lib.BackboneLoRAConfig.from_pytorch_style_targets(
            args.lora_targets.split(","), rank=args.lora_rank, alpha=args.lora_alpha,
        )
        print(f"Mode: lora — will install backbone LoRA adapters "
              f"(rank={lora_cfg.rank}, alpha={lora_cfg.alpha}, targets={lora_cfg.targets}) "
              f"right after the base pretrained checkpoint restores (not before: an "
              f"active patch makes that restore's own orbax target tree mismatch the "
              f"saved checkpoint's structure, since it doesn't have LoRA params).")

        def install_backbone_patches() -> None:
            lora_lib.install_mha_backbone_lora(lora_cfg)

    head_ids = {
        "splice_sites_classification": "splice_site",
        "splice_sites_usage": "splice_usage",
        "splice_sites_junction": "splice_junctions",
    }

    heads_cfg = []
    for kind, head_id in head_ids.items():
        entry = {
            "id": head_id,
            "source": "predefined",
            "kind": kind,
            "star_junctions": args.star_junctions,
            "max_splice_sites": args.max_splice_sites,
        }
        if args.ssu is not None:
            entry["ssu"] = args.ssu
        if args.gtf is not None:
            entry["gtf"] = args.gtf
        if kind == "splice_sites_junction":
            entry["junction_position_source"] = args.junction_position_source
            if args.junction_position_source == "predicted":
                entry["classification_head_id"] = head_ids["splice_sites_classification"]
        heads_cfg.append(entry)

    track_means = compute_track_means(
        args.bigwig, args.train_bed, args.sequence_length, args.track_means_samples,
    )
    heads_cfg.append({
        "id": "rna_seq",
        "source": "predefined",
        "kind": "rna_seq",
        "targets": [
            {"path": str(bw), "nonzero_mean": mean}
            for bw, mean in zip(args.bigwig, track_means)
        ],
        # Native GenomeTracksHeadConfig defaults to resolutions=[1, 128] and
        # GenomeTracksHead.loss() sums a separate multinomial-loss term per
        # resolution (sum-pooling 1bp targets to 128bp for the second term) -
        # without this override the rna_seq head trains against an extra,
        # unwanted 128bp loss term a PyTorch reference with
        # modality_resolutions: {"rna_seq": [1]} never computes.
        "resolutions": [1],
    })

    specs = ft_config.prepare_head_specs(
        {"heads": heads_cfg}, organism=args.organism,
    )
    ft_config.validate_head_specs(specs)
    register_predefined_heads(specs)

    checkpoint_dir = args.output_dir / args.run_name
    resume_dir = checkpoint_dir / "last"
    do_resume = args.resume == "auto" and (resume_dir / "train_state.json").exists()

    if not do_resume:
        # config.json alongside the checkpoint dir, matching
        # alphagenome-pytorch's convention (config.json colocated with the
        # checkpoint) -- lets downstream prediction/eval code use the SAME
        # dtype the checkpoint was actually trained under by default, rather
        # than requiring the caller to know/pass the right --dtype and risk
        # silently mismatching it. Written once at the start of a fresh run
        # (not on resume, since it's already correct from the first attempt
        # and the true value can't change mid-run).
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        with open(checkpoint_dir / "config.json", "w") as f:
            json.dump({"dtype": args.dtype, "mode": args.mode}, f, indent=2)

    if do_resume:
        print(f"Found existing checkpoint at {resume_dir} — resuming from it "
              f"(skipping fresh pretrained-weight load and --rope-init reinit, "
              f"both of which would clobber already-trained head weights).")
        model = load_checkpoint(
            resume_dir,
            base_checkpoint_path=args.checkpoint_path,
            init_seq_len=args.sequence_length,
            detach_backbone=detach_backbone,
            gradient_checkpointing=args.gradient_checkpointing,
            install_backbone_patches=install_backbone_patches,
            dtype=args.dtype,
        )
    else:
        print("Loading pretrained AlphaGenome JAX model from local checkpoint "
              f"cache: {args.checkpoint_path}")
        model = create_model_with_heads(
            heads=[spec.head_id for spec in specs],
            checkpoint_path=args.checkpoint_path,
            init_seq_len=args.sequence_length,
            # For --mode linear-probe, detach_backbone=True cuts every
            # gradient path into the backbone at a single point (the trunk's
            # output embeddings) -- without this, jax.grad backprops through
            # the full frozen trunk every step (train.py's grad_step also
            # stop_gradients each frozen weight individually as a second,
            # complementary safety net, but this embeddings-level cut is what
            # makes a single-GPU probing run tractable memory-wise at all).
            # For --mode lora, detach_backbone must be False: gradients need
            # to reach the LoRA adapters through the (otherwise frozen)
            # backbone, so train.py's per-weight stop_gradient is the only
            # thing preventing a full-backbone backward pass there.
            detach_backbone=detach_backbone,
            gradient_checkpointing=args.gradient_checkpointing,
            install_backbone_patches=install_backbone_patches,
            dtype=args.dtype,
        )

        if args.rope_init == "truncated_normal":
            print(f"Re-initializing junction head RoPE embeddings "
                  f"(std={args.rope_init_std}) — explicit ablation only; the "
                  f"fresh model's own init (--rope-init none, the default) "
                  f"is already TruncatedNormal-ish and correct, see args.py "
                  f"docstring.")
            reinit_junction_rope_embeddings(
                model, head_ids["splice_sites_junction"], std=args.rope_init_std, seed=args.seed,
            )
        elif args.rope_init == "zeros":
            print("Re-initializing junction head RoPE embeddings to exact "
                  "zeros — reproduces the since-fixed upstream dead-gradient "
                  "bug on purpose, ablation only, see args.py docstring.")
            reinit_junction_rope_embeddings(
                model, head_ids["splice_sites_junction"], std=0.0, seed=args.seed,
            )

        apply_pretrained_head_samples(model, args.pretrained_head_samples, organism_index)

    if args.usage_num_segments > 1:
        model.set_usage_num_segments(
            head_ids["splice_sites_usage"], args.usage_num_segments,
        )

    intervals = {
        "train": load_interval_list(args.train_bed, window_size=args.sequence_length),
        "valid": load_interval_list(args.val_bed, window_size=args.sequence_length),
    }

    # Pre-filter to chromosomes common to all --bigwig files using
    # BigWigDataModule's own helper, and construct BOTH data modules from
    # this identical, already-filtered interval dict (not the raw intervals
    # above) — this is what makes CombinedDataModule's lock-step zip safe;
    # see its docstring.
    rna_seq_spec = next(spec for spec in specs if spec.head_id == "rna_seq")
    intervals = BigWigDataModule._filter_intervals_by_bigwig_chromosomes(
        intervals, [rna_seq_spec],
    )

    # SpliceDataModule's own head-kind vocabulary ("splice_sites",
    # "splice_site_usage", "splice_junctions") differs from
    # finetune.config's SPLICE_KINDS naming
    # ("splice_sites_classification"/"splice_sites_usage"/"splice_sites_junction")
    # used above for prepare_head_specs — map explicitly rather than assuming
    # they line up.
    data_head_kinds = {
        "splice_sites": head_ids["splice_sites_classification"],
        "splice_site_usage": head_ids["splice_sites_usage"],
        "splice_junctions": head_ids["splice_sites_junction"],
    }

    splice_module = SpliceDataModule(
        intervals=intervals,
        fasta_path=args.genome,
        star_junction_files=args.star_junctions,
        head_kinds=data_head_kinds,
        batch_size=args.batch_size,
        shuffle=True,
        ssu_files=args.ssu,
        gtf_file=args.gtf,
        max_splice_sites=args.max_splice_sites,
        drop_last=args.num_devices > 1,
        emit_raw_junction_events=(args.junction_position_source == "predicted"),
        # Always False, not a CLI flag: CombinedDataModule requires both
        # underlying modules to share the identical window list (see its
        # docstring), and this also matches a full-fold reference run, which
        # trains over the entire split with no junction-presence filter.
        filter_to_junctions=False,
    )
    bigwig_module = BigWigDataModule(
        intervals=intervals,
        fasta_path=args.genome,
        head_specs=[rna_seq_spec],
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=args.num_devices > 1,
    )
    data_module = CombinedDataModule(bigwig_module, splice_module)

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
        lora_enabled=lora_enabled,
        checkpoint_dir=checkpoint_dir,
        organism=args.organism,
        num_devices=args.num_devices,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        resume_from=resume_dir if do_resume else None,
        save_every_steps=args.save_every_steps,
        gradient_clip_global_norm=args.max_grad_norm if args.max_grad_norm > 0 else None,
        verbose=True,
        log_every=args.log_every,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=args.run_name,
        modality_weights=parse_modality_weights(args.modality_weights),
        warmup_steps=args.warmup_steps,
        lr_schedule=args.lr_schedule,
        compute_pearson=not args.no_val_pearson,
        metrics_per_sample=args.metrics_per_sample,
    )

    # Distinct from checkpoint_dir/{last,best} (which --resume auto reads/
    # writes across invocations): callers that declare a Snakemake-style
    # output should point at this marker, never at last/ itself — see
    # CLAUDE.md's Snakemake gotcha about resumable checkpoints as rule
    # outputs.
    (checkpoint_dir / "training_complete.marker").touch()
    print(f"Done! Checkpoints written under {checkpoint_dir}")


if __name__ == "__main__":
    main()
