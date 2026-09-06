#!/usr/bin/env python
"""JAX splice-modality finetuning script — thin shim over the packaged implementation.

The implementation lives in the package, so it works from a wheel install
too (no repo checkout, no sys.path games):

    alphagenome_ft.finetune.args    — the flags
    alphagenome_ft.finetune.runner  — the training code

Run with:
    python scripts/finetune.py --mode lora ...

See ``python scripts/finetune.py --help`` for all flags and their defaults
(chosen to match alphagenome-pytorch's equivalent flags where one exists).
"""

from alphagenome_ft.finetune.runner import main

if __name__ == "__main__":
    main()
