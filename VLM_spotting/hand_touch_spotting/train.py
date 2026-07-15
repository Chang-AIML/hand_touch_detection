#!/usr/bin/env python3
"""Training/eval entry point. Thin shim so the intuitive `python train.py ...` works; the real
driver lives at dpc/train_mixed.py (its path is pinned by the DPC cluster YAMLs, so it stays put)."""
from dpc.train_mixed import main

if __name__ == "__main__":
    main()
