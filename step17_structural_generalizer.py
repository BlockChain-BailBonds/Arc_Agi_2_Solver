#!/usr/bin/env python3
# ============================================================
# Structural Pattern Generalizer — step17_structural_generalizer.py
# Extends ARC solver with minimal geometric reasoning.
# ============================================================

import numpy as np

def detect_structure(inp: np.ndarray, out: np.ndarray) -> np.ndarray:
    """
    Detect simple spatial or structural transformations between input and output grids.
    Currently supports a minimal translation inference as base generalization test.
    """
    mask_diff = inp != out
    if np.any(mask_diff):
        # Basic pattern translation test (shift one step right)
        return np.roll(inp, shift=(1, 0), axis=(0, 1))
    return inp
