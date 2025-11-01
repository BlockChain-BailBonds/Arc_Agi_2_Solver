#!/usr/bin/env python3
# step0_utils.py — core numeric and progress utilities

import numpy as np
import sys
import time

def to_np(grid):
    """Convert grid (list of lists) to NumPy array."""
    return np.array(grid, dtype=np.int64)

def to_grid(array):
    """Convert NumPy array back to grid (list of lists)."""
    return array.tolist()

def fit_to_shape(a: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Resize by nearest replication."""
    in_h, in_w = a.shape
    out = np.zeros((out_h, out_w), dtype=a.dtype)
    for i in range(out_h):
        for j in range(out_w):
            out[i, j] = a[i * in_h // out_h, j * in_w // out_w]
    return out

def scale_nearest(a: np.ndarray, scale: float) -> np.ndarray:
    """Scale an array by nearest-neighbor interpolation."""
    out_h = max(1, int(round(a.shape[0] * scale)))
    out_w = max(1, int(round(a.shape[1] * scale)))
    return fit_to_shape(a, out_h, out_w)

def ensure_integrity(a: np.ndarray) -> np.ndarray:
    """Clip invalid values and ensure 2D int64 array."""
    a = np.nan_to_num(a, nan=0).astype(np.int64)
    if a.ndim != 2:
        a = a.reshape((a.shape[0], -1))
    return np.clip(a, 0, 9)

def compute_confidence(pred: np.ndarray, target: np.ndarray) -> float:
    """Compute confidence as elementwise accuracy."""
    pred = ensure_integrity(pred)
    target = ensure_integrity(target)
    if pred.shape != target.shape:
        return 0.0
    total = pred.size
    correct = np.sum(pred == target)
    return round(float(correct) / total, 3)

def progress_bar(conf: float, width: int = 40):
    """Print inline confidence progress bar."""
    filled = int(conf * width)
    bar = "█" * filled + "-" * (width - filled)
    sys.stdout.write(f"\r[{bar}] {conf*100:5.1f}%")
    sys.stdout.flush()
