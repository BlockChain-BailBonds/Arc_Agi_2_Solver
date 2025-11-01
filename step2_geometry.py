#!/usr/bin/env python3
# step2_geometry.py — geometric pattern analyzer for ARC tasks

import numpy as np

def rotate90(a: np.ndarray, k: int = 1) -> np.ndarray:
    """Rotate grid 90° × k times clockwise."""
    return np.rot90(a, -k)

def flip_x(a: np.ndarray) -> np.ndarray:
    """Flip grid horizontally."""
    return np.fliplr(a)

def flip_y(a: np.ndarray) -> np.ndarray:
    """Flip grid vertically."""
    return np.flipud(a)

def transpose(a: np.ndarray) -> np.ndarray:
    """Transpose grid (swap axes)."""
    return a.T

def best_geometric_transform(inp: np.ndarray, out: np.ndarray) -> tuple[str, np.ndarray, float]:
    """Try all transforms and return (name, transformed, match_score)."""
    cands = {
        "none": inp,
        "rot90": rotate90(inp),
        "rot180": rotate90(inp, 2),
        "rot270": rotate90(inp, 3),
        "flip_x": flip_x(inp),
        "flip_y": flip_y(inp),
        "transpose": transpose(inp),
    }
    best_name, best_score, best_img = "none", 0.0, inp
    for name, img in cands.items():
        s = (img.shape == out.shape)
        score = 0.0
        if s:
            score = np.mean(img == out)
        if score > best_score:
            best_name, best_score, best_img = name, score, img
    return best_name, best_img, round(float(best_score), 3)
