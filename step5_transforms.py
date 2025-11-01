#!/usr/bin/env python3
import numpy as np

def rotate90(grid: np.ndarray, k: int = 1) -> np.ndarray:
    """Rotate grid 90° clockwise k times."""
    return np.rot90(grid, k=-k)

def flip_x(grid: np.ndarray) -> np.ndarray:
    """Flip grid horizontally."""
    return np.fliplr(grid)

def flip_y(grid: np.ndarray) -> np.ndarray:
    """Flip grid vertically."""
    return np.flipud(grid)
