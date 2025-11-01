#!/usr/bin/env python3
"""
step12_self_corrector.py — tolerant self-correction engine.
Accepts either dict or list[dict] color_maps, never raises attribute errors.
"""

import numpy as np

def apply_self_correction(task: dict, color_maps) -> list[dict]:
    """Generate corrective color maps safely from training pairs."""
    fixes = []
    try:
        train_pairs = task.get("train", [])
        # Normalize color_maps into list of dicts
        if isinstance(color_maps, dict):
            color_maps = [color_maps]
        elif not isinstance(color_maps, list):
            color_maps = []

        for cmap in color_maps:
            if not isinstance(cmap, dict):
                continue
            for pair in train_pairs:
                inp = np.array(pair["input"], dtype=int)
                out = np.array(pair["output"], dtype=int)
                mismatch = (inp != out)
                if np.any(mismatch):
                    corrected = dict(cmap)
                    for ci in np.unique(inp[mismatch]):
                        co = np.bincount(out[inp == ci]).argmax()
                        corrected[int(ci)] = int(co)
                    fixes.append(corrected)
        if fixes:
            print(f"[CORRECT] Applied {len(fixes)} fixes.")
    except Exception as e:
        print(f"[CORRECT] Error: {e}")
    return fixes
