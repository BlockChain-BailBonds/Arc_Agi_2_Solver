#!/usr/bin/env python3
"""
step15_meta_decay.py
Dampens runaway meta weights and rewards genuine improvement.
"""

import json
from pathlib import Path
import numpy as np

WORK = Path("/data/data/com.termux/files/home/arc_solver")
META_PATH = WORK / "meta_weights.json"

def _load_meta():
    if META_PATH.exists():
        try:
            with open(META_PATH) as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def _save_meta(meta: dict):
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)

def decay_meta_weights(last_conf: float, avg_conf: float):
    """Apply decay or recovery based on progress."""
    meta = _load_meta()
    if not meta:
        return

    improved = avg_conf > last_conf
    for k, v in list(meta.items()):
        v = float(v)
        if improved:
            # gentle recovery when performance rises
            v = min(v * 1.02, 1.6)
        else:
            # decay if stagnant or declining
            v = max(v * 0.97, 0.8)
        meta[k] = round(v, 3)

    _save_meta(meta)
    tag = "RECOVER" if improved else "DECAY"
    print(f"[META-{tag}] Adjusted meta weights → {meta}")
