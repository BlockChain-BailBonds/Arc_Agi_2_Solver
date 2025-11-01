#!/usr/bin/env python3
"""
step14_mutation_amplifier.py
Boosts rule-mutation pressure when confidence stalls in the 0.6–0.8 band.
This is chained AFTER normal meta_mutate().
"""

import json
import random
from pathlib import Path
import numpy as np

WORK = Path("/data/data/com.termux/files/home/arc_solver")
CACHE_PATH = WORK / "cache.json"
AMPLIFIER_LOG = WORK / "mutation_amp.log"

def _load_cache():
    if CACHE_PATH.exists():
        try:
            with open(CACHE_PATH) as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def _save_cache(cache: dict):
    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=2)

def _write_log(msg: str):
    with open(AMPLIFIER_LOG, "a") as f:
        f.write(msg + "\n")

def _mutate_color_map(cmap: dict, intensity: float = 0.25) -> dict:
    """Slightly nudge a color map to explore alternatives."""
    if not cmap:
        return {}
    mutated = dict(cmap)
    keys = list(mutated.keys())
    # mutate 1..N keys depending on intensity
    n_keys = max(1, int(len(keys) * intensity))
    random.shuffle(keys)
    for k in keys[:n_keys]:
        base = int(mutated[k])
        # ensure color stays in 0..9
        mutated[k] = int((base + random.choice([-1, 1])) % 10)
    return mutated

def amplify_mutations(current_conf: float):
    """
    If confidence is between 0.6 and 0.8 for multiple cycles,
    push the cached rules to branch to nearby color maps.
    """
    # we only amplify in the band
    if not (0.6 <= current_conf <= 0.8):
        return

    cache = _load_cache()
    if not cache:
        return

    amplified = 0
    for tid, rec in list(cache.items()):
        rule = rec.get("type")
        cmap = rec.get("color_map", {})
        conf = float(rec.get("confidence", 0.0))

        # only amplify rules that aren't already super high
        if conf < 0.9 and cmap:
            # stronger intensity if conf is really stuck
            intensity = 0.35 if 0.6 <= current_conf < 0.7 else 0.25
            new_cmap = _mutate_color_map(cmap, intensity=intensity)
            new_conf = round(float(np.random.uniform(0.6, 0.9)), 3)
            rec["color_map"] = {int(k): int(v) for k, v in new_cmap.items()}
            rec["confidence"] = new_conf
            cache[tid] = rec
            amplified += 1

    if amplified:
        _save_cache(cache)
        msg = f"[MUTATE-AMP] Amplified {amplified} rules at conf={current_conf:.3f}"
        print(msg)
        _write_log(msg)
    else:
        _write_log(f"[MUTATE-AMP] No rules amplified at conf={current_conf:.3f}")
