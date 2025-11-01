#!/usr/bin/env python3
# step9_cross_generalize.py — cross-task rule generalization and reuse

import json
import numpy as np
from pathlib import Path
from hashlib import sha1

WORK = Path("/data/data/com.termux/files/home/arc_solver")
BANK_PATH = WORK / "rule_bank.json"

def _hash_grid(grid):
    """Stable SHA1 hash of flattened grid."""
    flat = np.array(grid, dtype=np.int64).flatten()
    return sha1(flat.tobytes()).hexdigest()[:8]

def task_signature(task):
    """Generate a compact signature based on its training pairs."""
    parts = []
    for pair in task.get("train", []):
        inp, out = np.array(pair["input"]), np.array(pair["output"])
        parts.append(_hash_grid(inp))
        parts.append(_hash_grid(out))
    return sha1("".join(parts).encode()).hexdigest()[:12]

def _load_bank():
    if BANK_PATH.exists():
        try:
            with open(BANK_PATH) as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def _save_bank(bank):
    with open(BANK_PATH, "w") as f:
        json.dump(bank, f, indent=2)

def generalize_rule(task, new_rule):
    """Blend with best previous rule based on similarity of signatures."""
    bank = _load_bank()
    sig = task_signature(task)
    if not bank:
        bank[sig] = new_rule
        _save_bank(bank)
        return new_rule

    # compute hash distance (Hamming on first 12 chars)
    best_sig, best_rule, best_score = None, None, 0
    for s, r in bank.items():
        sim = sum(a == b for a, b in zip(sig, s)) / 12
        if sim > best_score:
            best_score, best_sig, best_rule = sim, s, r

    if best_rule and best_score > 0.75:
        # merge color maps
        cmap_new = new_rule.get("color_map", {})
        cmap_old = best_rule.get("color_map", {})
        merged = {**cmap_old, **cmap_new}
        new_rule["color_map"] = merged
        new_rule["type"] = new_rule.get("type", best_rule.get("type", "color_map"))
        print(f"[GENERALIZE] Merged rule from {best_sig} sim={best_score:.2f}")

    bank[sig] = new_rule
    _save_bank(bank)
    return new_rule
