#!/usr/bin/env python3
# step10_meta_mutate.py — heuristic meta-mutation for cached rules
import json, random
import numpy as np
from pathlib import Path

WORK = Path("/data/data/com.termux/files/home/arc_solver")
CACHE_PATH = WORK / "rule_cache.json"

def _load_cache():
    if CACHE_PATH.exists():
        try:
            with open(CACHE_PATH) as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def _save_cache(data):
    with open(CACHE_PATH, "w") as f:
        json.dump(data, f, indent=2)

def mutate_color_map(cmap: dict) -> dict:
    """Randomly perturb color map values within range [0,9]."""
    mutated = dict(cmap)
    if not mutated:
        return mutated
    for k in list(mutated.keys()):
        if random.random() < 0.3:  # mutate 30% of entries
            mutated[k] = int((mutated[k] + random.choice([-1, 1])) % 10)
    return mutated

def mutate_rule(rule: dict) -> dict:
    """Create a mutated copy of a rule."""
    new_rule = dict(rule)
    if "color_map" in new_rule:
        new_rule["color_map"] = mutate_color_map(new_rule["color_map"])
    if "confidence" in new_rule:
        new_rule["confidence"] = round(float(new_rule["confidence"]) * random.uniform(0.9, 1.1), 3)
    return new_rule

def meta_mutate():
    """Apply mutation to cached rules for exploration."""
    cache = _load_cache()
    if not cache:
        print("[MUTATE] No cached rules to mutate.")
        return
    mutated_count = 0
    for key, rec in list(cache.items()):
        rule = rec.get("rule")
        if not rule:
            continue
        if random.random() < 0.5:  # mutate half
            cache[key]["rule"] = mutate_rule(rule)
            mutated_count += 1
    _save_cache(cache)
    print(f"[MUTATE] Mutated {mutated_count}/{len(cache)} cached rules.")
