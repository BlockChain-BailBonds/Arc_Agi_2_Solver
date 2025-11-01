#!/usr/bin/env python3
"""
step8_memory_cache.py — persistent cache of rules including color_map for meta learning
"""

import json
from pathlib import Path
from hashlib import sha1

WORK = Path("/data/data/com.termux/files/home/arc_solver")
CACHE_PATH = WORK / "cache.json"

def _hash_task(task: dict) -> str:
    """Unique hash per task based on training data structure."""
    try:
        s = json.dumps(task["train"], sort_keys=True)
    except Exception:
        s = str(task)
    return sha1(s.encode()).hexdigest()[:8]

def _load_cache() -> dict:
    if CACHE_PATH.exists():
        try:
            return json.loads(CACHE_PATH.read_text())
        except Exception:
            return {}
    return {}

def _save_cache(cache: dict):
    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=2)

def get_cached_rule(task: dict):
    key = _hash_task(task)
    cache = _load_cache()
    if key in cache:
        rec = cache[key]
        print(f"[CACHE] Reusing rule from {key} conf={rec.get('confidence', 0.0)}")
        return rec["rule"]
    return None

def update_cache(task: dict, rule: dict, conf: float):
    """Store rule and its color_map for future meta-generalization."""
    key = _hash_task(task)
    cache = _load_cache()
    cache[key] = {
        "rule": {
            "type": rule.get("type", "unknown"),
            "color_map": {int(k): int(v) for k, v in rule.get("color_map", {}).items()},
            "confidence": round(float(conf), 3)
        }
    }
    _save_cache(cache)
    print(f"[CACHE] Stored rule for {key} conf={conf:.2f}")
