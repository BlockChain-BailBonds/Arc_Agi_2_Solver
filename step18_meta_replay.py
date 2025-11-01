#!/usr/bin/env python3
# ============================================================
# step18_meta_replay.py — Meta-Ledger Replay Memory
# Stores and reuses past high-confidence rules to stabilize learning.
# ============================================================

import json
from pathlib import Path
import numpy as np

WORK = Path("/data/data/com.termux/files/home/arc_solver")
REPLAY_PATH = WORK / "meta_replay.json"
MAX_MEMORY = 10  # max stored entries

# ============================================================
# Replay Buffer Operations
# ============================================================

def _load_replay() -> list:
    if REPLAY_PATH.exists():
        try:
            with open(REPLAY_PATH) as f:
                return json.load(f)
        except Exception:
            return []
    return []

def _save_replay(mem: list):
    with open(REPLAY_PATH, "w") as f:
        json.dump(mem[-MAX_MEMORY:], f, indent=2)

# ============================================================
# Core Functions
# ============================================================

def record_replay(rule_type: str, color_map: dict, confidence: float):
    """Store a new rule snapshot with confidence."""
    mem = _load_replay()
    entry = {
        "rule_type": rule_type,
        "color_map": {int(k): int(v) for k, v in color_map.items()},
        "confidence": round(float(confidence), 3)
    }
    mem.append(entry)
    _save_replay(mem)
    print(f"[REPLAY] Stored rule_type={rule_type} conf={confidence:.3f}")

def fetch_top_replay(threshold: float = 0.8) -> dict:
    """Retrieve highest-confidence rule above threshold."""
    mem = _load_replay()
    if not mem:
        return {}
    mem_sorted = sorted(mem, key=lambda x: x["confidence"], reverse=True)
    best = mem_sorted[0]
    if best["confidence"] >= threshold:
        print(f"[REPLAY] Using top replay rule conf={best['confidence']}")
        return best
    return {}
