#!/usr/bin/env python3
# step5_memory.py — persistent memory with cross-task transfer and summarization

import json
from pathlib import Path
import numpy as np

MEMORY_PATH = Path(__file__).parent / "solver_memory.json"

def load_memory() -> dict:
    """Load or initialize memory."""
    if MEMORY_PATH.exists():
        try:
            with open(MEMORY_PATH) as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return data
        except Exception:
            pass
    return {}

def save_memory(mem: dict):
    """Persist memory safely with NumPy-compatible types."""
    def _convert(obj):
        if isinstance(obj, (np.int64, np.integer)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        return obj

    serializable = json.loads(json.dumps(mem, default=_convert))
    with open(MEMORY_PATH, "w") as f:
        json.dump(serializable, f, indent=2)

def update_memory(rule: dict):
    """Update memory and compute transferable weights."""
    mem = load_memory()
    rtype = rule.get("type", "unknown")
    cmap = rule.get("color_map", {})
    conf = float(rule.get("confidence", 0.0))

    if rtype not in mem:
        mem[rtype] = {"records": [], "mean_conf": 0.0, "color_map": {}}

    recs = mem[rtype]["records"]
    recs.append(conf)
    mem[rtype]["mean_conf"] = round(float(np.mean(recs)), 3)

    # merge color maps from past runs for transfer learning
    stored_cmap = mem[rtype].get("color_map", {})
    for k, v in cmap.items():
        stored_cmap[str(k)] = int(v)
    mem[rtype]["color_map"] = stored_cmap

    save_memory(mem)
    print(f"[MEM] Updated {rtype}: {mem[rtype]['mean_conf']:.3f} (records={len(recs)})")

def get_best_color_map(rule_type: str):
    """Retrieve most recent color map for reuse."""
    mem = load_memory()
    entry = mem.get(rule_type)
    if not entry:
        return None
    cmap = entry.get("color_map", {})
    return {int(k): int(v) for k, v in cmap.items()} if cmap else None

def summarize_memory() -> float:
    """Return mean confidence across all rule types."""
    mem = load_memory()
    if not mem:
        return 0.0
    vals = [v.get("mean_conf", 0.0) for v in mem.values() if isinstance(v, dict)]
    return round(float(np.mean(vals)), 3) if vals else 0.0

def clear_memory():
    """Reset solver memory."""
    if MEMORY_PATH.exists():
        MEMORY_PATH.unlink()
        print("[MEM] Memory cleared.")
