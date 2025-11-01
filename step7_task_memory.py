#!/usr/bin/env python3
# step7_task_memory.py — persistent per-task learning memory

import json
from pathlib import Path
from statistics import mean

MEMORY_PATH = Path(__file__).parent / "task_memory.json"

def _load():
    if not MEMORY_PATH.exists():
        return {}
    try:
        with open(MEMORY_PATH) as f:
            return json.load(f)
    except Exception:
        return {}

def _save(data):
    with open(MEMORY_PATH, "w") as f:
        json.dump(data, f, indent=2)

def record_task_result(task_id: str, rule_type: str, confidence: float):
    """Store or update task memory."""
    memory = _load()
    rec = memory.get(task_id, {"rule_type": rule_type, "confidences": []})
    rec["confidences"].append(confidence)
    rec["mean_conf"] = round(mean(rec["confidences"]), 3)
    memory[task_id] = rec
    _save(memory)
    print(f"[MEM] Updated {task_id}: {rec['mean_conf']}")
    return rec["mean_conf"]

def summarize_memory():
    """Return overall mean confidence across tasks."""
    mem = _load()
    if not mem:
        return 0.0
    vals = [r["mean_conf"] for r in mem.values() if "mean_conf" in r]
    return round(mean(vals), 3)
