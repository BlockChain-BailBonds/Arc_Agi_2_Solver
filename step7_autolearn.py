#!/usr/bin/env python3
import json, random
import numpy as np
from datetime import datetime
from pathlib import Path

WORK = Path("/data/data/com.termux/files/home/arc_solver")
MEM_PATH = WORK / "memory.json"
LEDGER_PATH = WORK / "ledger.json"
WEIGHTS_PATH = WORK / "meta_weights.json"

def _load_json(path: Path):
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}

def _save_json(path: Path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def update_memory(rule_type: str, confidence: float):
    mem = _load_json(MEM_PATH)
    rec = mem.get(rule_type, {"count": 0, "mean": 0.0})
    rec["count"] += 1
    rec["mean"] = round((rec["mean"] * (rec["count"] - 1) + confidence) / rec["count"], 3)
    mem[rule_type] = rec
    _save_json(MEM_PATH, mem)
    print(f"[AUTOLEARN] Memory updated {rule_type}: mean={rec['mean']:.3f}, n={rec['count']}")

def log_event(rule_type: str, confidence: float):
    entry = {"time": datetime.utcnow().isoformat(), "rule_type": rule_type, "confidence": confidence}
    with open(LEDGER_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")

def summarize_ledger():
    try:
        with open(LEDGER_PATH) as f:
            lines = [json.loads(x) for x in f if x.strip()]
        grouped = {}
        for e in lines:
            r = e["rule_type"]
            grouped.setdefault(r, []).append(e["confidence"])
        return {r: round(float(np.mean(vals)), 3) for r, vals in grouped.items()}
    except Exception:
        return {}

def update_meta_weights():
    weights = _load_json(WEIGHTS_PATH) or {"color_map": 1.0, "none": 1.0, "unknown": 1.0}
    weights["color_map"] = round(weights["color_map"] * random.uniform(0.95, 1.05), 3)
    weights["none"] = round(weights["none"] * random.uniform(0.95, 1.05), 3)
    _save_json(WEIGHTS_PATH, weights)
    print(f"[AUTOLEARN] Meta weights → {weights}")

def get_rule_weights():
    if WEIGHTS_PATH.exists():
        try:
            with open(WEIGHTS_PATH) as f:
                return json.load(f)
        except Exception:
            pass
    return {"color_map": 1.0, "none": 1.0, "unknown": 1.0}

# === NEW: Dynamic confidence scaling ===
def adjust_confidence(current_conf: float, rule_type: str) -> float:
    """Boost confidence if meta learning stabilizes or repeats."""
    mem = _load_json(MEM_PATH)
    rule_info = mem.get(rule_type, {})
    mean = rule_info.get("mean", current_conf)
    count = rule_info.get("count", 1)

    # If repeated corrections, gradually increase
    scale = 1.0 + min(count / 100.0, 0.25)
    boosted = round(min(mean * scale, 1.0), 3)
    print(f"[ADAPT] Dynamic scaling → {rule_type}: base={mean:.3f}, boosted={boosted:.3f}")
    return boosted
