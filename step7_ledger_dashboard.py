#!/usr/bin/env python3
# step7_ledger_dashboard.py — combine observer + memory into a unified ledger summary

import json
from pathlib import Path
from datetime import datetime
from statistics import mean
from arc_solver.step5_memory import load_memory, summarize_memory

LEDGER_PATH = Path(__file__).parent / "observer_ledger.jsonl"
SUMMARY_PATH = Path(__file__).parent / "ledger_summary.json"

def _read_ledger():
    """Read observer events if available."""
    if not LEDGER_PATH.exists():
        return []
    lines = []
    try:
        with open(LEDGER_PATH) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    lines.append(json.loads(line))
                except Exception:
                    continue
    except Exception:
        pass
    return lines

def _aggregate_observer(events):
    """Aggregate mean confidence per rule type."""
    buckets = {}
    for e in events:
        rtype = e.get("rule_type", "unknown")
        conf = float(e.get("confidence", 0.0))
        if rtype not in buckets:
            buckets[rtype] = []
        buckets[rtype].append(conf)
    return {k: round(mean(v), 3) for k, v in buckets.items()} if buckets else {}

def build_summary():
    """Generate combined summary of memory and observer state."""
    obs_events = _read_ledger()
    obs_summary = _aggregate_observer(obs_events)
    mem = load_memory()
    mem_summary = {k: v.get("mean_conf", 0.0) for k, v in mem.items()}
    combined_mean = summarize_memory()
    data = {
        "time": datetime.utcnow().isoformat(),
        "observer": obs_summary,
        "memory": mem_summary,
        "overall_mean_conf": combined_mean,
    }
    with open(SUMMARY_PATH, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[LEDGER] Summary written → {SUMMARY_PATH}")
    return data

if __name__ == "__main__":
    build_summary()
