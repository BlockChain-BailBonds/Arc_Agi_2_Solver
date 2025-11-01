#!/usr/bin/env python3
# observer.py — learns from previous solver attempts and ranks rule types

import json
from pathlib import Path
from collections import defaultdict

LEDGER_PATH = Path(__file__).parent / "observer_log.jsonl"

def observe_event(event: dict):
    """Append observer event to JSONL ledger."""
    try:
        with open(LEDGER_PATH, "a") as f:
            f.write(json.dumps(event) + "\n")
    except Exception as e:
        print(f"[Observer] Write error: {e}")

def analyze_observer():
    """Aggregate observer data and return mean confidence by rule type."""
    if not LEDGER_PATH.exists():
        return {}
    stats = defaultdict(lambda: {"count": 0, "mean_conf": 0.0})
    try:
        with open(LEDGER_PATH) as f:
            for line in f:
                e = json.loads(line.strip())
                t = e.get("rule_type", "unknown")
                c = float(e.get("confidence", 0.0))
                stats[t]["count"] += 1
                stats[t]["mean_conf"] += c
        for t in stats:
            stats[t]["mean_conf"] /= stats[t]["count"]
    except Exception as e:
        print(f"[Observer] Read error: {e}")
    ranked = sorted(stats.items(), key=lambda x: x[1]["mean_conf"], reverse=True)
    return {r[0]: r[1] for r in ranked}

def get_top_rule_type():
    """Return the top-performing rule type based on historical observer data."""
    ranked = analyze_observer()
    if not ranked:
        return "unknown"
    return next(iter(ranked.keys()))

def get_rule_weights() -> dict:
    """
    Compute normalized weighting factors for each rule type
    based on cumulative mean confidence recorded so far.
    Returned dict example: {'color_map': 1.2, 'shift_map': 0.8}
    """
    ranked = analyze_observer()
    if not ranked:
        return {}
    # normalize mean_conf into weight factors
    max_conf = max(v["mean_conf"] for v in ranked.values() if v["mean_conf"] > 0)
    if max_conf == 0:
        return {t: 1.0 for t in ranked}
    return {t: round(v["mean_conf"] / max_conf, 3) for t, v in ranked.items()}
