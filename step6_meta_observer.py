#!/usr/bin/env python3
# step6_meta_observer.py — meta-weights controller with adaptive reinforcement

import json
from pathlib import Path
from datetime import datetime

META_PATH = Path(__file__).parent / "meta_weights.json"
FEEDBACK_LOG = Path(__file__).parent / "meta_feedback.jsonl"

def _load_weights():
    if META_PATH.exists():
        try:
            with open(META_PATH) as f:
                return json.load(f)
        except Exception:
            pass
    return {"color_map": 1.0, "geom": 1.0, "none": 1.0, "unknown": 1.0}

def _save_weights(weights: dict):
    with open(META_PATH, "w") as f:
        json.dump(weights, f, indent=2)

def log_feedback(rule_type: str, confidence: float):
    """Record performance feedback."""
    event = {
        "time": datetime.utcnow().isoformat(),
        "rule_type": rule_type,
        "confidence": float(confidence),
    }
    with open(FEEDBACK_LOG, "a") as f:
        f.write(json.dumps(event) + "\n")

def _aggregate_feedback():
    """Compute average confidence per rule type from feedback log."""
    scores = {}
    counts = {}
    if not FEEDBACK_LOG.exists():
        return {}
    with open(FEEDBACK_LOG) as f:
        for line in f:
            try:
                ev = json.loads(line)
                t = ev["rule_type"]
                c = ev["confidence"]
                scores[t] = scores.get(t, 0.0) + c
                counts[t] = counts.get(t, 0) + 1
            except Exception:
                continue
    return {t: round(scores[t] / counts[t], 3) for t in scores}

def update_meta_weights():
    """Update weights using feedback-based reinforcement."""
    weights = _load_weights()
    feedback = _aggregate_feedback()

    for t, mean_conf in feedback.items():
        if mean_conf > 0.85:
            weights[t] = round(min(weights.get(t, 1.0) * 1.1, 2.0), 3)
        elif mean_conf < 0.6:
            weights[t] = round(max(weights.get(t, 1.0) * 0.9, 0.5), 3)

    _save_weights(weights)
    print(f"[META] Updated rule weights: {weights}")
    return weights

def get_rule_weights():
    return _load_weights()
