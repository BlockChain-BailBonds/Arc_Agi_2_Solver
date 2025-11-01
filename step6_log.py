#!/usr/bin/env python3
# step6_log.py — persistent observer event logger

import json
from pathlib import Path
from datetime import datetime

LOG_PATH = Path(__file__).parent / "observer_ledger.jsonl"

def log_rule(rule: dict):
    """Append a rule entry to the ledger."""
    event = {
        "time": datetime.utcnow().isoformat(),
        "type": "rule",
        **rule
    }
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(event) + "\n")

def log_prediction(task_id: int, pred, target):
    """Append a prediction comparison entry."""
    event = {
        "time": datetime.utcnow().isoformat(),
        "type": "prediction",
        "task": task_id,
        "pred": pred,
        "target": target
    }
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(event) + "\n")
