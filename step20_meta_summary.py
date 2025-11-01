#!/usr/bin/env python3
# ============================================================
# step20_meta_summary.py — Meta Promotion Summary Tracker
# Records threshold, promotion count, and replay size over runs
# ============================================================

import json
from datetime import datetime
from pathlib import Path

WORK = Path("/data/data/com.termux/files/home/arc_solver")
SUMMARY_PATH = WORK / "meta_summary.json"
REPLAY_PATH = WORK / "meta_replay.json"

def _load_json(path: Path):
    if path.exists():
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def _save_json(path: Path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def record_summary(threshold: float, promoted: int):
    replay = _load_json(REPLAY_PATH)
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "threshold": round(threshold, 3),
        "promoted": promoted,
        "replay_size": len(replay) if isinstance(replay, list) else 0,
    }

    if SUMMARY_PATH.exists():
        data = _load_json(SUMMARY_PATH)
        if not isinstance(data, list):
            data = []
        data.append(entry)
    else:
        data = [entry]

    _save_json(SUMMARY_PATH, data)
    print(f"[SUMMARY] Logged promotion summary → {SUMMARY_PATH}")
