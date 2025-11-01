#!/usr/bin/env python3
# ============================================================
# step19_meta_promoter.py — Adaptive Meta Promotion System
# Dynamically adjusts threshold based on average replay confidence
# ============================================================

import json
import statistics
from pathlib import Path

WORK = Path("/data/data/com.termux/files/home/arc_solver")
REPLAY_PATH = WORK / "meta_replay.json"
META_PATH = WORK / "meta_cache.json"

def _load_json(path: Path):
    if path.exists():
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def _save_json(path: Path, data: dict):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def promote_replay_to_meta(base_threshold: float = 0.9):
    """Promote replayed rules with adaptive confidence threshold."""
    replay = _load_json(REPLAY_PATH)
    meta = _load_json(META_PATH)
    promoted = 0

    if not isinstance(replay, list) or not replay:
        print("[PROMOTE] No replay data.")
        return

    # Compute adaptive threshold
    confs = [float(r.get("confidence", 0)) for r in replay if "confidence" in r]
    mean_conf = statistics.mean(confs)
    std_conf = statistics.pstdev(confs) if len(confs) > 1 else 0
    dynamic_thresh = max(base_threshold * 0.8, mean_conf + 0.25 * std_conf)
    dynamic_thresh = round(dynamic_thresh, 3)

    for entry in replay:
        conf = float(entry.get("confidence", 0))
        if conf >= dynamic_thresh:
            rid = f"meta_promote_{len(meta)+1}"
            meta[rid] = {
                "type": f"{entry.get('rule_type', 'unknown')}_meta",
                "color_map": entry.get("color_map", {}),
                "confidence": conf,
                "source": "adaptive_replay",
            }
            promoted += 1

    if promoted:
        _save_json(META_PATH, meta)
        print(f"[PROMOTE] Promoted {promoted} replay rules → meta_cache.json (threshold={dynamic_thresh})")
    else:
        print(f"[PROMOTE] No rules passed threshold ({dynamic_thresh}).")
