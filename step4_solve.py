#!/usr/bin/env python3
import json, numpy as np
from pathlib import Path

from arc_solver.step3_learn import learn_from_pairs
from arc_solver.step5_transforms import rotate90, flip_x, flip_y  # kept available
from arc_solver.step7_autolearn import update_memory
from arc_solver.step12_self_corrector import apply_self_correction
from arc_solver.step18_meta_replay import record_replay
from arc_solver.step23_meta_ensemble import ensemble_predict

WORK = Path("/data/data/com.termux/files/home/arc_solver")
CACHE_PATH = WORK / "cache.json"

def _load_json(path: Path):
    try:
        if path.exists():
            with open(path) as f:
                return json.load(f)
    except Exception:
        pass
    return {}

def _save_json(path: Path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def solve_task(task: dict):
    """Main solver: learn/cache/self-correct, then predict via meta-ensemble."""
    cache = _load_json(CACHE_PATH)
    task_id = task.get("id", "unknown")

    # 1) get/learn base rule
    if task_id in cache:
        rule = cache[task_id]
        print(f"[CACHE] Reusing rule from {task_id[:8]} conf={rule.get('confidence', 0)}")
    else:
        result = learn_from_pairs(task.get("train", []))
        rule = result["best_rule"]
        cache[task_id] = rule
        _save_json(CACHE_PATH, cache)
        print(f"[CACHE] Stored rule for {task_id[:8]} conf={rule.get('confidence', 0)}")
        print(f"[SOLVE] Learned new rule type={rule.get('type','unknown')} (meta_refresh=True)")

    # 2) optional self-correction over training pairs
    base_map = rule.get("color_map", {})
    fixes = apply_self_correction(task, [base_map])
    if fixes:
        print(f"[CORRECT] Applied {len(fixes)} fixes.")
    else:
        print("[CORRECT] No fixes applied.")

    # 3) produce predictions with meta-ensemble (uses cache/meta/replay/rehearse)
    preds_all, mean_conf = ensemble_predict(task, topk=2)

    # 4) autolearn + replay log
    update_memory("meta_ensemble", mean_conf)
    # store the base_map to replay so it can be promoted/diversified later
    record_replay("meta_ensemble", base_map, mean_conf)

    return preds_all, mean_conf
