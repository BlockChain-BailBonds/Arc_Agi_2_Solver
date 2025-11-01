#!/usr/bin/env python3
# step11_self_correct.py — self-validation and corrective rule synthesis
import numpy as np
import json
from pathlib import Path
from arc_solver.step0_utils import ensure_integrity
from arc_solver.step8_memory_cache import update_cache
from arc_solver.step7_autolearn import log_event

WORK = Path("/data/data/com.termux/files/home/arc_solver")
CORR_PATH = WORK / "self_corrections.json"

def validate_prediction(pred, target):
    """Return accuracy score for grid comparison."""
    pred = ensure_integrity(np.array(pred))
    target = ensure_integrity(np.array(target))
    if pred.shape != target.shape:
        return 0.0
    return float(np.mean(pred == target))

def generate_correction(pred, target):
    """Infer corrective color mapping from mismatched cells."""
    pred = ensure_integrity(np.array(pred))
    target = ensure_integrity(np.array(target))
    diff_mask = pred != target
    cmap = {}
    for v in np.unique(pred[diff_mask]):
        if v in target:
            cmap[int(v)] = int(np.bincount(target[diff_mask]).argmax())
    conf = round(validate_prediction(pred, target), 3)
    return {"type": "color_map_fix", "color_map": cmap, "confidence": conf}

def apply_self_correction(task, preds):
    """Compare predictions to known outputs; update cache if fix found."""
    corrections = []
    for i, pair in enumerate(task.get("train", [])):
        if "output" not in pair:
            continue
        score = validate_prediction(preds[0], pair["output"])
        if score < 1.0:
            fix = generate_correction(preds[0], pair["output"])
            update_cache(task, fix, fix["confidence"])
            log_event(fix["type"], fix["confidence"])
            corrections.append(fix)
    if corrections:
        with open(CORR_PATH, "w") as f:
            json.dump(corrections, f, indent=2)
        print(f"[CORRECT] Applied {len(corrections)} fixes.")
    return corrections
