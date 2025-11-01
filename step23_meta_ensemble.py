#!/usr/bin/env python3
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any, Callable

from arc_solver.step5_transforms import rotate90, flip_x, flip_y

WORK = Path("/data/data/com.termux/files/home/arc_solver")
CACHE_PATH  = WORK / "cache.json"
META_PATH   = WORK / "meta_cache.json"
REPLAY_PATH = WORK / "replay.json"

# ---------------- IO ----------------
def _load_json(path: Path):
    try:
        if path.exists():
            with open(path) as f:
                return json.load(f)
    except Exception:
        pass
    if path == CACHE_PATH:
        return {}
    if path == META_PATH:
        return {}
    if path == REPLAY_PATH:
        return []
    return {}

# ---------------- color-map utils ----------------
def _norm_cmap(cmap: Dict[Any, Any]) -> Dict[int, int]:
    out = {}
    for k, v in (cmap or {}).items():
        try:
            out[int(k)] = int(v)
        except Exception:
            continue
    return out

def _cmap_sig(cmap: Dict[int,int]) -> str:
    items = sorted((int(k), int(v)) for k, v in (cmap or {}).items())
    return ";".join(f"{k}->{v}" for k, v in items)

def _apply_cmap(grid: np.ndarray, cmap: Dict[int,int]) -> np.ndarray:
    lut = np.arange(10, dtype=np.int16)
    for k, v in cmap.items():
        if 0 <= k <= 9 and 0 <= v <= 9:
            lut[k] = v
    return lut[grid]

# ---------------- transforms as (forward, inverse) ----------------
def _transforms() -> List[Tuple[str, Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray]]]:
    # Our rotate90(g, k) is clockwise by k*90 (wrapper uses np.rot90 with negative k).
    return [
        ("id",      lambda g: g,                 lambda g: g),
        ("rot90",   lambda g: rotate90(g, 1),   lambda g: rotate90(g, 3)),
        ("rot180",  lambda g: rotate90(g, 2),   lambda g: rotate90(g, 2)),
        ("rot270",  lambda g: rotate90(g, 3),   lambda g: rotate90(g, 1)),
        ("flip_x",  lambda g: flip_x(g),        lambda g: flip_x(g)),  # self-inverse
        ("flip_y",  lambda g: flip_y(g),        lambda g: flip_y(g)),  # self-inverse
    ]

# ---------------- candidate gathering ----------------
def collect_candidate_maps(task_id: str) -> List[Dict[str, Any]]:
    cands: List[Dict[str, Any]] = []
    cache  = _load_json(CACHE_PATH)
    meta   = _load_json(META_PATH)
    replay = _load_json(REPLAY_PATH)

    # task-specific cached rule
    if isinstance(cache, dict) and task_id in cache:
        rule = cache[task_id]
        if isinstance(rule, dict) and "color_map" in rule:
            cands.append({
                "type": rule.get("type", "cache"),
                "color_map": _norm_cmap(rule.get("color_map", {})),
                "confidence": float(rule.get("confidence", 0.6)),
                "source": f"cache:{task_id[:8]}",
            })

    # rehearse_* injected meta rules
    if isinstance(cache, dict):
        for k, rule in cache.items():
            if isinstance(k, str) and k.startswith("rehearse_") and isinstance(rule, dict):
                cm = _norm_cmap(rule.get("color_map", {}))
                if cm:
                    cands.append({
                        "type": rule.get("type", "meta"),
                        "color_map": cm,
                        "confidence": float(rule.get("confidence", 0.7)),
                        "source": f"cache:{k}",
                    })

    # meta rules
    if isinstance(meta, dict):
        for rid, rule in meta.items():
            if isinstance(rule, dict) and str(rule.get("type","")).endswith("_meta"):
                cm = _norm_cmap(rule.get("color_map", {}))
                if cm:
                    cands.append({
                        "type": rule.get("type", "meta"),
                        "color_map": cm,
                        "confidence": float(rule.get("confidence", 0.7)),
                        "source": f"meta:{rid}",
                    })

    # replay memory
    if isinstance(replay, list):
        for i, entry in enumerate(replay):
            cm = _norm_cmap(entry.get("color_map", {}))
            if cm:
                cands.append({
                    "type": entry.get("rule_type", "replay"),
                    "color_map": cm,
                    "confidence": float(entry.get("confidence", 0.6)),
                    "source": f"replay:{i}",
                })

    # identity fallback
    ident = {i: i for i in range(10)}
    cands.append({"type":"identity","color_map":ident,"confidence":0.5,"source":"fallback:identity"})

    # dedupe by signature, keep highest conf
    best_by_sig: Dict[str, Dict[str, Any]] = {}
    for c in cands:
        sig = _cmap_sig(c["color_map"])
        if sig not in best_by_sig or c["confidence"] > best_by_sig[sig]["confidence"]:
            best_by_sig[sig] = c
    return list(best_by_sig.values())

# ---------------- scoring (supervised on train pairs) ----------------
def _score_variant_on_pairs(pairs: List[Dict[str, Any]],
                            cmap: Dict[int,int],
                            tname: str,
                            fwd, inv) -> float:
    if not pairs:
        return 0.5
    scores = []
    for p in pairs:
        inp = np.array(p["input"], dtype=np.int16)
        out = np.array(p["output"], dtype=np.int16)
        # Align with ground truth: inv(fwd(inp) → cmap → pred_t) should match out
        x_t   = fwd(inp)
        y_t   = _apply_cmap(x_t, cmap)
        pred  = inv(y_t)
        if pred.shape != out.shape:
            # guardrail: mismatched shapes get a low score rather than crash
            scores.append(0.0)
        else:
            acc = float(np.mean(pred == out))
            scores.append(acc)
    return float(np.mean(scores)) if scores else 0.5

# ---------------- public API ----------------
def ensemble_predict(task: Dict[str, Any], topk: int = 2) -> Tuple[List[List[List[int]]], float]:
    task_id = task.get("id", "unknown")
    train_pairs = task.get("train", [])
    tests = task.get("test", [])

    cands = collect_candidate_maps(task_id)
    if not cands:
        return [], 0.0

    # Build (cmap × transform) variants and score on training pairs
    variants: List[Tuple[float, Dict[str, Any], str, Callable, Callable]] = []
    for c in cands:
        cm = c["color_map"]
        for tname, fwd, inv in _transforms():
            s = _score_variant_on_pairs(train_pairs, cm, tname, fwd, inv)
            variants.append((s, c, tname, fwd, inv))

    # rank by supervised score
    variants.sort(key=lambda x: x[0], reverse=True)

    # take the best K variants
    top = variants[:max(1, topk)]
    mean_conf = float(np.mean([s for s, *_ in top])) if top else 0.0
    mean_conf = round(mean_conf, 3)

    # predict tests using the same fwd→cmap→inv pipeline
    preds_all: List[List[List[int]]] = []
    for sample in tests:
        inp = np.array(sample["input"], dtype=np.int16)
        outs: List[List[int]] = []
        for s, c, tname, fwd, inv in top:
            x_t   = fwd(inp)
            y_t   = _apply_cmap(x_t, c["color_map"])
            pred  = inv(y_t)
            outs.append(pred.tolist())
        while len(outs) < 2:
            outs.append(outs[0])
        preds_all.append(outs)

    return preds_all, mean_conf

if __name__ == "__main__":
    pass
