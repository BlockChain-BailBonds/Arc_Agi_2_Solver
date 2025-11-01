#!/usr/bin/env python3
"""
step13_cross_generalizer.py — derive meta-rules across all cached and learned rules.

Purpose:
- Scan cached rules (from step8_memory_cache)
- Scan memory ledger (from step7_autolearn)
- Cluster similar color maps
- Average confidence and build cross-task meta-rules
- Save into meta_cache.json for future preloading
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict

WORK = Path("/data/data/com.termux/files/home/arc_solver")
CACHE_PATH = WORK / "cache.json"
MEM_PATH = WORK / "autolearn_memory.json"
META_PATH = WORK / "meta_cache.json"


def _load_json(path: Path) -> Any:
    if not path.exists():
        return {}
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_json(path: Path, data: Any):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _color_map_distance(map1: Dict[str, int], map2: Dict[str, int]) -> float:
    """Simple symmetric distance metric between two color maps."""
    keys = set(map1.keys()) | set(map2.keys())
    if not keys:
        return 0.0
    diff = 0
    for k in keys:
        v1 = map1.get(k, -1)
        v2 = map2.get(k, -1)
        diff += (v1 != v2)
    return diff / len(keys)


def _merge_maps(maps: List[Dict[str, int]]) -> Dict[str, int]:
    """Average vote per input color across multiple maps."""
    tally = defaultdict(lambda: defaultdict(int))
    for cmap in maps:
        for k, v in cmap.items():
            tally[k][v] += 1
    merged = {}
    for k, counts in tally.items():
        best_v = max(counts.items(), key=lambda x: x[1])[0]
        merged[k] = best_v
    return {int(k): int(v) for k, v in merged.items()}


def build_meta_rules() -> Dict[str, Any]:
    """Aggregate all cached and memory rules into generalized meta-rules."""
    cache = _load_json(CACHE_PATH)
    memory = _load_json(MEM_PATH)
    all_rules = []

    # --- collect rules from cache ---
    for tid, rec in cache.items():
        rule = rec.get("rule")
        if rule and isinstance(rule, dict):
            cmap = rule.get("color_map", {})
            conf = float(rule.get("confidence", 0.0))
            all_rules.append({"type": rule.get("type", "unknown"), "color_map": cmap, "confidence": conf})

    # --- collect from memory ---
    for rtype, rec in memory.items():
        if isinstance(rec, dict):
            conf = float(rec.get("mean", rec.get("mean_conf", 0.0)))
            if "color_map" in rec:
                all_rules.append({"type": rtype, "color_map": rec["color_map"], "confidence": conf})

    # --- group similar maps ---
    clusters: List[List[Dict[str, Any]]] = []
    for rule in all_rules:
        placed = False
        for cluster in clusters:
            if _color_map_distance(rule["color_map"], cluster[0]["color_map"]) < 0.3:
                cluster.append(rule)
                placed = True
                break
        if not placed:
            clusters.append([rule])

    # --- merge each cluster ---
    meta_rules = {}
    for i, cluster in enumerate(clusters):
        merged_cmap = _merge_maps([r["color_map"] for r in cluster])
        avg_conf = round(float(np.mean([r["confidence"] for r in cluster])), 3)
        meta_rules[f"meta_rule_{i+1}"] = {
            "type": "color_map_meta",
            "color_map": merged_cmap,
            "confidence": avg_conf,
            "sources": [r["type"] for r in cluster],
            "size": len(cluster),
        }

    _save_json(META_PATH, meta_rules)
    print(f"[META-GEN] Built {len(meta_rules)} meta-rules → {META_PATH}")
    return meta_rules


if __name__ == "__main__":
    meta = build_meta_rules(); print("[META-LINK] Imported self-corrected color maps into meta-cache")
    print(json.dumps(meta, indent=2))
