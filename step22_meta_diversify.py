#!/usr/bin/env python3
import json, math, hashlib
from pathlib import Path
from collections import Counter
from typing import Dict, Any, List, Tuple

WORK = Path("/data/data/com.termux/files/home/arc_solver")
META_PATH = WORK / "meta_cache.json"
REPLAY_PATH = WORK / "replay.json"

# ---------- IO ----------
def _load_json(path: Path):
    try:
        if path.exists():
            with open(path) as f:
                return json.load(f)
    except Exception:
        pass
    return {} if path.suffix == ".json" else {}

def _save_json(path: Path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

# ---------- helpers ----------
def _norm_cmap(cmap: Dict[Any, Any]) -> Dict[int,int]:
    out = {}
    for k, v in (cmap or {}).items():
        try:
            out[int(k)] = int(v)
        except Exception:
            continue
    return out

def _pairs(cmap: Dict[int,int]) -> List[Tuple[int,int]]:
    return sorted((int(k), int(v)) for k, v in cmap.items())

def _sig_str(cmap: Dict[int,int]) -> str:
    p = _pairs(cmap)
    raw = "n=%d|" % len(p) + ";".join(f"{k}->{v}" for k, v in p)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()

def _is_bijection(cmap: Dict[int,int]) -> bool:
    vals = list(cmap.values())
    return len(set(vals)) == len(vals)

def _invert(cmap: Dict[int,int]) -> Dict[int,int]:
    if not _is_bijection(cmap):
        return {}
    return {int(v): int(k) for k, v in cmap.items()}

def _shift(cmap: Dict[int,int], s: int) -> Dict[int,int]:
    # shift outputs mod 10 (ARC colors 0..9)
    return {k: (v + s) % 10 for k, v in cmap.items()}

def _prune_least_supported(cmap: Dict[int,int], support: Dict[Tuple[int,int], float]) -> Dict[int,int]:
    if len(cmap) <= 1:
        return {}
    # rank by support (low first), drop one weakest pair
    items = list(cmap.items())
    items.sort(key=lambda kv: support.get((int(kv[0]), int(kv[1])), 0.0))
    drop_k, _ = items[0]
    out = dict(cmap)
    out.pop(drop_k, None)
    return out

# ---------- core ----------
def diversify_meta(target: int = 24, min_new: int = 8, max_shifts: int = 2) -> int:
    """
    Synthesize distinct color_map_meta entries from replay+meta to raise signature diversity.

    Args:
      target: soft target of total meta entries desired after diversification.
      min_new: minimum number of *new* signatures to add per call (if available).
      max_shifts: how many output shifts to try per map.
    Returns:
      Number of new meta entries added.
    """
    meta = _load_json(META_PATH)
    if not isinstance(meta, dict):
        meta = {}

    replay = _load_json(REPLAY_PATH)
    if not isinstance(replay, list):
        replay = []

    # gather base maps (existing meta + replay)
    base_maps: List[Dict[str, Any]] = []

    # from meta
    for rid, rule in (meta or {}).items():
        if isinstance(rule, dict) and str(rule.get("type","")).endswith("_meta"):
            cmap = _norm_cmap(rule.get("color_map", {}))
            if cmap:
                base_maps.append({"rid": rid, "conf": float(rule.get("confidence", 0.7)), "cmap": cmap})

    # from replay (as candidates)
    for i, entry in enumerate(replay):
        cmap = _norm_cmap(entry.get("color_map", {}))
        if cmap:
            base_maps.append({"rid": f"replay_{i}", "conf": float(entry.get("confidence", 0.0)), "cmap": cmap})

    if not base_maps:
        print("[DIVERSIFY] No base maps available.")
        return 0

    # compute support for pairs from replay (confidence-weighted)
    support = Counter()
    for entry in replay:
        conf = float(entry.get("confidence", 0.0))
        weight = 1.0 + max(0.0, conf)  # ≥1
        cmap = _norm_cmap(entry.get("color_map", {}))
        for k, v in cmap.items():
            support[(int(k), int(v))] += weight

    # set of existing signatures to avoid dup
    existing_sigs = set()
    for rule in meta.values():
        if isinstance(rule, dict) and "color_map" in rule:
            existing_sigs.add(_sig_str(_norm_cmap(rule["color_map"])))

    # candidate generation
    generated: List[Dict[str, Any]] = []
    for bm in base_maps:
        base = bm["cmap"]
        sig_base = _sig_str(base)
        # 1) inverse if bijection
        inv = _invert(base)
        if inv:
            sig = _sig_str(inv)
            if sig not in existing_sigs:
                generated.append({"cmap": inv, "conf": bm["conf"] * 0.98, "src": f"{bm['rid']}:invert"})

        # 2) shifts
        for s in range(1, max_shifts + 1):
            sh = _shift(base, s)
            sig = _sig_str(sh)
            if sig not in existing_sigs:
                generated.append({"cmap": sh, "conf": bm["conf"] * (0.97 - 0.01*(s-1)), "src": f"{bm['rid']}:shift{s}"})

        # 3) prune weakest pair (if any support known)
        pr = _prune_least_supported(base, support)
        if pr:
            sig = _sig_str(pr)
            if sig not in existing_sigs:
                generated.append({"cmap": pr, "conf": max(0.6, bm["conf"] * 0.95), "src": f"{bm['rid']}:prune1"})

    if not generated:
        print("[DIVERSIFY] No new variants synthesized.")
        return 0

    # rank by confidence (variants carry slightly decayed conf)
    generated.sort(key=lambda x: x["conf"], reverse=True)

    # add until we hit min_new or target total size, skipping duplicate signatures
    added = 0
    for cand in generated:
        if len(meta) >= max(target, len(meta) + min_new):  # safety
            break
        sig = _sig_str(cand["cmap"])
        if sig in existing_sigs:
            continue
        rid = f"meta_div_{len(meta)+1}"
        meta[rid] = {
            "type": "color_map_meta",
            "color_map": cand["cmap"],
            "confidence": round(float(cand["conf"]), 3),
            "source": f"diversify({cand['src']})",
        }
        existing_sigs.add(sig)
        added += 1
        if added >= min_new:
            # we reached minimum; continue a bit if total < target
            if len(meta) >= target:
                break

    _save_json(META_PATH, meta)
    print(f"[DIVERSIFY] Added {added} meta variants → {META_PATH} (now total={len(meta)})")
    return added

if __name__ == "__main__":
    diversify_meta()
