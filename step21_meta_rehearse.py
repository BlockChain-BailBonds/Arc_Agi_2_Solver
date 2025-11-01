#!/usr/bin/env python3
import json, math, hashlib
from pathlib import Path
from typing import Dict, Any, List, Tuple

WORK = Path("/data/data/com.termux/files/home/arc_solver")
META_PATH = WORK / "meta_cache.json"
CACHE_PATH = WORK / "cache.json"

# ---------------- IO ----------------

def _load_json(path: Path):
    try:
        if path.exists():
            with open(path) as f:
                return json.load(f)
    except Exception:
        pass
    # default container types
    return {} if path.suffix == ".json" else {}

def _save_json(path: Path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

# ---------------- signatures & distances ----------------

def _norm_cmap(cmap: Dict[Any, Any]) -> Dict[int, int]:
    """Coerce to {int:int}, drop invalid keys."""
    out = {}
    for k, v in (cmap or {}).items():
        try:
            out[int(k)] = int(v)
        except Exception:
            continue
    return out

def _pairs(cmap: Dict[int, int]) -> List[Tuple[int, int]]:
    """Sorted (k->v) pairs for stable hashing/compare."""
    return sorted((int(k), int(v)) for k, v in cmap.items())

def _sig_str(cmap: Dict[int,int]) -> str:
    """Stable signature string for a color map."""
    p = _pairs(cmap)
    raw = ";".join(f"{k}->{v}" for k,v in p)
    # include size to avoid collisions of different sizes with same prefix
    raw = f"n={len(p)}|" + raw
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()

def _sig_distance(a: Dict[int,int], b: Dict[int,int]) -> float:
    """
    Distance in [0,1] based on Jaccard of mapping pairs.
      dist = 1 - |A∩B| / |A∪B|
    """
    A = set(_pairs(a))
    B = set(_pairs(b))
    if not A and not B: 
        return 0.0
    jacc = len(A& B) / float(len(A | B))
    return 1.0 - jacc

# ---------------- capacity heuristic ----------------

def _auto_cap(n: int) -> int:
    """
    Sublinear growth: ≤8: n, else ~4*sqrt(n) clamped to [8,64]
    """
    if n <= 8:
        return n
    return max(8, min(64, int(math.ceil(n ** 0.5) * 4)))

# ---------------- main rehearse ----------------

def rehearse_meta(cap: int | str = "auto", diversity: float = 0.33, min_sig_dist: float = 0.35) -> int:
    """
    Load top meta rules into solver cache with capacity & signature diversity.

    Args:
      cap: "auto" or integer count of rules to load.
      diversity: fraction of K we try to make signature-distinct (ceil(diversity*K)).
      min_sig_dist: minimum pairwise signature distance between selected rules.
    Returns:
      Number of rules injected into cache.
    """
    meta = _load_json(META_PATH)
    cache = _load_json(CACHE_PATH)

    # Collect candidates
    items: List[Dict[str, Any]] = []
    if isinstance(meta, dict):
        for rid, rule in meta.items():
            if not isinstance(rule, dict):
                continue
            rtype = str(rule.get("type", ""))
            if not rtype.endswith("_meta"):
                continue
            cmap = _norm_cmap(rule.get("color_map", {}))
            conf = float(rule.get("confidence", 0.0))
            items.append({
                "rid": rid,
                "type": rtype,               # often "color_map_meta"
                "color_map": cmap,
                "confidence": conf,
                "sig": _sig_str(cmap),
            })

    if not items:
        print("[REHEARSE] Meta cache contained no *meta* rules.")
        return 0

    # Sort by confidence desc, dedupe by signature first
    items.sort(key=lambda x: x["confidence"], reverse=True)
    seen_sig = set()
    uniq: List[Dict[str,Any]] = []
    for it in items:
        if it["sig"] in seen_sig:
            continue
        seen_sig.add(it["sig"])
        uniq.append(it)
    items = uniq

    N = len(items)
    if N == 0:
        print("[REHEARSE] All meta rules were duplicates by signature.")
        return 0

    # Determine K
    if cap == "auto":
        K = _auto_cap(N)
    elif isinstance(cap, int) and cap > 0:
        K = cap
    else:
        K = N
    K = max(1, min(N, K))

    # Target number of signature-distinct picks
    target_distinct = max(1, min(K, int(math.ceil(diversity * K))))

    # Greedy selection with signature distance constraint
    selected: List[Dict[str,Any]] = []
    sigs: List[str] = []

    # Adaptive relaxation if we can't fill K
    relax = min_sig_dist
    idx = 0
    while len(selected) < K and idx < N:
        cand = items[idx]
        ok = True
        for s in selected:
            d = _sig_distance(cand["color_map"], s["color_map"])
            if d < relax:
                ok = False
                break
        if ok:
            selected.append(cand)
            sigs.append(cand["sig"])
        idx += 1

        # If we ran through list and still underfilled: relax and restart pass
        if idx == N and len(selected) < K:
            # Relax only if we still haven't reached target distinct
            if len({x["sig"] for x in selected}) < target_distinct and relax > 0.05:
                relax = max(0.05, round(relax * 0.85, 3))  # relax by 15%
                idx = 0
                # keep current selected but allow closer future picks
            else:
                break  # accept fewer than K if constraints tight

    # If still short, top up ignoring distance (but keep unique signatures first)
    if len(selected) < K:
        needed = K - len(selected)
        pool = [it for it in items if it["sig"] not in {x["sig"] for x in selected}]
        selected.extend(pool[:needed])

    # Purge old rehearse_* entries
    if not isinstance(cache, dict):
        cache = {}
    else:
        for k in list(cache.keys()):
            if isinstance(k, str) and k.startswith("rehearse_"):
                del cache[k]

    # Inject selected rules
    for i, r in enumerate(selected, 1):
        tid = f"rehearse_{r['rid']}_{i}"
        cache[tid] = {
            "type": r["type"],                 # e.g., "color_map_meta"
            "color_map": r["color_map"],
            "confidence": r["confidence"],
            "sig": r["sig"],
        }

    _save_json(CACHE_PATH, cache)

    # Reporting
    # logical "types" (likely 1) + signature diversity (what we care about)
    type_count = len({r["type"] for r in selected})
    sig_count = len({r["sig"] for r in selected})
    top3 = ", ".join(f"{r['rid']}:{r['confidence']:.3f}" for r in selected[:3])
    print(f"[REHEARSE] Loaded {len(selected)}/{N} meta rules → cache "
          f"(types={type_count}, sigs={sig_count}, min_sig_dist={min_sig_dist}, relax_final={relax}, top3: {top3})")
    return len(selected)

if __name__ == "__main__":
    rehearse_meta()
