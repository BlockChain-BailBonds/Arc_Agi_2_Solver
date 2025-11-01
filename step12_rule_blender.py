#!/usr/bin/env python3
"""
step12_rule_blender.py — blend multiple ARC rules into a single stronger rule.

We support (for now):
- color_map
- color_map_fix

Strategy:
1. Normalize all rule types to a color-map-like structure.
2. Weight each candidate by its confidence.
3. For each input color, pick the output color with the highest total weight.
"""

from __future__ import annotations
from typing import List, Dict, Any

def _rule_to_color_map(rule: Dict[str, Any]) -> Dict[int, int]:
    """Convert different rule types to a canonical color_map."""
    if not rule:
        return {}
    rtype = rule.get("type", "color_map")
    cmap = {}

    if rtype in ("color_map", "color_map_fix"):
        raw = rule.get("color_map", {})
        # normalize possible numpy/int types
        for k, v in raw.items():
            ck = int(k)
            cv = int(v)
            cmap[ck] = cv
    else:
        # fallback – unknown rule type
        pass
    return cmap

def blend_rules(rules: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Blend multiple rules into a single color_map rule.
    rules: list of dicts, each should have 'type', 'color_map', 'confidence'
    """
    if not rules:
        return {"type": "color_map", "color_map": {}, "confidence": 0.0}

    # aggregate votes per input color
    votes: Dict[int, Dict[int, float]] = {}
    total_conf = 0.0

    for r in rules:
        conf = float(r.get("confidence", 0.0))
        total_conf += conf
        cmap = _rule_to_color_map(r)
        for inc, outc in cmap.items():
            inc = int(inc)
            outc = int(outc)
            if inc not in votes:
                votes[inc] = {}
            votes[inc][outc] = votes[inc].get(outc, 0.0) + conf

    # pick the best target color per input color
    blended_cmap: Dict[int, int] = {}
    for inc, out_votes in votes.items():
        # sort by weight desc
        best_out = max(out_votes.items(), key=lambda x: x[1])[0]
        blended_cmap[inc] = int(best_out)

    # confidence of the blend = avg of contributors
    blend_conf = round(total_conf / max(len(rules), 1), 3)

    return {
        "type": "color_map_blend",
        "color_map": blended_cmap,
        "confidence": blend_conf,
        "sources": [r.get("type", "unknown") for r in rules],
    }
