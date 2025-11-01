#!/usr/bin/env python3
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any

WORK = Path("/data/data/com.termux/files/home/arc_solver")
MERGED_PATH = WORK / "merged_dataset.json"

def _is_rect_grid(g: Any) -> bool:
    if not isinstance(g, list) or not g:
        return False
    if not all(isinstance(row, list) and row for row in g):
        return False
    w = len(g[0])
    return all(len(r) == w for r in g)

def _vals_ok(g: List[List[int]]) -> bool:
    try:
        for r in g:
            for v in r:
                if not isinstance(v, int) or v < 0 or v > 9:
                    return False
        return True
    except Exception:
        return False

def _load_merged() -> Dict[str, Any]:
    with open(MERGED_PATH) as f:
        return json.load(f)

def validate(results: Dict[str, Any], merged: Dict[str, Any]) -> List[str]:
    issues: List[str] = []
    tasks = merged if isinstance(merged, list) else merged.get("tasks", merged)

    # tasks could be list or dict keyed by id; normalize into list of items with id+test
    norm = []
    if isinstance(tasks, list):
        norm = tasks
    elif isinstance(tasks, dict):
        for v in tasks.values():
            norm.append(v)

    for t in norm:
        tid = t.get("id", "unknown")
        tests = t.get("test", [])
        # Must exist
        if tid not in results:
            issues.append(f"Missing key for task {tid}")
            continue
        pred_tests = results[tid]
        # Must match #tests
        if not isinstance(pred_tests, list) or len(pred_tests) != len(tests):
            issues.append(f"Task {tid} has {len(pred_tests) if isinstance(pred_tests, list) else 'non-list'} preds "
                          f"but dataset has {len(tests)} tests")
            continue
        # Each test: exactly 2 attempts; each attempt rectangular 0..9
        for i, outs in enumerate(pred_tests):
            if not isinstance(outs, list):
                issues.append(f"{tid}[{i}] predictions not a list")
                continue
            if len(outs) != 2:
                issues.append(f"{tid}[{i}] has {len(outs)} attempts (expected 2)")
                continue
            for a_idx, grid in enumerate(outs):
                if not _is_rect_grid(grid):
                    issues.append(f"{tid}[{i}][{a_idx}] not a rectangular 2D list")
                    continue
                if not _vals_ok(grid):
                    issues.append(f"{tid}[{i}][{a_idx}] contains non-int or out-of-range values")
    return issues

def validate_and_fix(results: Dict[str, Any], merged: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    """Best-effort fixer: ensures 2 attempts per test by duplicating first; drops invalid items."""
    issues = []
    tasks = merged if isinstance(merged, list) else merged.get("tasks", merged)

    # Normalize task list
    norm = []
    if isinstance(tasks, list):
        norm = tasks
    elif isinstance(tasks, dict):
        for v in tasks.values():
            norm.append(v)

    fixed = dict(results)
    for t in norm:
        tid = t.get("id", "unknown")
        tests = t.get("test", [])
        if tid not in fixed:
            # fabricate empty two-attempt grids mirroring input shapes (fallback)
            fallback = []
            for s in tests:
                h = len(s["input"])
                w = len(s["input"][0]) if h else 0
                g = [[0]*w for _ in range(h)]
                fallback.append([g, g])
            fixed[tid] = fallback
            issues.append(f"Injected fallback for missing task {tid}")
            continue

        outs = fixed[tid]
        if not isinstance(outs, list) or len(outs) != len(tests):
            # try to coerce: if single test provided, replicate to match count
            if isinstance(outs, list) and len(outs) == 1 and len(tests) > 1:
                fixed[tid] = outs * len(tests)
                issues.append(f"Replicated single test preds for {tid} to {len(tests)}")
            else:
                # truncate or pad with duplicate of first
                if isinstance(outs, list):
                    if len(outs) > len(tests):
                        fixed[tid] = outs[:len(tests)]
                        issues.append(f"Truncated extra test preds for {tid}")
                    else:
                        while len(outs) < len(tests):
                            outs.append(outs[0] if outs else [[[]],[[]]])
                        issues.append(f"Padded missing test preds for {tid}")
                else:
                    # make fully new
                    fallback = []
                    for s in tests:
                        h = len(s["input"])
                        w = len(s["input"][0]) if h else 0
                        g = [[0]*w for _ in range(h)]
                        fallback.append([g, g])
                    fixed[tid] = fallback
                    issues.append(f"Replaced non-list preds for {tid}")

        # ensure exactly two attempts per test
        new_tests = []
        for outs in fixed[tid]:
            if not isinstance(outs, list) or not outs:
                new_tests.append([[[0]], [[0]]])
                issues.append("Replaced invalid attempt list with 1x1 zeros")
                continue
            outs2 = list(outs)
            if len(outs2) == 1:
                outs2.append(outs2[0])
                issues.append("Duplicated single attempt to two attempts")
            elif len(outs2) > 2:
                outs2 = outs2[:2]
                issues.append("Truncated to two attempts")
            # final sanity: rectangular 0..9; if not, coerce to zeros with same shape as first
            def ok(g):
                return isinstance(g, list) and _is_rect_grid(g) and _vals_ok(g)
            if not ok(outs2[0]):
                outs2[0] = [[0]]
                issues.append("Coerced bad attempt[0] to 1x1 zero")
            if not ok(outs2[1]):
                # try use shape of attempt[0]
                h = len(outs2[0]); w = len(outs2[0][0]) if h else 1
                outs2[1] = [[0]*w for _ in range(h if h else 1)]
                issues.append("Coerced bad attempt[1] to zeros")
            new_tests.append(outs2)
        fixed[tid] = new_tests

    # final report
    problems = validate(fixed, merged)
    return fixed, issues + problems

if __name__ == "__main__":
    merged = _load_merged()
    # For CLI quick-check of the saved submission.json
    sub_path = WORK / "submission.json"
    if sub_path.exists():
        data = json.loads(sub_path.read_text())
        problems = validate(data, merged)
        if problems:
            print("[SUBMIT-CHECK] Problems:")
            for p in problems:
                print(" -", p)
        else:
            print("[SUBMIT-CHECK] OK")
    else:
        print("[SUBMIT-CHECK] No submission.json present")
