from arc_solver.step24_check_submission import validate_and_fix
#!/usr/bin/env python3
# ============================================================
# Main Pipeline — ARC Solver with Meta, Self-Correction, Amplifier, and Decay
# ============================================================

import os
import json
import numpy as np
from pathlib import Path
from arc_solver.step4_solve import solve_task
from arc_solver.step10_meta_mutate import meta_mutate
from arc_solver.step14_mutation_amplifier import amplify_mutations
from arc_solver.step15_meta_decay import decay_meta_weights
from arc_solver.step7_autolearn import summarize_ledger, update_meta_weights

WORK = Path("/data/data/com.termux/files/home/arc_solver")
SUBMISSION_PATH = WORK / "submission.json"
CONF_THRESH = 0.85
MAX_CYCLES = 5

def load_tasks():
    merged = WORK / "merged_dataset.json"
    with open(merged) as f:
        return json.load(f)

def run_cycle(tasks):
    results = {}
    confs = []
    for task in tasks:
        preds, conf = solve_task(task)
        results[task.get("id", "unknown")] = preds
        confs.append(conf)
    return results, float(np.mean(confs))

def main():
    print("[INIT] Loading dataset...")
    tasks = load_tasks()
    last_conf = 0.0

    for cycle in range(1, MAX_CYCLES + 1):
        print(f"[CYCLE {cycle}] Running solver...")
        results, avg_conf = run_cycle(tasks)
        results, _fix_issues = validate_and_fix(results, tasks)
        print(f"[SUBMIT-CHECK] {len(_fix_issues)} post-fix issues detected" if _fix_issues else "[SUBMIT-CHECK] OK")
        print(f"[CYCLE {cycle}] Mean confidence = {avg_conf:.2f}")

        if avg_conf >= CONF_THRESH:
            print(f"[CYCLE {cycle}] Confidence threshold met ({avg_conf:.2f}). Stopping retrain.")
            meta_mutate()
            amplify_mutations(avg_conf)
            decay_meta_weights(last_conf, avg_conf)
            break
        else:
            print(f"[CYCLE {cycle}] Re-training...")
            meta_mutate()
            amplify_mutations(avg_conf)
            decay_meta_weights(last_conf, avg_conf)

        last_conf = avg_conf

    ledger_summary = summarize_ledger()
    print(f"[LEDGER SUMMARY] {ledger_summary}")
    update_meta_weights()
    from arc_solver.step19_meta_promoter import promote_replay_to_meta
    promote_replay_to_meta(base_threshold=0.9)
    from arc_solver.step20_meta_summary import record_summary
    from arc_solver.step21_meta_rehearse import rehearse_meta
    record_summary(threshold=0.801, promoted=10)
    from arc_solver.step22_meta_diversify import diversify_meta
    diversify_meta(target=24, min_new=8, max_shifts=2)
    rehearse_meta(cap=24, diversity=0.5, min_sig_dist=0.4)

    with open(SUBMISSION_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[DONE] Submission saved → {SUBMISSION_PATH}")

if __name__ == "__main__":
    main()
