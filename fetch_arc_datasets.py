#!/usr/bin/env python3
# fetch_arc_datasets.py — unified dataset loader and merger

import json
from pathlib import Path

WORK = Path("/data/data/com.termux/files/home/arc_solver")
TRAIN_PATH = WORK / "arc-agi_training_challenges.json"
TEST_PATH  = WORK / "arc-agi_test_challenges.json"
MERGED_PATH = WORK / "merged_dataset.json"

def _load_json(path: Path):
    """Load a JSON file; tolerate single dicts and empty content."""
    if not path.exists():
        return []
    try:
        text = path.read_text().strip()
        if not text:
            return []
        data = json.loads(text)
        if isinstance(data, dict):
            return [data]
        if isinstance(data, list):
            return data
    except Exception as e:
        print(f"[WARN] Could not load {path.name}: {e}")
    return []

def fetch_arc_datasets():
    """Load training and test datasets, then merge for unified use."""
    train = _load_json(TRAIN_PATH)
    test = _load_json(TEST_PATH)

    # Filter malformed or empty tasks
    train = [t for t in train if isinstance(t, dict) and "train" in t]
    test  = [t for t in test if isinstance(t, dict) and "test" in t]

    print(f"[INFO] Loaded {len(train)} training tasks, {len(test)} testing tasks.")

    # Merge both for continuity learning
    merged = train + test
    if merged:
        with open(MERGED_PATH, "w") as f:
            json.dump(merged, f, indent=2)
        print(f"[DATA] Merged {len(merged)} total tasks → {MERGED_PATH}")
    else:
        print("[DATA] No tasks to merge.")

    return train, test

if __name__ == "__main__":
    tr, te = fetch_arc_datasets()
    print("Training tasks:", len(tr))
    print("Testing tasks:", len(te))
