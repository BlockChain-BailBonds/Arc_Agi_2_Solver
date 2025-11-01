# arc_solver/step7_confidence.py
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict
def score_confidence(pred: np.ndarray, ref: np.ndarray) -> float:
    """0–1 similarity between prediction and reference."""
    if pred.shape != ref.shape:
        return 0.0
    total = pred.size
    correct = (pred == ref).sum()
    return float(correct) / float(total)
def plot_confidences(confidences: List[float], title: str = "Solver Confidence"):
    plt.clf()
    plt.title(title)
    plt.xlabel("Test index")
    plt.ylabel("Confidence")
    plt.ylim(0, 1.05)
    plt.plot(range(len(confidences)), confidences, marker="o", color="blue")
    plt.pause(0.3)

