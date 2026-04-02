"""McNemar test for paired classifier comparison.

Uses continuity-corrected chi2 (Dietterich 1998).
"""
import numpy as np
from scipy.stats import chi2

def mcnemar_test(y_true, y_pred_a, y_pred_b):
    """Compare classifiers A (baseline) and B (corrected)."""
    b = int(np.sum((y_pred_a != y_true) & (y_pred_b == y_true)))
    c = int(np.sum((y_pred_a == y_true) & (y_pred_b != y_true)))
    if b + c == 0: return {"b": b, "c": c, "chi2": 0.0, "p": 1.0}
    stat = (abs(b - c) - 1)**2 / (b + c)
    p = 1 - chi2.cdf(stat, df=1)
    return {"b": b, "c": c, "chi2": round(stat, 4), "p": round(p, 6)}
