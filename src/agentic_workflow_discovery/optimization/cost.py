"""Asymmetric cost matrix for binary task-detection decisions.

In an automation pipeline, the consequences of a false negative (FN) and
a false positive (FP) are not symmetric:
  - FN (missed task) → the pipeline fails to automate → user disappointment
  - FP (false alarm) → the pipeline triggers a spurious action → minor annoyance

Therefore C_fn >> C_fp.  The default ratio is 10:1.

The total cost for a set of predictions is:
    total_cost = FN × C_fn + FP × C_fp

The optimal decision threshold minimises the *expected* cost under the
empirical class priors.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix


class AsymmetricCost:
    """Asymmetric cost evaluator for binary classification.

    Parameters
    ----------
    cost_fn : float
        Cost of a false negative (default 10.0).
    cost_fp : float
        Cost of a false positive (default 1.0).
    """

    def __init__(self, cost_fn: float = 10.0, cost_fp: float = 1.0):
        self.cost_fn = cost_fn
        self.cost_fp = cost_fp

    @property
    def ratio(self) -> float:
        """The FN:FP cost ratio — a single-number summary of asymmetry."""
        return self.cost_fn / self.cost_fp

    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> dict[str, float]:
        """Compute total cost and confusion counts.

        Parameters
        ----------
        y_true : np.ndarray
            Ground-truth labels (0 = negative, 1 = positive).
        y_pred : np.ndarray
            Predicted labels (0 = negative, 1 = positive).

        Returns
        -------
        dict with keys: total_cost, fn, fp, tn, tp, n_pos, n_neg.
        """
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        total_cost = fn * self.cost_fn + fp * self.cost_fp

        return {
            "total_cost": float(total_cost),
            "fn": int(fn),
            "fp": int(fp),
            "tn": int(tn),
            "tp": int(tp),
            "n_pos": int(tp + fn),
            "n_neg": int(tn + fp),
        }
