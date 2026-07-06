"""Threshold optimisation under asymmetric costs.

Given model confidence scores (softmax probabilities for the positive
class) and ground-truth labels, this module:
  1. Sweeps a dense grid of decision thresholds.
  2. At each threshold, computes the confusion matrix and the expected
     cost under the given FN/FP costs.
  3. Returns the threshold that minimises expected cost, along with the
     full ROC curve data for visualisation.

Expected cost at threshold θ:
    J(θ) = π⁺ · C_fn · FNR(θ) + π⁻ · C_fp · FPR(θ)

where π⁺, π⁻ are the empirical class priors (prevalences).
"""

from __future__ import annotations

import numpy as np

from agentic_workflow_discovery.optimization.cost import AsymmetricCost


class ThresholdOptimizer:
    """Find the decision threshold that minimises expected asymmetric cost.

    Parameters
    ----------
    cost_fn : float
        False negative cost (default 10.0).
    cost_fp : float
        False positive cost (default 1.0).
    n_thresholds : int
        Number of candidate thresholds to evaluate (default 1000).
    """

    def __init__(
        self,
        cost_fn: float = 10.0,
        cost_fp: float = 1.0,
        n_thresholds: int = 1000,
    ):
        self.cost = AsymmetricCost(cost_fn=cost_fn, cost_fp=cost_fp)
        self.n_thresholds = n_thresholds

    def optimize(
        self,
        confidences: np.ndarray,
        y_true: np.ndarray,
    ) -> tuple[float, dict[str, float], dict[str, np.ndarray]]:
        """Find the optimal threshold that minimises expected cost.

        Parameters
        ----------
        confidences : np.ndarray, shape (n_samples,)
            Model softmax scores for the positive class (range [0, 1]).
        y_true : np.ndarray, shape (n_samples,)
            Ground-truth binary labels.

        Returns
        -------
        best_threshold : float
            Threshold that minimises expected cost.
        best_metrics : dict
            Cost and confusion stats at the best threshold.
        roc : dict
            Full ROC curve: thresholds, fpr, tpr, expected_cost arrays
            for plotting.
        """
        thresholds = np.linspace(0.01, 0.99, self.n_thresholds)
        fpr_arr = np.empty(self.n_thresholds)
        tpr_arr = np.empty(self.n_thresholds)
        cost_arr = np.empty(self.n_thresholds)

        best_cost = float("inf")
        best_threshold = 0.5
        best_metrics = {}

        for i, theta in enumerate(thresholds):
            y_pred = (confidences >= theta).astype(int)
            metrics = self.cost.evaluate(y_true, y_pred)

            fpr_arr[i] = metrics["fp"] / max(metrics["n_neg"], 1)
            tpr_arr[i] = metrics["tp"] / max(metrics["n_pos"], 1)
            cost_arr[i] = metrics["total_cost"]

            if metrics["total_cost"] < best_cost:
                best_cost = metrics["total_cost"]
                best_threshold = float(theta)
                best_metrics = metrics

        roc = {
            "thresholds": thresholds,
            "fpr": fpr_arr,
            "tpr": tpr_arr,
            "expected_cost": cost_arr,
        }

        return best_threshold, best_metrics, roc

    def apply_threshold(
        self,
        confidences: np.ndarray,
        threshold: float,
    ) -> np.ndarray:
        """Apply a decision threshold to confidence scores.

        Returns binary predictions (0 or 1).
        """
        return (confidences >= threshold).astype(int)
