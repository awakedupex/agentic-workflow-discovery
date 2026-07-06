"""Operating point controller — selects and applies the optimal decision
threshold for a trained model.

This is the production interface that wraps ThresholdOptimizer together
with a Predictor instance.  Usage:

    controller = OperatingPointController(predictor, cost_fn=10.0)
    controller.fit(val_events, val_labels)
    decision = controller.decide(context_events)  # bool
"""

from __future__ import annotations

import numpy as np

from agentic_workflow_discovery.model.predictor import Predictor
from agentic_workflow_discovery.optimization.threshold import ThresholdOptimizer


class OperatingPointController:
    """Selects and applies an optimal decision threshold.

    Parameters
    ----------
    predictor : Predictor
        Trained model wrapper.
    cost_fn : float
        False negative cost (default 10.0).
    cost_fp : float
        False positive cost (default 1.0).
    n_thresholds : int
        Grid density for threshold sweep (default 1000).
    """

    def __init__(
        self,
        predictor: Predictor,
        cost_fn: float = 10.0,
        cost_fp: float = 1.0,
        n_thresholds: int = 1000,
    ):
        self.predictor = predictor
        self.optimizer = ThresholdOptimizer(
            cost_fn=cost_fn,
            cost_fp=cost_fp,
            n_thresholds=n_thresholds,
        )
        self.best_threshold: float | None = None
        self.best_metrics: dict | None = None
        self.roc_data: dict | None = None

    def fit(
        self,
        confidences: np.ndarray,
        y_true: np.ndarray,
    ) -> float:
        """Find the optimal threshold on validation data.

        Parameters
        ----------
        confidences : np.ndarray
            Softmax scores from the predictor on validation events.
        y_true : np.ndarray
            Ground-truth binary labels.

        Returns
        -------
        best_threshold : float
        """
        threshold, metrics, roc = self.optimizer.optimize(confidences, y_true)
        self.best_threshold = threshold
        self.best_metrics = metrics
        self.roc_data = roc
        return threshold

    def decide(
        self,
        confidence: float,
    ) -> bool:
        """Apply the operating threshold to a single confidence score.

        Returns True (positive — task detected) if confidence >= threshold,
        else False.
        """
        if self.best_threshold is None:
            raise RuntimeError("OperatingPointController must be fit before calling decide()")
        return confidence >= self.best_threshold
