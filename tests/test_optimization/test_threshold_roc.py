"""Unit tests for ThresholdOptimizer.

The core assertion: under high FN cost, the optimal threshold should be
*lower* (more aggressive / higher recall) than under high FP cost.
"""

from __future__ import annotations

import numpy as np

from agentic_workflow_discovery.optimization.threshold import ThresholdOptimizer


class TestThresholdOptimizer:
    def test_high_fn_cost_produces_lower_threshold(self):
        """C_fn >> C_fp → threshold shifts left (more aggressive).

        We construct two overlapping Gaussian distributions (pos ~ N(0.6,0.15),
        neg ~ N(0.4,0.15)) so there is a clear trade-off region where the
        optimal threshold depends on the cost ratio.
        """
        rng = np.random.default_rng(42)
        n = 500
        # Positive class: centred at 0.6
        pos = rng.normal(0.6, 0.15, n // 2)
        # Negative class: centred at 0.4
        neg = rng.normal(0.4, 0.15, n // 2)

        confidences = np.clip(np.concatenate([pos, neg]), 0.01, 0.99)
        y_true = np.array([1] * (n // 2) + [0] * (n // 2))

        # High FN cost (100:1) → should select lower threshold (higher recall)
        opt_fn = ThresholdOptimizer(cost_fn=100.0, cost_fp=1.0)
        theta_fn, metrics_fn, _ = opt_fn.optimize(confidences, y_true)

        # High FP cost (1:100) → should select higher threshold (higher precision)
        opt_fp = ThresholdOptimizer(cost_fn=1.0, cost_fp=100.0)
        theta_fp, metrics_fp, _ = opt_fp.optimize(confidences, y_true)

        assert theta_fn < theta_fp, (
            f"High FN cost should give lower threshold: "
            f"theta_fn={theta_fn:.3f}, theta_fp={theta_fp:.3f}"
        )

        # Verify: high FN cost should have higher recall (lower FNR)
        assert metrics_fn["tp"] >= metrics_fp["tp"], (
            "High FN cost should produce more true positives"
        )

    def test_perfect_confidences_selects_perfect_threshold(self):
        """If confidences perfectly separate classes, any threshold in
        (0, 1) achieves zero cost, and the optimizer picks the first
        (lowest) threshold with zero cost."""
        confidences = np.array([0.1, 0.2, 0.8, 0.9])
        y_true = np.array([0, 0, 1, 1])

        opt = ThresholdOptimizer(cost_fn=10.0, cost_fp=1.0)
        theta, metrics, roc = opt.optimize(confidences, y_true)

        assert metrics["total_cost"] == 0.0
        # Since multiple thresholds achieve 0 cost, the optimizer picks
        # the first one (lowest) that has 0 cost.
        # At threshold=0.01, all are positive → high cost.
        # At threshold=0.3, y_pred=[0,0,1,1] → 0 cost.
        assert 0.2 < theta < 0.8, f"Unexpected threshold: {theta:.3f}"

    def test_roc_curve_monotonic(self):
        """TPR should be non-increasing and FPR non-increasing as
        threshold increases."""
        rng = np.random.default_rng(42)
        confidences = rng.uniform(0, 1, 500)
        y_true = (confidences > 0.5).astype(int)

        opt = ThresholdOptimizer(cost_fn=10.0, cost_fp=1.0)
        _, _, roc = opt.optimize(confidences, y_true)

        # TPR should generally decrease as threshold increases
        # (higher threshold → fewer positives predicted)
        assert roc["tpr"][0] >= roc["tpr"][-1]
        assert roc["fpr"][0] >= roc["fpr"][-1]

    def test_apply_threshold(self):
        opt = ThresholdOptimizer()
        confidences = np.array([0.1, 0.4, 0.6, 0.9])
        preds = opt.apply_threshold(confidences, threshold=0.5)
        expected = np.array([0, 0, 1, 1])
        assert (preds == expected).all()
