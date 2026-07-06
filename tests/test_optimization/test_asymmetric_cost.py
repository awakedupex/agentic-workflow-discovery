"""Unit tests for AsymmetricCost."""

from __future__ import annotations

import numpy as np

from agentic_workflow_discovery.optimization.cost import AsymmetricCost


class TestAsymmetricCost:
    def test_perfect_predictions_zero_cost(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        cost = AsymmetricCost(cost_fn=10.0, cost_fp=1.0)
        result = cost.evaluate(y_true, y_pred)
        assert result["total_cost"] == 0.0
        assert result["fn"] == 0
        assert result["fp"] == 0

    def test_all_missed_high_fn_cost(self):
        y_true = np.array([1, 1, 1])
        y_pred = np.array([0, 0, 0])
        cost = AsymmetricCost(cost_fn=10.0, cost_fp=1.0)
        result = cost.evaluate(y_true, y_pred)
        assert result["total_cost"] == 30.0  # 3 FN × 10
        assert result["fn"] == 3
        assert result["fp"] == 0

    def test_all_false_alarms_low_fp_cost(self):
        y_true = np.array([0, 0, 0])
        y_pred = np.array([1, 1, 1])
        cost = AsymmetricCost(cost_fn=10.0, cost_fp=1.0)
        result = cost.evaluate(y_true, y_pred)
        assert result["total_cost"] == 3.0  # 3 FP × 1
        assert result["fp"] == 3
        assert result["fn"] == 0

    def test_fn_cost_dominates_total(self):
        """With a 10:1 ratio, a FN should cost 10× more than a FP."""
        cost = AsymmetricCost(cost_fn=10.0, cost_fp=1.0)
        fn_only = cost.evaluate(y_true=np.array([1]), y_pred=np.array([0]))
        fp_only = cost.evaluate(y_true=np.array([0]), y_pred=np.array([1]))
        assert fn_only["total_cost"] == 10.0 * fp_only["total_cost"]

    def test_cost_ratio_property(self):
        cost = AsymmetricCost(cost_fn=20.0, cost_fp=1.0)
        assert cost.ratio == 20.0

    def test_confusion_matrix_counts(self):
        y_true = np.array([0, 0, 0, 1, 1, 1, 1])
        y_pred = np.array([0, 0, 1, 0, 0, 1, 1])
        cost = AsymmetricCost(cost_fn=10.0, cost_fp=1.0)
        result = cost.evaluate(y_true, y_pred)
        assert result["tn"] == 2  # true negatives
        assert result["fp"] == 1  # false positives
        assert result["fn"] == 2  # false negatives
        assert result["tp"] == 2  # true positives
        assert result["n_pos"] == 4
        assert result["n_neg"] == 3
