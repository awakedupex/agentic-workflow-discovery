"""Latency budget test: model inference must complete within 15ms per prefix.

This is a critical engineering constraint — the model runs online, predicting
the next action as the user interacts.  Every millisecond of latency degrades
the user experience.

The test runs 100 timed forward passes after a warm-up phase and asserts
that the average latency stays under 15ms on CPU.
"""

from __future__ import annotations

import time

import pytest
import torch

from agentic_workflow_discovery.model.lstm_model import SequenceLSTM

pytestmark = pytest.mark.benchmark

# Target: 15ms per inference pass on CPU
LATENCY_BUDGET_MS = 15.0


@pytest.fixture(scope="module")
def model_and_input() -> tuple[SequenceLSTM, torch.Tensor]:
    """Small model matching production scale: vocab=1000, d=128, h=256, 2 layers."""
    torch.manual_seed(42)
    model = SequenceLSTM(
        vocab_size=1000,
        embedding_dim=128,
        hidden_dim=256,
        n_layers=2,
        dropout=0.0,
    )
    model.eval()

    # Batch size 1, context_length=5 (matches training)
    input_ids = torch.randint(0, 1000, (1, 5), dtype=torch.long)
    return model, input_ids


class TestLatencyBudget:
    def test_inference_latency_below_15ms(self, model_and_input):
        """Average forward-pass latency must be < 15ms."""
        model, input_ids = model_and_input

        # Warmup — 10 passes to warm caches and avoid one-time CUDA/MPS init
        for _ in range(10):
            _ = model(input_ids)

        # Timed runs
        n_runs = 100
        times: list[float] = []

        for _ in range(n_runs):
            start = time.perf_counter()
            _ = model(input_ids)
            end = time.perf_counter()
            times.append((end - start) * 1000)  # convert to ms

        avg_latency = sum(times) / len(times)
        max_latency = max(times)

        assert avg_latency < LATENCY_BUDGET_MS, (
            f"Average latency {avg_latency:.2f}ms exceeds budget "
            f"{LATENCY_BUDGET_MS}ms  (max={max_latency:.2f}ms)"
        )

    def test_p99_latency_below_20ms(self, model_and_input):
        """99th percentile latency should not spike above 20ms (headroom)."""
        model, input_ids = model_and_input

        for _ in range(10):
            _ = model(input_ids)

        n_runs = 200
        times: list[float] = []

        for _ in range(n_runs):
            start = time.perf_counter()
            _ = model(input_ids)
            times.append((time.perf_counter() - start) * 1000)

        times.sort()
        p99_idx = int(len(times) * 0.99)
        p99 = times[p99_idx]

        assert p99 < 20.0, f"P99 latency {p99:.2f}ms exceeds 20ms headroom budget"
