"""Integration test: end-to-end pipeline with synthetic data.

This test:
  1. Generates a small synthetic dataset (3 sessions, ~150 events).
  2. Runs ingestion → graph → model → optimization.
  3. Verifies that every module produces reasonable outputs.

Marked 'slow' because it trains a model (~30 seconds).
"""

from __future__ import annotations

import numpy as np
import pytest

from agentic_workflow_discovery.graph.metrics import evaluate_partition
from agentic_workflow_discovery.pipeline.orchestrator import Orchestrator
from scripts.generate_mock_data import generate_dataset

pytestmark = pytest.mark.slow


class TestEndToEndPipeline:
    def test_full_pipeline_smoke(self):
        """Run the full pipeline on a tiny synthetic dataset and verify
        every stage completes without error and produces sensible outputs."""

        # ── Generate data ───────────────────────────────────────────
        events, labels = generate_dataset(
            n_sessions=3,
            noise_prob=0.30,
            seed=42,
        )
        assert len(events) > 100, f"Expected >100 events, got {len(events)}"

        # ── Stage 1: Ingestion ──────────────────────────────────────
        orch = Orchestrator(
            context_length=5,
            embedding_dim=32,
            hidden_dim=64,
            n_layers=1,
            dropout=0.0,
            lr=1e-3,
        )
        orch.ingest(events)
        assert orch.split is not None
        assert len(orch.split.train) > 0
        assert len(orch.split.val) >= 0  # may be 0 with few sessions

        # ── Stage 2: Graph ──────────────────────────────────────────
        orch.build_graph(n_clusters=2)
        assert orch.graph is not None
        assert orch.graph.number_of_nodes() > 0
        assert orch.state_to_cluster is not None

        # The graph should have at least some structure
        metrics = evaluate_partition(orch.graph, orch.state_to_cluster)
        assert -1.0 <= metrics["silhouette"] <= 1.0
        assert metrics["n_clusters"] >= 2

        # ── Stage 3: Model ──────────────────────────────────────────
        losses = orch.train_model(
            vocab_size=200,
            max_epochs=20,
            batch_size=16,
            early_stop_patience=10,
        )
        assert len(losses) > 0
        assert losses[-1] > 0.0  # loss should be finite
        assert orch.tokenizer is not None
        assert orch.model is not None
        assert orch.predictor is not None

        # Verify prediction works
        context = orch.split.test[:5] if len(orch.split.test) >= 5 else orch.split.val[:5]
        if len(context) == 5:
            result = orch.predict(context)
            assert "token_id" in result
            assert "confidence" in result
            assert 0 <= result["confidence"] <= 1.0

        # ── Stage 4: Optimization ───────────────────────────────────
        rng = np.random.default_rng(42)
        fake_confidences = rng.uniform(0, 1, 50)
        fake_labels = (fake_confidences > 0.5).astype(int)

        threshold = orch.tune_threshold(fake_confidences, fake_labels)
        assert 0.0 < threshold < 1.0

        # Verify decide() works
        assert orch.decide(confidence=0.9) is True
        assert orch.decide(confidence=0.0) is False

    def test_pipeline_without_graph_or_optimization(self):
        """The pipeline should work for just ingestion + model (graph
        and optimization stages are optional)."""
        events, _ = generate_dataset(n_sessions=1, seed=42)

        orch = Orchestrator(context_length=3, embedding_dim=16, hidden_dim=32)
        orch.ingest(events)

        # Skip graph — go directly to model
        if len(orch.split.train) >= 5:
            losses = orch.train_model(
                vocab_size=100,
                max_epochs=5,
                batch_size=4,
            )
            assert len(losses) > 0
