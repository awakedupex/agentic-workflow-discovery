"""End-to-end pipeline orchestrator.

Wires together:
  Ingestion → Graph → Model → Optimization

Each stage is optional — you can run just the graph module, just the model,
or the full pipeline.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from agentic_workflow_discovery.graph.builder import build_transition_graph
from agentic_workflow_discovery.graph.cluster import cluster_macro_tasks
from agentic_workflow_discovery.ingestion.pipeline import prepare_event_data
from agentic_workflow_discovery.ingestion.schemas import EventRecord
from agentic_workflow_discovery.model.dataset import SequenceDataset
from agentic_workflow_discovery.model.lstm_model import SequenceLSTM
from agentic_workflow_discovery.model.predictor import Predictor
from agentic_workflow_discovery.model.tokenizer import EventTokenizer
from agentic_workflow_discovery.model.trainer import Trainer
from agentic_workflow_discovery.optimization.controller import (
    OperatingPointController,
)

logger = logging.getLogger(__name__)


class Orchestrator:
    """End-to-end pipeline for workflow discovery and prediction.

    Usage:
        orch = Orchestrator(context_length=5)
        orch.ingest(raw_events)
        orch.build_graph()
        orch.train_model()
        result = orch.predict(context_events)
    """

    def __init__(
        self,
        context_length: int = 5,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        n_layers: int = 2,
        dropout: float = 0.2,
        lr: float = 1e-3,
        collapse_duplicates: bool = True,
        cost_fn: float = 10.0,
        cost_fp: float = 1.0,
        device: str = "auto",
    ):
        self.context_length = context_length
        self.collapse_duplicates = collapse_duplicates
        self.device = device

        # Model hyperparameters
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.lr = lr
        self.cost_fn = cost_fn
        self.cost_fp = cost_fp

        # Pipeline state (populated by calling methods below)
        self.events: list[EventRecord] | None = None
        self.split = None
        self.graph = None
        self.state_to_cluster: dict | None = None
        self.tokenizer: EventTokenizer | None = None
        self.model: SequenceLSTM | None = None
        self.trainer: Trainer | None = None
        self.predictor: Predictor | None = None
        self.controller: OperatingPointController | None = None

    # ── Stage 1: Ingestion ──────────────────────────────────────────

    def ingest(
        self,
        events: list[EventRecord],
        train_frac: float = 0.70,
        val_frac: float = 0.15,
    ) -> None:
        """Validate, clean, and chronologically split events."""
        self.events = events
        self.split = prepare_event_data(
            events,
            collapse_duplicates=self.collapse_duplicates,
            train_frac=train_frac,
            val_frac=val_frac,
        )
        logger.info(
            f"Ingested {len(events)} events → "
            f"train={len(self.split.train)}, "
            f"val={len(self.split.val)}, "
            f"test={len(self.split.test)}"
        )

    # ── Stage 2: Graph ──────────────────────────────────────────────

    def build_graph(self, n_clusters: int | None = None) -> None:
        """Build transition graph and run spectral clustering."""
        if self.split is None:
            raise RuntimeError("Call ingest() before build_graph()")

        self.graph = build_transition_graph(self.split.train)
        self.state_to_cluster = cluster_macro_tasks(
            self.graph,
            n_clusters=n_clusters,
        )
        logger.info(
            f"Graph: {self.graph.number_of_nodes()} nodes, "
            f"{self.graph.number_of_edges()} edges, "
            f"{len(set(self.state_to_cluster.values()))} clusters"
        )

    # ── Stage 3: Model ──────────────────────────────────────────────

    def train_model(
        self,
        vocab_size: int = 1000,
        max_epochs: int = 100,
        batch_size: int = 64,
        early_stop_patience: int = 5,
        checkpoint_path: str | Path | None = None,
    ) -> list[float]:
        """Tokenize, create datasets, train the LSTM model."""
        if self.split is None:
            raise RuntimeError("Call ingest() before train_model()")

        # Tokenizer — fit on training data ONLY
        self.tokenizer = EventTokenizer(max_vocab_size=vocab_size)
        self.tokenizer.fit(self.split.train)

        # Encode all splits
        train_ids = torch.tensor(self.tokenizer.encode(self.split.train), dtype=torch.long)
        val_ids = torch.tensor(self.tokenizer.encode(self.split.val), dtype=torch.long)

        # Timestamps as Unix floats (for the leakage guard)
        train_ts = torch.tensor(
            [e.timestamp.timestamp() for e in self.split.train],
            dtype=torch.float64,
        )
        val_ts = torch.tensor(
            [e.timestamp.timestamp() for e in self.split.val],
            dtype=torch.float64,
        )

        # Datasets with built-in leakage guard
        train_ds = SequenceDataset(train_ids, train_ts, context_length=self.context_length)
        val_ds = SequenceDataset(val_ids, val_ts, context_length=self.context_length)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        # Model
        actual_vocab_size = self.tokenizer.vocab_size
        self.model = SequenceLSTM(
            vocab_size=actual_vocab_size,
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            n_layers=self.n_layers,
            dropout=self.dropout,
        )

        self.trainer = Trainer(
            model=self.model,
            lr=self.lr,
            device=self.device,
        )

        losses = self.trainer.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            max_epochs=max_epochs,
            early_stop_patience=early_stop_patience,
            checkpoint_path=checkpoint_path,
        )

        # Predictor wrapper
        self.predictor = Predictor(
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
        )

        return losses

    # ── Stage 4: Optimization ───────────────────────────────────────

    def tune_threshold(
        self,
        confidences: np.ndarray,
        y_true: np.ndarray,
    ) -> float:
        """Find the optimal operating threshold under asymmetric costs."""
        self.controller = OperatingPointController(
            predictor=self.predictor,  # type: ignore[arg-type]
            cost_fn=self.cost_fn,
            cost_fp=self.cost_fp,
        )
        threshold = self.controller.fit(confidences, y_true)
        logger.info(
            f"Optimal threshold={threshold:.3f} (cost_fn={self.cost_fn}, cost_fp={self.cost_fp})"
        )
        return threshold

    # ── Prediction ──────────────────────────────────────────────────

    def predict(
        self,
        context_events: list[EventRecord],
    ) -> dict[str, Any]:
        """Predict the next event given a context window."""
        if self.predictor is None:
            raise RuntimeError("Call train_model() before predict()")
        return self.predictor.predict_next(context_events)

    def decide(
        self,
        confidence: float,
    ) -> bool:
        """Apply the operating threshold to a confidence score."""
        if self.controller is None:
            raise RuntimeError("Call tune_threshold() before decide()")
        return self.controller.decide(confidence)
