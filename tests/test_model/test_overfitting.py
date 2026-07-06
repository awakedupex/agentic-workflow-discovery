"""Overfitting sanity check: the model must be able to memorise a tiny dataset.

This is the "smoke test" for the training loop:
- Create 5 short sequences with known tokens
- Train the model until it reaches near-zero loss
- If this fails, the model architecture or training loop is fundamentally broken.

Marked 'slow' because it takes ~30 seconds of training.
"""

from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from agentic_workflow_discovery.model.lstm_model import SequenceLSTM
from agentic_workflow_discovery.model.trainer import Trainer

# Re-use the ingestion make_event fixture via conftest
pytestmark = pytest.mark.slow


def _make_overfit_data(
    n_sequences: int = 5,
    seq_len: int = 6,
    vocab_size: int = 50,
) -> tuple[DataLoader, SequenceLSTM]:
    """Create a micro-batch dataset and a small model for overfitting."""
    torch.manual_seed(42)

    # Fixed patterns the model must memorise
    X = torch.randint(2, vocab_size, (n_sequences, seq_len - 1))
    y = torch.randint(2, vocab_size, (n_sequences,))
    # Use a different seed for labels to ensure mapping isn't trivial
    y = (X[:, 0] + 1) % vocab_size  # deterministic mapping: label = (first_token + 1) % V

    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=n_sequences, shuffle=False)

    model = SequenceLSTM(
        vocab_size=vocab_size,
        embedding_dim=32,
        hidden_dim=64,
        n_layers=1,
        dropout=0.0,  # zero dropout for overfitting test
    )

    return loader, model


class TestOverfitting:
    def test_overfit_micro_batch(self):
        """The model should reach near-zero loss on 5 sequences."""
        loader, model = _make_overfit_data(n_sequences=5, vocab_size=50)
        trainer = Trainer(model, lr=1e-3, gradient_clip_norm=0.0)

        losses = trainer.fit(
            train_loader=loader,
            max_epochs=500,
            early_stop_patience=100,  # don't early stop; we want to see convergence
        )

        final_loss = losses[-1]
        assert final_loss < 0.05, (
            f"Model failed to overfit: final loss={final_loss:.4f} "
            f"(expected < 0.05).  Training loop may be broken."
        )

    def test_overfit_convergence_rate(self):
        """Loss should decrease monotonically (roughly)."""
        loader, model = _make_overfit_data(n_sequences=5, vocab_size=50)
        trainer = Trainer(model, lr=1e-3, gradient_clip_norm=0.0)

        losses = trainer.fit(
            train_loader=loader,
            max_epochs=200,
            early_stop_patience=100,
        )

        # Loss at epoch 100 should be lower than at epoch 10
        assert losses[-1] < losses[0], (
            f"Loss did not decrease: loss[0]={losses[0]:.4f}, loss[-1]={losses[-1]:.4f}"
        )
