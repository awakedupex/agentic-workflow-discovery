"""Training loop with validation, early stopping, and MPS/CPU support.

Design decisions:
-----------------
- AdamW with decoupled weight decay (standard for transformer-style models).
- Gradient clipping (max_norm=1.0) prevents loss spikes from destabilising
  training on noisy UI sequences.
- Early stopping (patience=5) on validation loss to avoid overfitting.
- Device-agnostic: auto-selects MPS (Apple Silicon) if available, else CPU.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from agentic_workflow_discovery.model.lstm_model import SequenceLSTM

logger = logging.getLogger(__name__)


def _resolve_device() -> torch.device:
    """Auto-select device: MPS > CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class Trainer:
    """Train a SequenceLSTM model with validation and early stopping.

    Parameters
    ----------
    model : SequenceLSTM
    lr : float
        Learning rate (default 1e-3).
    weight_decay : float
        AdamW weight decay (default 1e-5).
    gradient_clip_norm : float
        Max gradient norm (default 1.0).
    device : torch.device | str
        "auto" for auto-detect, or explicit device.
    """

    def __init__(
        self,
        model: SequenceLSTM,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        gradient_clip_norm: float = 1.0,
        device: torch.device | str = "auto",
    ):
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.gradient_clip_norm = gradient_clip_norm

        if isinstance(device, str) and device == "auto":
            device = _resolve_device()
        self.device = torch.device(device)
        self.model.to(self.device)

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)  # ignore [PAD]

    def train_epoch(self, loader: DataLoader) -> float:
        """Run one training epoch.  Returns average loss."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for input_ids, labels in loader:
            input_ids = input_ids.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(input_ids)  # (B, V)
            loss = self.criterion(logits, labels)  # scalar
            loss.backward()

            if self.gradient_clip_norm > 0:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.gradient_clip_norm,
                )

            self.optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    @torch.inference_mode()
    def evaluate(self, loader: DataLoader) -> float:
        """Evaluate on validation set.  Returns average loss."""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        for input_ids, labels in loader:
            input_ids = input_ids.to(self.device)
            labels = labels.to(self.device)

            logits = self.model(input_ids)
            loss = self.criterion(logits, labels)

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        max_epochs: int = 100,
        early_stop_patience: int = 5,
        checkpoint_path: str | Path | None = None,
    ) -> list[float]:
        """Full training loop with early stopping.

        Returns
        -------
        train_losses : list[float]
            Loss per epoch.
        """
        best_val_loss = float("inf")
        patience_counter = 0
        train_losses: list[float] = []

        for epoch in range(1, max_epochs + 1):
            train_loss = self.train_epoch(train_loader)
            train_losses.append(train_loss)

            if val_loader is not None:
                val_loss = self.evaluate(val_loader)
                logger.info(
                    f"Epoch {epoch:3d}/{max_epochs}  train={train_loss:.4f}  val={val_loss:.4f}"
                )

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    if checkpoint_path:
                        self.save(checkpoint_path)
                else:
                    patience_counter += 1
                    if patience_counter >= early_stop_patience:
                        logger.info(
                            f"Early stopping at epoch {epoch} "
                            f"(val loss did not improve for {patience_counter} epochs)"
                        )
                        break
            else:
                logger.info(f"Epoch {epoch:3d}/{max_epochs}  train={train_loss:.4f}")

        return train_losses

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)
        logger.info(f"Model saved to {path}")

    def load(self, path: str | Path) -> None:
        path = Path(path)
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.to(self.device)
        logger.info(f"Model loaded from {path}")
