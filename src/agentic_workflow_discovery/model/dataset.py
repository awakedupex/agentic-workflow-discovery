"""PyTorch Dataset with a strict temporal leakage guard.

Every sample consists of `context_length` input tokens and 1 label token
(the next event).  The guard asserts that *all* input timestamps are
strictly less than the label timestamp.

This Dataset assumes the caller has already performed a chronological
train/val/test split at the ingestion layer.  It does NOT split — it
simply wraps pre-split token IDs and timestamps.
"""

from __future__ import annotations

import torch
from torch.utils.data import Dataset


class SequenceDataset(Dataset):
    """Sliding-window dataset over tokenised event sequences.

    Parameters
    ----------
    token_ids : torch.Tensor
        1-D tensor of integer token IDs, shape (seq_len,).
    timestamps : torch.Tensor
        1-D tensor of Unix timestamps matching token_ids, shape (seq_len,).
    context_length : int
        Number of input tokens per sample (default 5).

    Guarantee
    ---------
    __getitem__(i) returns (input_ids, label_id) where:
        input_ids = token_ids[i : i + context_length]
        label_id  = token_ids[i + context_length]

    An assertion verifies that:
        max(timestamps[i : i + context_length]) < timestamps[i + context_length]

    This guarantee is the primary anti-leakage mechanism: the label's
    timestamp must always be *after* every input token's timestamp.
    """

    def __init__(
        self,
        token_ids: torch.Tensor,
        timestamps: torch.Tensor,
        context_length: int = 5,
    ):
        assert token_ids.dim() == 1, "token_ids must be 1-D"
        assert timestamps.dim() == 1, "timestamps must be 1-D"
        assert token_ids.shape == timestamps.shape, (
            f"Shape mismatch: {token_ids.shape} vs {timestamps.shape}"
        )

        self.token_ids = token_ids
        self.timestamps = timestamps
        self.context_length = context_length

    def __len__(self) -> int:
        return max(0, len(self.token_ids) - self.context_length)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        input_ids = self.token_ids[idx : idx + self.context_length]  # (C,)
        label_id = self.token_ids[idx + self.context_length]  # scalar

        # ── Temporal leakage guard ──────────────────────────────────
        # Every input token's timestamp must be strictly less than the
        # label's timestamp.  If this fires, the caller has a bug in
        # the train/val/test split or the tokenisation pipeline.
        input_times = self.timestamps[idx : idx + self.context_length]
        label_time = self.timestamps[idx + self.context_length]

        assert (input_times < label_time).all(), (
            f"Temporal leakage at index {idx}: "
            f"max(input_timestamps)={input_times.max().item():.4f} >= "
            f"label_timestamp={label_time.item():.4f}"
        )

        return input_ids, label_id
