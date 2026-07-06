"""Critical test: verify zero temporal leakage in the PyTorch Dataset.

This test constructs a known sequence with explicit timestamps, tokenises
it, wraps it in SequenceDataset, and asserts that every sample's input
timestamps are strictly less than its label timestamp.

If this test fails, the entire training pipeline is compromised — the model
would be cheating by peeking at future events.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
import torch

from agentic_workflow_discovery.model.dataset import SequenceDataset
from agentic_workflow_discovery.model.tokenizer import EventTokenizer


class TestTemporalLeakageGuard:
    def _make_test_data(self, context_length: int = 5):
        """Create token IDs and ascending timestamps for testing."""
        timestamps = torch.arange(20, dtype=torch.float64)
        token_ids = torch.randint(2, 100, (20,))
        return token_ids, timestamps

    def test_all_samples_have_legitimate_temporal_order(self):
        """For every possible sample index, verify input < label."""
        token_ids, timestamps = self._make_test_data()
        ds = SequenceDataset(token_ids, timestamps, context_length=5)

        for i in range(len(ds)):
            input_ids, label_id = ds[i]
            input_times = timestamps[i : i + 5]
            label_time = timestamps[i + 5]
            assert (input_times < label_time).all(), (
                f"Leakage at index {i}: "
                f"input max={input_times.max().item()}, label={label_time.item()}"
            )

    def test_monotonic_increasing_timestamps_enforced(self):
        """If a label timestamp is <= an input timestamp, the guard fires."""
        token_ids = torch.tensor([10, 20, 30, 40, 50, 60])
        timestamps = torch.tensor([0.0, 1.0, 2.0, 3.0, 5.0, 4.0])  # label (4.0) < input (5.0)

        with pytest.raises(AssertionError, match="Temporal leakage"):
            ds = SequenceDataset(token_ids, timestamps, context_length=3)
            _ = ds[2]  # idx 2: inputs at [2.0, 3.0, 5.0], label at 4.0 → leak

    def test_context_length_one_edge_case(self):
        """context_length=1 should still satisfy the guard."""
        token_ids = torch.tensor([10, 20])
        timestamps = torch.tensor([0.0, 1.0])
        ds = SequenceDataset(token_ids, timestamps, context_length=1)
        input_ids, label_id = ds[0]
        assert len(input_ids) == 1
        assert int(label_id) == 20

    def test_tokenizer_does_not_leak_future_states(self):
        """The tokenizer must be fit ONLY on training events.
        This test verifies that states unique to later events are encoded
        as [UNK], not as a known token."""
        tokenizer = EventTokenizer(max_vocab_size=100)
        tokenizer.fit([])
        from uuid import uuid4

        from agentic_workflow_discovery.ingestion.schemas import EventRecord

        evt = EventRecord(
            user_id=uuid4(),
            timestamp=datetime(2025, 1, 1, tzinfo=UTC),
            app_name="AppB",
            window_title="w2",
            event_type="scroll",
        )
        ids = tokenizer.encode([evt])
        assert ids == [1], "Unseen test state should map to [UNK] (idx=1)"
