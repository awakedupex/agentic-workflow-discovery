"""Unit tests for EventTokenizer."""

from __future__ import annotations

import pytest

from agentic_workflow_discovery.model.tokenizer import (
    PAD_IDX,
    UNK_IDX,
    EventTokenizer,
)


class TestEventTokenizer:
    def test_fit_empty_events(self):
        tokenizer = EventTokenizer(max_vocab_size=100)
        tokenizer.fit([])
        assert tokenizer.vocab_size == 2  # [PAD], [UNK]

    def test_special_tokens_always_present(self, make_event):
        tokenizer = EventTokenizer(max_vocab_size=100)
        tokenizer.fit([make_event()])
        assert PAD_IDX == 0
        assert UNK_IDX == 1
        assert tokenizer.decode(PAD_IDX) == "[PAD]"
        assert tokenizer.decode(UNK_IDX) == "[UNK]"

    def test_fit_and_encode_known_state(self, make_event):
        e = make_event(app_name="Outlook", window_title="Inbox", event_type="focus_in")
        tokenizer = EventTokenizer(max_vocab_size=100)
        tokenizer.fit([e])
        ids = tokenizer.encode([e])
        assert ids == [2]  # first non-special token

    def test_unknown_state_encodes_to_unk(self, make_event):
        train_e = make_event(app_name="A", window_title="w1", event_type="click")
        test_e = make_event(app_name="Z", window_title="unknown", event_type="scroll")
        tokenizer = EventTokenizer(max_vocab_size=100)
        tokenizer.fit([train_e])
        ids = tokenizer.encode([test_e])
        assert ids == [UNK_IDX]

    def test_vocab_respects_max_size(self, make_event):
        events = [
            make_event(app_name=f"App{i}", window_title="w", event_type="click") for i in range(50)
        ]
        tokenizer = EventTokenizer(max_vocab_size=10)
        tokenizer.fit(events)
        assert tokenizer.vocab_size <= 10

    def test_round_trip_decode(self, make_event):
        e = make_event(app_name="Slack", window_title="#general", event_type="keystroke")
        tokenizer = EventTokenizer(max_vocab_size=100)
        tokenizer.fit([e])
        ids = tokenizer.encode([e])
        decoded = tokenizer.decode(ids[0])
        assert decoded == "Slack||#general||keystroke"

    def test_encode_requires_fit(self, make_event):
        tokenizer = EventTokenizer(max_vocab_size=100)
        with pytest.raises(RuntimeError, match="must be fit"):
            tokenizer.encode([make_event()])

    def test_state_exists_check(self, make_event):
        train_e = make_event(app_name="A", window_title="w1", event_type="click")
        test_e = make_event(app_name="B", window_title="w2", event_type="click")
        tokenizer = EventTokenizer(max_vocab_size=100)
        tokenizer.fit([train_e])
        assert tokenizer.state_exists(train_e)
        assert not tokenizer.state_exists(test_e)
