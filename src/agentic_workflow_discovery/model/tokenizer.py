"""Map (app, window_title, event_type) triples to integer token IDs.

Vocabulary design:
  [PAD] = 0   — padding / mask for variable-length batching
  [UNK] = 1   — any unseen state during encoding
  [2..V)      — observed (app, title, event_type) triples

The tokenizer is fit *exclusively* on training data so that unseen states
in validation/test are mapped to [UNK].  This is a deliberate anti-leakage
measure: the model never sees future-state tokens during training.
"""

from __future__ import annotations

from collections import Counter

from agentic_workflow_discovery.ingestion.schemas import EventRecord

# Reserved token IDs
PAD_IDX = 0
UNK_IDX = 1
SPECIAL_TOKENS = {"[PAD]": PAD_IDX, "[UNK]": UNK_IDX}


def _state_key(event: EventRecord) -> str:
    """Hashable string key for a UI state."""
    return f"{event.app_name}||{event.window_title}||{event.event_type}"


class EventTokenizer:
    """Bidirectional mapping between UI states and integer token IDs.

    Usage:
        tokenizer = EventTokenizer(max_vocab_size=10_000)
        tokenizer.fit(train_events)
        ids = tokenizer.encode(events)        # list[int]
        state = tokenizer.decode(token_id)     # str
    """

    def __init__(self, max_vocab_size: int = 10_000):
        self.max_vocab_size = max_vocab_size
        self._stoi: dict[str, int] = dict(SPECIAL_TOKENS)
        self._itos: dict[int, str] = {v: k for k, v in SPECIAL_TOKENS.items()}
        self._fitted = False

    @property
    def vocab_size(self) -> int:
        return len(self._stoi)

    @property
    def fitted(self) -> bool:
        return self._fitted

    def fit(self, events: list[EventRecord]) -> None:
        """Build vocabulary from a list of training events.

        Only the top `max_vocab_size - 2` most frequent states are kept;
        the rest map to [UNK].
        """
        counter: Counter[str] = Counter()
        for event in events:
            counter[_state_key(event)] += 1

        most_common = counter.most_common(self.max_vocab_size - len(SPECIAL_TOKENS))
        for key, _ in most_common:
            idx = len(self._stoi)
            self._stoi[key] = idx
            self._itos[idx] = key

        self._fitted = True

    def encode(self, events: list[EventRecord]) -> list[int]:
        """Convert events to token IDs.  Unseen states → [UNK]."""
        if not self._fitted:
            raise RuntimeError("Tokenizer must be fit before encoding")
        return [self._stoi.get(_state_key(e), UNK_IDX) for e in events]

    def decode(self, token_id: int) -> str:
        """Convert a token ID back to its string representation."""
        return self._itos.get(token_id, "[UNK]")

    def state_exists(self, event: EventRecord) -> bool:
        """Check whether a state is in the vocabulary (not [UNK])."""
        return _state_key(event) in self._stoi
