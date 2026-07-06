from __future__ import annotations

from agentic_workflow_discovery.ingestion.cleaner import (
    collapse_consecutive_duplicates,
)
from agentic_workflow_discovery.ingestion.schemas import EventRecord, EventSequence
from agentic_workflow_discovery.ingestion.splitter import (
    TemporalSplit,
    chronological_split,
)


def prepare_event_data(
    events: list[EventRecord],
    collapse_duplicates: bool = True,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> TemporalSplit:
    """End-to-end ingestion pipeline: validate → clean → split.

    Parameters
    ----------
    events : list[EventRecord]
        Raw event sequence (order must be chronological within each session).
    collapse_duplicates : bool
        If True, collapse consecutive duplicate state triples (default: True).
    train_frac, val_frac : float
        Chronological split fractions.

    Returns
    -------
    TemporalSplit with train / val / test event lists.

    Raises
    ------
    ValueError
        If events fail structural validation (non-monotonic timestamps,
        missing timezone, empty strings, etc.).
    """
    # Step 1 — Structural validation
    # EventSequence.__init__ raises ValueError on any violation.
    EventSequence(events=events)

    # Step 2 — UI loop collapse
    working = events
    if collapse_duplicates:
        working = collapse_consecutive_duplicates(working)

    # Step 3 — Chronological split
    return chronological_split(
        working,
        train_frac=train_frac,
        val_frac=val_frac,
    )
