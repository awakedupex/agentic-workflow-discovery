from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator

EventType = Literal[
    "click",
    "keystroke",
    "focus_in",
    "focus_out",
    "scroll",
    "copy",
    "paste",
    "idle",
]


class EventRecord(BaseModel):
    """A single UI interaction event.

    Design decisions:
    -----------------
    frozen=True : prevents mutation after construction. An EventRecord flows
        through 5+ pipeline stages; accidental mutation would corrupt the
        transition graph silently.
    extra="forbid" : rejects unknown fields at construction time rather than
        silently dropping them. This forces any upstream schema change to
        fail loudly at the ingestion boundary.
    UUID event_id : globally unique; enables idempotent dedup if the same
        event is ingested twice from different sources.
    timezone-aware timestamp : all downstream time arithmetic (chronological
        split, gap detection) is undefined without tz.
    """

    model_config = {"frozen": True, "extra": "forbid"}

    event_id: UUID = Field(default_factory=uuid4)
    user_id: UUID
    timestamp: datetime
    app_name: str = Field(min_length=1, max_length=256)
    window_title: str = Field(min_length=0, max_length=512)
    event_type: EventType

    @field_validator("timestamp")
    @classmethod
    def _must_have_timezone(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError(f"timestamp must be timezone-aware, got {v}")
        return v


class EventSequence(BaseModel):
    """An ordered, validated sequence of events (one user session).

    The monotonic_timestamps validator is O(n) and runs once at construction.
    After that, every consumer of this object can assume sorted input without
    re-checking — zero-cost invariant enforcement.
    """

    events: list[EventRecord]

    @field_validator("events")
    @classmethod
    def _monotonic_timestamps(cls, v: list[EventRecord]) -> list[EventRecord]:
        for i in range(1, len(v)):
            if v[i].timestamp < v[i - 1].timestamp:
                raise ValueError(
                    f"Events out of order at index {i}: {v[i - 1].timestamp} > {v[i].timestamp}"
                )
        return v
