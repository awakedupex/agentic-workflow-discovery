"""Shared test fixtures for all test modules."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from agentic_workflow_discovery.ingestion.schemas import EventRecord


@pytest.fixture
def make_event():
    """Factory fixture — returns a callable that creates EventRecords
    with sensible defaults and allows selective overrides.

    Usage in tests:
        def test_something(make_event):
            e = make_event(app_name="Outlook", event_type="click")
    """

    def _make_event(
        app_name: str = "TestApp",
        window_title: str = "DefaultWindow",
        event_type: str = "click",
        timestamp: datetime | None = None,
        **overrides,
    ) -> EventRecord:
        if timestamp is None:
            timestamp = datetime(2025, 6, 1, tzinfo=UTC)
        return EventRecord(
            user_id=uuid4(),
            timestamp=timestamp,
            app_name=app_name,
            window_title=window_title,
            event_type=event_type,
            **overrides,
        )

    return _make_event


@pytest.fixture
def sample_sequence(make_event):
    """A canonical 5-event sequence used across multiple tests."""
    return [
        make_event(app_name="Outlook", window_title="Inbox", event_type="focus_in"),
        make_event(app_name="Outlook", window_title="Inbox", event_type="click"),
        make_event(app_name="VS Code", window_title="main.py", event_type="focus_in"),
        make_event(app_name="VS Code", window_title="main.py", event_type="keystroke"),
        make_event(app_name="Slack", window_title="#general", event_type="focus_in"),
    ]


@pytest.fixture
def disjoint_tasks(make_event):
    """Two completely separate task graphs for clustering tests.

    Task A (ERP invoice):  login → form → review → submit
    Task B (Email):        inbox → read → reply → send

    No shared (app, title, event_type) triples between tasks.
    Used to verify NMI == 1.0 in spectral clustering.
    """
    task_a = [
        make_event(app_name="ERP", window_title="login", event_type="focus_in"),
        make_event(app_name="ERP", window_title="form", event_type="click"),
        make_event(app_name="ERP", window_title="review", event_type="click"),
        make_event(app_name="ERP", window_title="submit", event_type="click"),
    ]
    task_b = [
        make_event(app_name="Mail", window_title="inbox", event_type="focus_in"),
        make_event(app_name="Mail", window_title="thread", event_type="click"),
        make_event(app_name="Mail", window_title="compose", event_type="keystroke"),
        make_event(app_name="Mail", window_title="sent", event_type="click"),
    ]
    return task_a + task_b
