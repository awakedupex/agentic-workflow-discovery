"""Unit tests for collapse_consecutive_duplicates."""

from __future__ import annotations

from agentic_workflow_discovery.ingestion.cleaner import (
    collapse_consecutive_duplicates,
)


class TestCollapseConsecutiveDuplicates:
    """Edge cases verified:
    - Empty input
    - Single event preserved
    - No duplicates → unchanged
    - Consecutive duplicates → collapsed
    - Interleaved duplicates → preserved (pattern A → B → A)
    """

    def test_empty_input(self):
        assert collapse_consecutive_duplicates([]) == []

    def test_single_event_preserved(self, make_event):
        e = make_event()
        assert collapse_consecutive_duplicates([e]) == [e]

    def test_no_duplicates_left_unchanged(self, make_event):
        e1 = make_event(app_name="A", window_title="w1", event_type="click")
        e2 = make_event(app_name="B", window_title="w2", event_type="focus_in")
        assert collapse_consecutive_duplicates([e1, e2]) == [e1, e2]

    def test_consecutive_duplicates_collapsed(self, make_event):
        e1 = make_event(app_name="Outlook", window_title="Inbox", event_type="focus_in")
        e2 = make_event(app_name="Outlook", window_title="Inbox", event_type="focus_in")
        e3 = make_event(app_name="Outlook", window_title="Inbox", event_type="focus_in")
        e4 = make_event(app_name="Slack", window_title="#general", event_type="click")
        result = collapse_consecutive_duplicates([e1, e2, e3, e4])
        assert result == [e1, e4], f"Expected 2 events, got {len(result)}"

    def test_interleaved_duplicates_preserved(self, make_event):
        """A → B → A should NOT be collapsed (different transitions)."""
        e1 = make_event(app_name="Outlook", window_title="Inbox", event_type="click")
        e2 = make_event(app_name="Slack", window_title="#general", event_type="click")
        e3 = make_event(app_name="Outlook", window_title="Inbox", event_type="click")
        result = collapse_consecutive_duplicates([e1, e2, e3])
        assert result == [e1, e2, e3]

    def test_all_identical_collapses_to_one(self, make_event):
        e = make_event()
        result = collapse_consecutive_duplicates([e, e, e, e, e])
        assert len(result) == 1

    def test_only_three_fields_matter_for_identity(self, make_event):
        """Events with different user_id or event_id should still collapse
        if their (app, title, event_type) triples match."""
        e1 = make_event(app_name="A", window_title="w1", event_type="click")
        e2 = make_event(app_name="A", window_title="w1", event_type="click")
        # e1.user_id != e2.user_id — but should still collapse
        result = collapse_consecutive_duplicates([e1, e2])
        assert len(result) == 1
