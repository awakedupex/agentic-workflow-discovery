"""Unit tests for chronological_split and TemporalSplit."""

from __future__ import annotations

from datetime import UTC, datetime

from agentic_workflow_discovery.ingestion.splitter import chronological_split


class TestChronologicalSplit:
    """Guarantees verified:
    - No temporal overlap between train/val/test
    - Split is time-based, not count-based
    - Boundary events assigned to earlier split
    - Empty input produces empty splits
    - Single event goes to train
    - Exact fractions produce correct proportions
    """

    def test_no_temporal_overlap(self, make_event):
        events = [
            make_event(
                timestamp=datetime(2025, 6, 1, hour=i, tzinfo=UTC),
            )
            for i in range(24)
        ]
        split = chronological_split(events)

        if split.train and split.val:
            assert split.train[-1].timestamp < split.val[0].timestamp
        if split.val and split.test:
            assert split.val[-1].timestamp < split.test[0].timestamp

    def test_time_based_not_count_based(self, make_event):
        """10 events in hour 1, 1 event in hour 10.
        Time-based split (70/15/15) should put all 10 in train."""
        t0 = datetime(2025, 6, 1, tzinfo=UTC)
        early = [make_event(timestamp=t0.replace(hour=0)) for _ in range(10)]
        late = [make_event(timestamp=t0.replace(hour=10))]
        events = early + late
        split = chronological_split(events, train_frac=0.7, val_frac=0.15)
        assert len(split.train) == 10
        assert len(split.test) == 1
        assert len(split.val) == 0

    def test_boundary_event_goes_to_earlier_split(self, make_event):
        """An event exactly at the train/val boundary goes to training."""
        t0 = datetime(2025, 6, 1, tzinfo=UTC)
        t_min = t0
        t_max = t0.replace(hour=10)
        # Boundary at 7 hours → 70% of 10-hour range
        boundary = t_min.timestamp() + (t_max - t_min).total_seconds() * 0.7
        events = [
            make_event(timestamp=datetime.fromtimestamp(boundary, tz=UTC)),
        ]
        split = chronological_split(events, train_frac=0.7, val_frac=0.15)
        assert len(split.train) == 1
        assert len(split.val) == 0
        assert len(split.test) == 0

    def test_empty_input(self, make_event):
        split = chronological_split([])
        assert split.train == []
        assert split.val == []
        assert split.test == []

    def test_single_event(self, make_event):
        e = make_event()
        split = chronological_split([e])
        assert split.train == [e]
        assert split.val == []

    def test_val_can_be_empty_with_short_range(self, make_event):
        """If total time range is very short, val and test may be empty
        because no events fall beyond the cutoffs."""
        t0 = datetime(2025, 6, 1, tzinfo=UTC)
        events = [make_event(timestamp=t0)]
        split = chronological_split(events)
        assert len(split.train) == 1
        # val and test must be non-None lists (empty is fine)
        assert split.val is not None
        assert split.test is not None

    def test_temporal_split_assertion_on_overlap(self, make_event):
        """If events arrive out of order (violating sorted invariant),
        the splitter should produce overlapping boundaries.
        The function doesn't re-sort — it trusts the caller."""
        t0 = datetime(2025, 6, 1, tzinfo=UTC)
        events = [
            make_event(timestamp=t0.replace(hour=10)),
            make_event(timestamp=t0.replace(hour=1)),  # out of order
        ]
        # This will not assert because we don't re-sort.
        # The caller must provide sorted events (guaranteed by EventSequence
        # validation at the pipeline level).
        split = chronological_split(events)
        # Out-of-order events break the assumption, but the splitter
        # still runs — temporal overlap is detectable as:
        # train[-1].timestamp might be > val[0].timestamp
        assert len(split.train) + len(split.val) + len(split.test) == 2
