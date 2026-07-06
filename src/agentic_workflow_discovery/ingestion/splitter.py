from __future__ import annotations

from typing import NamedTuple

from agentic_workflow_discovery.ingestion.schemas import EventRecord


class TemporalSplit(NamedTuple):
    """Strictly chronological train / validation / test split.

    Guarantees (enforced by assertions at construction):
        train.timestamp.max() < val.timestamp.min()
        val.timestamp.max() < test.timestamp.min()

    This is the single most important anti-leakage measure in the pipeline.
    A random train_test_split would leak future context into the training set,
    producing artificially high validation scores that collapse at deployment.
    """

    train: list[EventRecord]
    val: list[EventRecord]
    test: list[EventRecord]


def chronological_split(
    events: list[EventRecord],
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> TemporalSplit:
    """Split events by time-span quantiles — not by event count.

    Parameters
    ----------
    events : list[EventRecord]
        Must be sorted by timestamp (guaranteed by EventSequence validation).
    train_frac : float
        Fraction of the **time range** assigned to training.
    val_frac : float
        Fraction of the **time range** assigned to validation.

    Returns
    -------
    TemporalSplit with strictly non-overlapping timestamp intervals.

    Why split by time and not by count?
    ------------------------------------
    Consider a 10-hour session with 1 000 events in hour 1 and 10 events
    in hour 10.  A count-based split would place the first 700 events
    (hours 1-2) into training.  A time-based split correctly places the
    first 7 hours into training, which is the real deployment scenario:
    the model sees past data and predicts future data.

    Boundary rule:
    --------------
    An event whose timestamp falls exactly on a boundary is assigned to
    the *earlier* split.  This means train covers [t_min, t_train_end]
    and val covers (t_train_end, t_val_end].
    """
    if not events:
        return TemporalSplit([], [], [])

    t_min = events[0].timestamp
    t_max = events[-1].timestamp
    t_range_seconds = (t_max - t_min).total_seconds()

    train_cutoff = t_min.timestamp() + t_range_seconds * train_frac
    val_cutoff = t_min.timestamp() + t_range_seconds * (train_frac + val_frac)

    train: list[EventRecord] = []
    val: list[EventRecord] = []
    test: list[EventRecord] = []

    for e in events:
        ts = e.timestamp.timestamp()
        if ts <= train_cutoff:
            train.append(e)
        elif ts <= val_cutoff:
            val.append(e)
        else:
            test.append(e)

    # Assert no temporal overlap between splits
    if train and val:
        assert train[-1].timestamp < val[0].timestamp, (
            f"Train/val overlap: train max={train[-1].timestamp} val min={val[0].timestamp}"
        )
    if val and test:
        assert val[-1].timestamp < test[0].timestamp, (
            f"Val/test overlap: val max={val[-1].timestamp} test min={test[0].timestamp}"
        )

    return TemporalSplit(train=train, val=val, test=test)
