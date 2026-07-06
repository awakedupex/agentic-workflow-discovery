from __future__ import annotations

from agentic_workflow_discovery.ingestion.schemas import EventRecord


def collapse_consecutive_duplicates(events: list[EventRecord]) -> list[EventRecord]:
    """Collapse consecutive identical (app, window_title, event_type) triples.

    Why this matters:
    -----------------
    Enterprise clickstreams are full of "UI loops" — users rapidly click
    back and forth between two fields, or an app fires redundant focus events
    on every keystroke.  If fed directly into a transition matrix, these loops
    produce a dense subgraph that can dominate the Laplacian spectrum and
    hide the real macro-task structure.

    Strategy:
    ---------
    Walk the sequence linearly; keep the *first* event in each run of identical
    triples and drop the rest.  This preserves the existence of the transition
    *into* that state and *out of* it, but removes the multiplicity.

    Trade-off:
    ----------
    We lose fine-grained timing information (dwell time).  If dwell time
    becomes a signal for task boundaries later, we can replace this with an
    inter-event gap threshold instead of blind collapse.

    Complexity: O(n), single pass, no allocations beyond the output list.
    """
    if not events:
        return []

    cleaned: list[EventRecord] = [events[0]]

    for i in range(1, len(events)):
        prev = cleaned[-1]
        curr = events[i]

        # A "duplicate" means all three state components match.
        # We deliberately exclude event_id and timestamp from this check.
        if (
            curr.app_name == prev.app_name
            and curr.window_title == prev.window_title
            and curr.event_type == prev.event_type
        ):
            continue

        cleaned.append(curr)

    return cleaned
