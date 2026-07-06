"""Generate synthetic enterprise UI interaction traces with embedded macro-tasks.

Usage:
    python scripts/generate_mock_data.py --n_sessions 500 --output data/synthetic/events.parquet

This script creates event sequences where each session interleaves a set of
predefined macro-tasks (invoice, email, code review) with noise events.
Ground-truth task labels are embedded so clustering quality (NMI) can be
quantitatively validated.

Task DAGs (deterministic state sequences):
  Task 1 — Invoice (ERP):      login → form → line_item → review → submit
  Task 2 — Email (Outlook):    inbox → thread → compose → sent
  Task 3 — Code Review (VSC):  open → diff → comment → merge

Noise model:
  - 40% of events are spurious (random clicks, idles, alt-tab switches)
  - Inter-event gaps follow LogNormal(mu=2, sigma=0.8) — 1-30 seconds
  - Sessions are 50-200 events long
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import numpy as np
import pandas as pd

from agentic_workflow_discovery.ingestion.schemas import EventRecord

# ── Task definitions ──────────────────────────────────────────────────────
# Each task is a list of (app, window_title, event_type) triples forming a
# deterministic state chain.

TASKS: dict[int, list[tuple[str, str, str]]] = {
    1: [
        ("ERP", "login", "focus_in"),
        ("ERP", "invoice_form", "keystroke"),
        ("ERP", "invoice_form", "click"),
        ("ERP", "line_items", "keystroke"),
        ("ERP", "review", "click"),
        ("ERP", "submitted", "click"),
    ],
    2: [
        ("Outlook", "inbox", "focus_in"),
        ("Outlook", "thread_view", "click"),
        ("Outlook", "compose", "keystroke"),
        ("Outlook", "sent", "click"),
    ],
    3: [
        ("VS Code", "file_tree", "focus_in"),
        ("VS Code", "diff_view", "click"),
        ("VS Code", "comment_form", "keystroke"),
        ("VS Code", "merged", "click"),
    ],
}

# Noise actions — these are sampled uniformly and injected between task steps.
NOISE_POOL: list[tuple[str, str, str]] = [
    ("Finder", "desktop", "click"),
    ("Slack", "#random", "focus_in"),
    ("Slack", "#random", "keystroke"),
    ("Browser", "news_site", "focus_in"),
    ("Browser", "news_site", "click"),
    ("Terminal", "zsh", "focus_in"),
    ("Terminal", "zsh", "keystroke"),
    ("Spotify", "main", "focus_in"),
    ("System", "idle", "idle"),
]


def _sample_timestamp_gap(rng: np.random.Generator) -> timedelta:
    """Log-normal inter-event gap.  Typical range: 1-30 seconds."""
    seconds = float(rng.lognormal(mean=2.0, sigma=0.8))
    return timedelta(seconds=max(0.5, seconds))


def generate_session(
    session_id: int,
    rng: np.random.Generator,
    noise_prob: float = 0.40,
    min_events: int = 50,
    max_events: int = 200,
) -> tuple[list[EventRecord], dict[str, int]]:
    """Generate one user session of interleaved macro-tasks + noise.

    Returns
    -------
    events : list[EventRecord]
        Chronologically sorted event sequence.
    ground_truth : dict[str, int]
        Mapping from event_id (hex string) → task label {0: noise, 1, 2, 3}.
    """
    user_id = uuid4()
    current_time = datetime(2025, 6, 1, 8, 0, tzinfo=UTC)

    # Determine task order for this session (shuffle, repeat)
    n_tasks = rng.integers(2, 4)
    task_order = list(TASKS.keys())
    rng.shuffle(task_order)
    task_order = task_order[:n_tasks]

    events: list[EventRecord] = []
    ground_truth: dict[str, int] = {}
    target_length = rng.integers(min_events, max_events)

    # We interleave task steps: advance one step in each task cyclically,
    # with random noise events injected between steps.
    task_pointers: dict[int, int] = {tid: 0 for tid in task_order}

    while len(events) < target_length:
        # Decide: task step or noise?
        if rng.random() < noise_prob and len(events) > 3:
            # Noise event
            app, title, etype = NOISE_POOL[rng.integers(len(NOISE_POOL))]
            current_time += _sample_timestamp_gap(rng)
            event = EventRecord(
                user_id=user_id,
                timestamp=current_time,
                app_name=app,
                window_title=title,
                event_type=etype,
            )
            events.append(event)
            ground_truth[event.event_id.hex] = 0
        else:
            # Advance one task step
            if not task_order:
                continue
            tid = task_order[rng.integers(len(task_order))]
            pointer = task_pointers[tid]
            if pointer < len(TASKS[tid]):
                app, title, etype = TASKS[tid][pointer]
                current_time += _sample_timestamp_gap(rng)
                event = EventRecord(
                    user_id=user_id,
                    timestamp=current_time,
                    app_name=app,
                    window_title=title,
                    event_type=etype,
                )
                events.append(event)
                ground_truth[event.event_id.hex] = tid
                task_pointers[tid] = pointer + 1

        # Reset completed tasks so they repeat
        for tid in task_order:
            if task_pointers[tid] >= len(TASKS[tid]):
                task_pointers[tid] = 0

    return events, ground_truth


def generate_dataset(
    n_sessions: int = 500,
    noise_prob: float = 0.40,
    seed: int = 42,
) -> tuple[list[EventRecord], dict[str, int]]:
    """Generate a full synthetic dataset.

    Returns a flat list of all events (concatenated across sessions) and
    a dictionary mapping event_id → task label.
    """
    rng = np.random.default_rng(seed)
    all_events: list[EventRecord] = []
    all_labels: dict[str, int] = {}

    for sid in range(n_sessions):
        events, labels = generate_session(sid, rng, noise_prob=noise_prob)
        all_events.extend(events)
        all_labels.update(labels)

    # Sort by timestamp to guarantee global monotonicity.
    # Real data arrives sorted; the synthetic generator produces
    # sessions that start at the same base time, so we sort here.
    all_events.sort(key=lambda e: e.timestamp)
    return all_events, all_labels


def save_parquet(events: list[EventRecord], path: str) -> None:
    """Serialize EventRecords to Parquet via pandas."""
    rows = []
    for e in events:
        rows.append(
            {
                "event_id": e.event_id.hex,
                "user_id": e.user_id.hex,
                "timestamp": e.timestamp.isoformat(),
                "app_name": e.app_name,
                "window_title": e.window_title,
                "event_type": e.event_type,
            }
        )
    df = pd.DataFrame(rows)
    df.to_parquet(path, index=False)
    print(f"Wrote {len(events)} events to {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_sessions", type=int, default=500)
    parser.add_argument("--output", default="data/synthetic/events.parquet")
    parser.add_argument("--labels_output", default="data/synthetic/labels.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    events, labels = generate_dataset(
        n_sessions=args.n_sessions,
        seed=args.seed,
    )
    save_parquet(events, args.output)

    import json

    with open(args.labels_output, "w") as f:
        json.dump(labels, f)
    print(f"Wrote {len(labels)} ground-truth labels to {args.labels_output}")
