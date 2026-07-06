"""Build a directed, weighted NetworkX graph from UI event sequences.

Design rationale for NetworkX:
-------------------------------
- Readability: graph algorithms (weakly connected components, PageRank,
  clustering coefficient) are one-liners rather than sparse-matrix math.
- Serialization: GEXF/GraphML export for Gephi visualization during
  interview walkthroughs and debugging.
- Scale: enterprise UI traces typically yield 5-20 K unique states.
  NetworkX handles this comfortably.  At 100 K+ states we'd migrate to
  igraph or a CSR matrix with scipy.sparse.
"""

from __future__ import annotations

from collections import Counter

import networkx as nx

from agentic_workflow_discovery.ingestion.schemas import EventRecord

# A UI state is uniquely identified by (app, window_title, event_type).
# Using a named tuple gives us free hashing + readable repr in debug output.
StateKey = tuple[str, str, str]


def _to_state_key(event: EventRecord) -> StateKey:
    """Derive a hashable state key from an event."""
    return (event.app_name, event.window_title, event.event_type)


def build_transition_graph(events: list[EventRecord]) -> nx.DiGraph:
    """Construct a directed, weighted transition graph from an event sequence.

    Algorithm
    ---------
    1. Walk the event list with a sliding window of size 2.
    2. For each pair (src, dst), increment an edge counter.
    3. Self-loops (src == dst) are excluded — they carry no transition
       signal and would inflate the diagonal of the Laplacian.

    Nodes are inserted even if they have zero outgoing edges (the last
    event in a sequence) so that every seen state appears in the graph.

    Parameters
    ----------
    events : list[EventRecord]
        Sorted event sequence (should already be collapse-cleaned).

    Returns
    -------
    nx.DiGraph
        Nodes = UI states (app, title, event_type).
        Edges weighted by integer transition count.
    """
    G = nx.DiGraph()

    # Phase 1: insert every event as a node (including isolates — the
    # last event in a sequence has no outgoing edge but is still a state).
    for event in events:
        G.add_node(_to_state_key(event))

    # Phase 2: walk adjacent pairs and count directed transitions.
    edge_counter: Counter[tuple[StateKey, StateKey]] = Counter()

    for i in range(len(events) - 1):
        src = _to_state_key(events[i])
        dst = _to_state_key(events[i + 1])

        if src != dst:
            edge_counter[(src, dst)] += 1

    for (src, dst), weight in edge_counter.items():
        G.add_edge(src, dst, weight=weight)

    return G
