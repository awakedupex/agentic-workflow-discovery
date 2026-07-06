"""Unit tests for build_transition_graph."""

from __future__ import annotations

from agentic_workflow_discovery.graph.builder import build_transition_graph
from agentic_workflow_discovery.ingestion.cleaner import (
    collapse_consecutive_duplicates,
)


class TestBuildTransitionGraph:
    def test_empty_events_returns_empty_graph(self):
        G = build_transition_graph([])
        assert G.number_of_nodes() == 0
        assert G.number_of_edges() == 0

    def test_single_event_produces_isolated_node(self, make_event):
        G = build_transition_graph([make_event()])
        assert G.number_of_nodes() == 1
        assert G.number_of_edges() == 0

    def test_two_states_produces_one_edge(self, make_event):
        e1 = make_event(app_name="A", window_title="w1", event_type="click")
        e2 = make_event(app_name="B", window_title="w2", event_type="focus_in")
        G = build_transition_graph([e1, e2])
        assert G.number_of_nodes() == 2
        assert G.number_of_edges() == 1

    def test_collapsed_duplicates_do_not_create_self_loops(self, make_event):
        """Five identical events → collapse to one → zero edges."""
        raw = [make_event(app_name="A", window_title="w1", event_type="click")] * 5
        collapsed = collapse_consecutive_duplicates(raw)
        G = build_transition_graph(collapsed)
        assert G.number_of_nodes() == 1
        assert G.number_of_edges() == 0

    def test_edge_weight_accumulates_across_repeated_transitions(self, make_event):
        """A→B repeated 3 times → edge weight = 3."""
        e1 = make_event(app_name="A", window_title="w1", event_type="click")
        e2 = make_event(app_name="B", window_title="w2", event_type="click")
        events = [e1, e2, e1, e2, e1, e2]  # A→B × 3
        G = build_transition_graph(events)
        src = ("A", "w1", "click")
        dst = ("B", "w2", "click")
        assert G.has_edge(src, dst)
        assert G.edges[src, dst]["weight"] == 3

    def test_self_loops_not_recorded(self, make_event):
        """A→A transitions should never create an edge."""
        e1 = make_event(app_name="A", window_title="w1", event_type="click")
        G = build_transition_graph([e1, e1])
        assert G.number_of_edges() == 0

    def test_graph_is_directed(self, make_event):
        """A→B is recorded, but B→A is a separate edge."""
        e1 = make_event(app_name="A", window_title="w1", event_type="click")
        e2 = make_event(app_name="B", window_title="w2", event_type="click")
        e3 = make_event(app_name="A", window_title="w1", event_type="click")
        G = build_transition_graph([e1, e2, e3])
        src_a = ("A", "w1", "click")
        src_b = ("B", "w2", "click")
        assert G.has_edge(src_a, src_b)
        assert G.has_edge(src_b, src_a)
        assert G.number_of_edges() == 2

    def test_all_nodes_present_including_terminal(self, make_event):
        """The last event in the sequence produces a node."""
        e1 = make_event(app_name="A", window_title="w1", event_type="click")
        e2 = make_event(app_name="B", window_title="w2", event_type="click")
        G = build_transition_graph([e1, e2])
        assert G.has_node(("B", "w2", "click"))
