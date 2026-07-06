"""Unit tests for cluster_macro_tasks and evaluate_partition.

The critical test is test_disjoint_tasks_perfect_separation — it verifies
that spectral clustering achieves NMI = 1.0 on two completely disconnected
task graphs.  This is the mathematical sanity check for the entire graph
module.
"""

from __future__ import annotations

import pytest

from agentic_workflow_discovery.graph.builder import build_transition_graph
from agentic_workflow_discovery.graph.cluster import (
    auto_detect_n_clusters,
    cluster_macro_tasks,
)
from agentic_workflow_discovery.graph.metrics import evaluate_partition


class TestAutoDetectClusters:
    def test_disconnected_graphs_detect_two_components(self, disjoint_tasks):
        G = build_transition_graph(disjoint_tasks)
        n = auto_detect_n_clusters(G)
        assert n == 2, f"Expected 2 components, got {n}"

    def test_empty_graph_returns_two(self):
        G = build_transition_graph([])
        assert auto_detect_n_clusters(G) == 2

    def test_fully_connected_graph_returns_at_least_two(self, make_event):
        events = [
            make_event(app_name="A", window_title=f"w{i}", event_type="click") for i in range(5)
        ]
        G = build_transition_graph(events)
        n = auto_detect_n_clusters(G)
        assert n >= 2


class TestClusterMacroTasks:
    def test_empty_graph_returns_empty_dict(self):
        G = build_transition_graph([])
        assert cluster_macro_tasks(G) == {}

    def test_single_node_graph_produces_one_cluster(self, make_event):
        G = build_transition_graph([make_event()])
        mapping = cluster_macro_tasks(G, n_clusters=2)
        assert len(mapping) == 1

    def test_disjoint_tasks_perfect_separation(self, disjoint_tasks):
        """Two completely disconnected task graphs should be perfectly
        separated by spectral clustering.

        Task A:  login → form → review → submit   (all under "ERP")
        Task B:  inbox → read → compose → sent     (all under "Mail")

        Because there are zero shared UI states between the two tasks,
        the adjacency matrix is block-diagonal and the Laplacian
        eigenvalues unambiguously separate the two components.

        Expected: NMI = 1.0
        """
        G = build_transition_graph(disjoint_tasks)
        state_to_cluster = cluster_macro_tasks(G, n_clusters=2)

        # Build ground-truth labels
        ground_truth = {}
        for e in disjoint_tasks[:-1]:  # last event has no outgoing edge
            key = (e.app_name, e.window_title, e.event_type)
            # "ERP" → cluster 0, "Mail" → cluster 1
            ground_truth[key] = 0 if e.app_name == "ERP" else 1

        metrics = evaluate_partition(G, state_to_cluster, ground_truth)
        assert metrics["nmi"] == pytest.approx(1.0, abs=1e-6), (
            f"NMI should be 1.0 for disjoint tasks, got {metrics['nmi']}"
        )

    def test_silhouette_score_in_valid_range(self, disjoint_tasks):
        G = build_transition_graph(disjoint_tasks)
        state_to_cluster = cluster_macro_tasks(G, n_clusters=2)
        metrics = evaluate_partition(G, state_to_cluster)
        assert -1.0 <= metrics["silhouette"] <= 1.0

    def test_three_disjoint_tasks_perfect_separation(self, make_event):
        """Three completely separate task graphs."""
        task_a = [
            make_event(app_name="ERP", window_title=f"step{i}", event_type="click")
            for i in range(3)
        ]
        task_b = [
            make_event(app_name="Mail", window_title=f"step{i}", event_type="click")
            for i in range(3)
        ]
        task_c = [
            make_event(app_name="Slack", window_title=f"step{i}", event_type="click")
            for i in range(3)
        ]
        events = task_a + task_b + task_c
        G = build_transition_graph(events)
        state_to_cluster = cluster_macro_tasks(G, n_clusters=3)

        ground_truth = {}
        for e in events[:-1]:
            key = (e.app_name, e.window_title, e.event_type)
            label = {"ERP": 0, "Mail": 1, "Slack": 2}[e.app_name]
            ground_truth[key] = label

        metrics = evaluate_partition(G, state_to_cluster, ground_truth)
        assert metrics["nmi"] == pytest.approx(1.0, abs=1e-6), (
            f"NMI should be 1.0 for three disjoint tasks, got {metrics['nmi']}"
        )

    def test_seed_ensures_reproducibility(self, disjoint_tasks):
        """Same random_state should produce identical cluster assignments."""
        G = build_transition_graph(disjoint_tasks)
        result_a = cluster_macro_tasks(G, n_clusters=2, random_state=42)
        result_b = cluster_macro_tasks(G, n_clusters=2, random_state=42)
        assert result_a == result_b
