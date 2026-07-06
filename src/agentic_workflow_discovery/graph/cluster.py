"""Partition UI states into macro-task clusters via spectral clustering.

Why spectral clustering?
------------------------
1. No spherical assumption — K-Means directly on state coordinates would
   fail on chain-like UI workflows (A → B → C → D).
2. The graph Laplacian's eigenstructure naturally reveals weakly connected
   communities — exactly what "macro-tasks" are.
3. sklearn's SpectralClustering wraps ARPACK (Lanczos), which is
   O(n² · k) and fast for n < 10 000 nodes.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
from sklearn.cluster import SpectralClustering


def auto_detect_n_clusters(G: nx.DiGraph) -> int:
    """Estimate the number of macro-tasks from the graph structure.

    Strategy: count the weakly connected components of the directed graph.
    This is a *conservative* lower bound — real macro-tasks often
    outnumber components because different tasks can share UI states
    (e.g., the OS taskbar appears in every task).

    Returns at least 2 (spectral clustering requires n_clusters >= 2).
    """
    n = nx.number_weakly_connected_components(G)
    return max(2, n)


def cluster_macro_tasks(
    G: nx.DiGraph,
    n_clusters: int | None = None,
    random_state: int = 42,
) -> dict[tuple, int]:
    """Partition UI state nodes into macro-task clusters.

    Parameters
    ----------
    G : nx.DiGraph
        Transition graph from build_transition_graph.
    n_clusters : int, optional
        Number of clusters.  Auto-detected from connected components
        if None.
    random_state : int
        Seed for reproducibility (SpectralClustering + K-Means is
        non-deterministic).

    Returns
    -------
    dict[tuple, int]
        Mapping from state key → cluster ID (0-indexed).
    """
    if G.number_of_nodes() == 0:
        return {}

    # SpectralClustering requires at least 2 nodes and 2 <= n_clusters.
    # Degenerate cases return every node in its own trivial cluster.
    n_nodes = G.number_of_nodes()
    if n_nodes < 2:
        return {node: 0 for node in G.nodes()}

    # Convert to undirected sparse adjacency matrix — the Laplacian
    # is only defined for symmetric matrices.
    # SpectralClustering's 'precomputed' affinity expects a symmetric
    # matrix; we symmetrize explicitly: W ← (W + Wᵀ) / 2.
    # This encodes "transition in either direction" as similarity.
    adjacency = nx.to_scipy_sparse_array(G, weight="weight", format="csr")
    adjacency = (adjacency + adjacency.T) / 2.0

    # scikit-learn >= 1.9 requires 32-bit sparse indices.
    adjacency = adjacency.astype(np.float64)
    if adjacency.indices.dtype == np.int64 or adjacency.indptr.dtype == np.int64:
        adjacency.indices = adjacency.indices.astype(np.int32)
        adjacency.indptr = adjacency.indptr.astype(np.int32)

    if n_clusters is None:
        n_clusters = auto_detect_n_clusters(G)

    # Guard: n_clusters must be in [2, n_nodes].
    n_clusters = max(2, min(n_clusters, n_nodes))

    # sklearn's SpectralClustering with 'precomputed' affinity uses
    # our adjacency matrix directly as the similarity matrix.
    # assign_labels='kmeans' follows the Ng-Jordan-Weiss algorithm:
    #   1. Compute normalized Laplacian L_norm = D^{-1/2} L D^{-1/2}
    #   2. Eigendecompose → top k eigenvectors → matrix U
    #   3. Row-normalize U → matrix T
    #   4. K-Means on rows of T
    model = SpectralClustering(
        n_clusters=n_clusters,
        affinity="precomputed",
        assign_labels="kmeans",
        random_state=random_state,
        n_init=20,  # multiple restarts for stability
    )

    labels = model.fit_predict(adjacency)

    return {node: int(label) for node, label in zip(G.nodes(), labels)}
