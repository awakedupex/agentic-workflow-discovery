"""Clustering quality metrics — Silhouette score and Normalized Mutual
Information (NMI) when ground-truth task labels are available.

All metrics are computed against the *undirected* adjacency matrix to
match the symmetric Laplacian used by SpectralClustering.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
from sklearn.metrics import normalized_mutual_info_score, silhouette_score


def evaluate_partition(
    G: nx.DiGraph,
    state_to_cluster: dict[tuple, int],
    ground_truth: dict[tuple, int] | None = None,
) -> dict[str, float]:
    """Compute partition-quality metrics.

    Parameters
    ----------
    G : nx.DiGraph
        The transition graph.
    state_to_cluster : dict[tuple, int]
        Output of cluster_macro_tasks.
    ground_truth : dict[tuple, int], optional
        True task labels for each state.  Only needed for NMI.

    Returns
    -------
    dict[str, float]
        "silhouette" : float  [-1, 1] — cohesion vs. separation.
        "n_clusters" : float  (int cast to float for serialization).
        "nmi"        : float  [0, 1] — only if ground_truth provided.
    """
    if not state_to_cluster:
        return {"silhouette": float("nan"), "n_clusters": 0.0}

    adjacency = nx.to_scipy_sparse_array(G, weight="weight", format="csr")
    adjacency = adjacency.astype(np.float64)
    if adjacency.indices.dtype == np.int64 or adjacency.indptr.dtype == np.int64:
        adjacency.indices = adjacency.indices.astype(np.int32)
        adjacency.indptr = adjacency.indptr.astype(np.int32)
    labels = np.array(list(state_to_cluster.values()))

    result: dict[str, float] = {
        "silhouette": float(silhouette_score(adjacency, labels, metric="precomputed")),
        "n_clusters": float(len(set(labels))),
    }

    if ground_truth is not None:
        true_labels = np.array([ground_truth.get(node, -1) for node in G.nodes()])
        mask = true_labels != -1
        if mask.sum() > 0:
            result["nmi"] = float(normalized_mutual_info_score(true_labels[mask], labels[mask]))

    return result
