"""Dataset utilities for the EGKD reproducible experiment.

This module provides the stable data-loading interface used by
``egkd_main_reproducible.py``:

    load_data(...)
    get_sir_path(...)
    load_sir_scores(...)

The generated centrality feature matrix keeps the historical 13-column order
used by the original scripts:

    0 degree centrality
    1 approximate betweenness centrality
    2 closeness centrality
    3 eigenvector centrality
    4 PageRank
    5 clustering coefficient
    6 average neighbor degree
    7 k-shell / core number
    8 H-index
    9 LDP-min
    10 LDP-max
    11 LDP-mean
    12 LDP-std

Keeping this order is important because the EGKD main script selects a subset
of these columns by index.
"""

from __future__ import annotations

import os
import pickle
from typing import Dict, Iterable, List, Tuple

import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix


FEATURE_NAMES = [
    "degree",
    "betweenness",
    "closeness",
    "eigenvector",
    "pagerank",
    "clustering",
    "avg_neighbor_degree",
    "kshell",
    "hindex",
    "ldp_min",
    "ldp_max",
    "ldp_mean",
    "ldp_std",
]


def _module_dir() -> str:
    """Return the directory containing this file."""
    return os.path.dirname(os.path.abspath(__file__))


def _safe_minmax(values: np.ndarray) -> np.ndarray:
    """Min-max normalize a vector; return zeros for constant vectors."""
    values = np.asarray(values, dtype=np.float64)
    v_min = float(np.min(values))
    v_max = float(np.max(values))
    if v_max - v_min < 1e-12:
        return np.zeros_like(values, dtype=np.float64)
    return (values - v_min) / (v_max - v_min)


def _read_edge_list(file_path: str) -> Tuple[nx.Graph, List[str]]:
    """Read an undirected simple graph from an edge-list file.

    Supported formats:
    - first line may be ``num_nodes num_edges``;
    - subsequent lines contain at least ``u v``;
    - extra columns are ignored.

    Node identifiers are kept as strings to stay compatible with cached SIR
    label dictionaries from earlier scripts.
    """
    graph = nx.Graph()
    with open(file_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    if not lines:
        raise ValueError(f"Empty dataset file: {file_path}")

    first = lines[0].split()
    data_lines = lines[1:] if len(first) == 2 and all(x.lstrip("-").isdigit() for x in first) else lines

    for line in data_lines:
        parts = line.split()
        if len(parts) >= 2:
            graph.add_edge(str(parts[0]), str(parts[1]))

    graph.remove_edges_from(nx.selfloop_edges(graph))
    nodes = list(graph.nodes())
    return graph, nodes


def _load_cached_features(cache_path: str):
    """Load cached adjacency/features/nodes if the cache is complete."""
    if not os.path.exists(cache_path):
        return None
    try:
        with open(cache_path, "rb") as f:
            data = pickle.load(f)
    except Exception as exc:
        print(f"[utils_data] Feature cache is unreadable and will be regenerated: {exc}")
        return None

    if not isinstance(data, dict):
        return None
    if not {"adj", "features", "nodes"}.issubset(data.keys()):
        return None
    if data["adj"] is None or data["features"] is None or data["nodes"] is None:
        return None
    return data["adj"], data["features"], data["nodes"]


def _save_cached_features(cache_path: str, adj: np.ndarray, features: np.ndarray, nodes: List[str]) -> None:
    """Persist raw, unnormalized centrality features for reuse."""
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    payload = {
        "adj": adj,
        "features": features.copy(),
        "nodes": list(nodes),
        "feature_names": FEATURE_NAMES,
    }
    with open(cache_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


def calculate_hindex_for_graph(graph: nx.Graph, nodes: Iterable[str]) -> np.ndarray:
    """Compute the H-index centrality for each node in ``nodes``.

    A node has H-index h if it has at least h neighbors whose degrees are at
    least h. This captures local influence supported by influential neighbors.
    """
    degrees = dict(graph.degree())
    values = []
    for node in nodes:
        neighbor_degrees = sorted((degrees[nbr] for nbr in graph.neighbors(node)), reverse=True)
        h = 0
        for idx, degree in enumerate(neighbor_degrees, start=1):
            if degree >= idx:
                h = idx
            else:
                break
        values.append(h)
    return np.asarray(values, dtype=np.float64)


def calculate_ldp_features(graph: nx.Graph, nodes: Iterable[str]) -> np.ndarray:
    """Compute Local Degree Profile features for each node.

    LDP summarizes the degree distribution of one-hop neighbors using
    min/max/mean/std and is used by the original EGKD feature matrix.
    """
    degrees = dict(graph.degree())
    rows = []
    for node in nodes:
        neighbor_degrees = [degrees[nbr] for nbr in graph.neighbors(node)]
        if not neighbor_degrees:
            rows.append([0.0, 0.0, 0.0, 0.0])
        else:
            rows.append([
                float(np.min(neighbor_degrees)),
                float(np.max(neighbor_degrees)),
                float(np.mean(neighbor_degrees)),
                float(np.std(neighbor_degrees)) if len(neighbor_degrees) > 1 else 0.0,
            ])
    return np.asarray(rows, dtype=np.float64)


def compute_centrality_features(graph: nx.Graph, nodes: List[str]) -> np.ndarray:
    """Compute the 13 topological features used by EGKD experiments."""
    n_nodes = len(nodes)
    if n_nodes == 0:
        raise ValueError("Cannot compute features for an empty graph.")

    print(f"[utils_data] Computing centrality features for {n_nodes} nodes and {graph.number_of_edges()} edges.")

    degree_centrality = nx.degree_centrality(graph)
    degree = np.asarray([degree_centrality[node] for node in nodes], dtype=np.float64)

    # Approximate betweenness keeps feature generation tractable on larger graphs.
    if n_nodes > 1000:
        k = max(1, int(n_nodes * 0.1))
        betweenness_dict = nx.betweenness_centrality(graph, k=k, endpoints=False, seed=42)
    else:
        betweenness_dict = nx.betweenness_centrality(graph, endpoints=False)
    betweenness = np.asarray([betweenness_dict[node] for node in nodes], dtype=np.float64)

    closeness_dict = nx.closeness_centrality(graph)
    closeness = np.asarray([closeness_dict[node] for node in nodes], dtype=np.float64)

    try:
        eigenvector_dict = nx.eigenvector_centrality(graph, max_iter=1000)
        eigenvector = np.asarray([eigenvector_dict[node] for node in nodes], dtype=np.float64)
    except nx.PowerIterationFailedConvergence:
        eigenvector = np.zeros(n_nodes, dtype=np.float64)

    pagerank_dict = nx.pagerank(graph)
    pagerank = np.asarray([pagerank_dict[node] for node in nodes], dtype=np.float64)

    clustering_dict = nx.clustering(graph)
    clustering = np.asarray([clustering_dict[node] for node in nodes], dtype=np.float64)

    avg_neighbor_degree_dict = nx.average_neighbor_degree(graph)
    avg_neighbor_degree = np.asarray([avg_neighbor_degree_dict[node] for node in nodes], dtype=np.float64)

    kshell_dict = nx.core_number(graph)
    kshell = np.asarray([kshell_dict[node] for node in nodes], dtype=np.float64)

    hindex = calculate_hindex_for_graph(graph, nodes)
    ldp = calculate_ldp_features(graph, nodes)

    features = np.column_stack([
        degree,
        betweenness,
        closeness,
        eigenvector,
        pagerank,
        clustering,
        avg_neighbor_degree,
        kshell,
        hindex,
        ldp[:, 0],
        ldp[:, 1],
        ldp[:, 2],
        ldp[:, 3],
    ])
    return features


def _postprocess_features(features: np.ndarray, log_transform: bool, normalize_features: bool) -> np.ndarray:
    """Apply the same feature preprocessing used by the historical scripts."""
    features = np.asarray(features, dtype=np.float64)
    if log_transform:
        features = np.log1p(np.maximum(features, 0.0))
    if normalize_features:
        from sklearn.preprocessing import StandardScaler

        features = StandardScaler().fit_transform(features)
    return features


def load_data(file_path: str, log_transform: bool = True, normalize_features: bool = True):
    """Load a graph and its topological node feature matrix.

    Parameters
    ----------
    file_path:
        Path to the dataset edge-list file.
    log_transform:
        Whether to apply ``log1p`` to non-negative raw features.
    normalize_features:
        Whether to standardize features using ``StandardScaler``.

    Returns
    -------
    adj:
        Dense numpy adjacency matrix, ordered by ``nodes``.
    features:
        Node feature matrix with the 13-column order documented above.
    nodes:
        Node identifiers corresponding to matrix rows.
    """
    dataset_name = os.path.splitext(os.path.basename(file_path))[0]
    cache_path = os.path.join(_module_dir(), "data", f"{dataset_name}_centrality_features.pkl")

    cached = _load_cached_features(cache_path)
    if cached is not None:
        adj, raw_features, nodes = cached
        print(f"[utils_data] Loaded cached features: {cache_path}, shape={np.asarray(raw_features).shape}")
    else:
        graph, nodes = _read_edge_list(file_path)
        adj = nx.adjacency_matrix(graph, nodelist=nodes).toarray()
        raw_features = compute_centrality_features(graph, nodes)
        _save_cached_features(cache_path, adj, raw_features, nodes)
        print(f"[utils_data] Saved feature cache: {cache_path}")

    features = _postprocess_features(raw_features, log_transform, normalize_features)
    return adj, features, nodes


def load_sir_scores(sir_path: str) -> Dict:
    """Load precomputed SIR influence labels from a pickle file."""
    try:
        with open(sir_path, "rb") as f:
            return pickle.load(f)
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Failed to load the SIR label file. It may have been saved with a "
            "different numpy/Python environment. Regenerate the SIR labels with "
            "the current environment if this error persists.\n"
            f"File: {sir_path}"
        ) from exc


def get_sir_path(dataset_name: str, propagation_threshold_multiple: float) -> str:
    """Return the canonical local SIR-label path for a dataset and beta multiplier."""
    beta = float(propagation_threshold_multiple)
    beta_str = str(int(beta)) if beta.is_integer() else str(propagation_threshold_multiple)
    sir_dir = os.path.join(_module_dir(), "SIR", f"{dataset_name}_SIR分数")
    return os.path.join(sir_dir, f"{dataset_name}_beta_{beta_str}_sir_scores.pkl")


def run_agm(adj, max_terms: int = 10) -> np.ndarray:
    """Compute a lightweight AGM-style proxy from average shortest-path distance.

    This function is retained for compatibility with older pseudo-label scripts.
    It is not used by ``egkd_main_reproducible.py``.
    """
    del max_terms  # kept for signature compatibility
    from scipy.sparse.csgraph import floyd_warshall

    dist_matrix, _ = floyd_warshall(csr_matrix(adj), return_predecessors=True)
    values = np.zeros(adj.shape[0], dtype=np.float64)
    for i in range(adj.shape[0]):
        dist_i = dist_matrix[i]
        finite = dist_i[(dist_i > 0) & np.isfinite(dist_i)]
        values[i] = 1.0 / (1.0 + np.mean(finite)) if finite.size else 0.0
    return values


def calculate_hindex_centrality(adj) -> np.ndarray:
    """Compute min-max normalized H-index centrality from an adjacency matrix."""
    adj_csr = csr_matrix(adj)
    degrees = np.asarray(adj_csr.sum(axis=1)).ravel()
    values = np.zeros(adj_csr.shape[0], dtype=np.float64)
    for node in range(adj_csr.shape[0]):
        neighbor_degrees = np.sort(degrees[adj_csr[node].indices])[::-1]
        h = 0
        for idx, degree in enumerate(neighbor_degrees, start=1):
            if degree >= idx:
                h = idx
            else:
                break
        values[node] = h
    return _safe_minmax(values)


def _graph_from_adj(adj) -> nx.Graph:
    """Build a NetworkX graph from a dense or sparse adjacency matrix."""
    adj_csr = csr_matrix(adj)
    try:
        return nx.from_scipy_sparse_array(adj_csr)
    except AttributeError:
        return nx.from_scipy_sparse_matrix(adj_csr)


def generate_pseudo_labels(adj, method: str = "degree", dataset_name: str | None = None,
                           max_terms: int = 10, nodes=None) -> np.ndarray:
    """Generate normalized structural pseudo-labels for legacy experiments.

    The cleaned EGKD main script builds its pseudo-labels directly from selected
    feature columns. This helper is retained so older ablation scripts importing
    ``utils_data`` remain usable.
    """
    del nodes  # legacy argument; no longer required
    adj_csr = csr_matrix(adj)

    if method == "degree":
        return _safe_minmax(np.asarray(adj_csr.sum(axis=1)).ravel())
    if method == "agm":
        cache_path = None
        values = None
        if dataset_name is not None:
            cache_path = os.path.join(_module_dir(), "data", f"{dataset_name}_agm_values.pkl")
            if os.path.exists(cache_path):
                with open(cache_path, "rb") as f:
                    values = pickle.load(f)
        if values is None or len(values) != adj_csr.shape[0]:
            values = run_agm(adj_csr, max_terms=max_terms)
            if cache_path is not None:
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                with open(cache_path, "wb") as f:
                    pickle.dump(values, f, protocol=pickle.HIGHEST_PROTOCOL)
        return _safe_minmax(np.asarray(values))
    if method == "hindex":
        return calculate_hindex_centrality(adj_csr)

    graph = _graph_from_adj(adj_csr)
    if method == "betweenness":
        return _safe_minmax(np.asarray(list(nx.betweenness_centrality(graph).values())))
    if method == "closeness":
        return _safe_minmax(np.asarray(list(nx.closeness_centrality(graph).values())))
    if method == "pr":
        return _safe_minmax(np.asarray(list(nx.pagerank(graph).values())))
    if method == "mixed":
        degree = _safe_minmax(np.asarray(list(nx.degree_centrality(graph).values())))
        pagerank = _safe_minmax(np.asarray(list(nx.pagerank(graph).values())))
        return _safe_minmax(0.7 * degree + 0.3 * pagerank)

    raise ValueError(f"Invalid pseudo label method: {method}")


def normalize_adjacency(adj: np.ndarray) -> np.ndarray:
    """Return symmetric GCN-style normalized adjacency with self-loops."""
    adj = np.asarray(adj, dtype=np.float64)
    adj_hat = adj + np.eye(adj.shape[0], dtype=np.float64)
    degree = np.sum(adj_hat, axis=1)
    degree_inv_sqrt = np.power(degree, -0.5)
    degree_inv_sqrt[np.isinf(degree_inv_sqrt)] = 0.0
    return degree_inv_sqrt[:, None] * adj_hat * degree_inv_sqrt[None, :]


def preprocess_data(adj: np.ndarray, features: np.ndarray):
    """Normalize adjacency and standardize features for legacy scripts."""
    from sklearn.preprocessing import StandardScaler

    return normalize_adjacency(adj), StandardScaler().fit_transform(features)
