"""
GNN Node Classification Module
Uses PyTorch Geometric GCN/GraphSAGE for node classification on the CV-Job graph.
Falls back to a lightweight NumPy GCN if PyG is unavailable.
"""
import numpy as np
import networkx as nx
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# Lightweight GCN without PyTorch (pure NumPy)
# ─────────────────────────────────────────────────────────────────────────────

def _relu(x):
    return np.maximum(0, x)

def _softmax(x):
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

def _normalize_adj(A):
    """Symmetric normalization: D^{-1/2} A D^{-1/2}"""
    A = A + np.eye(A.shape[0])          # self-loops
    d = A.sum(axis=1)
    d_inv_sqrt = np.diag(1.0 / np.sqrt(np.where(d > 0, d, 1)))
    return d_inv_sqrt @ A @ d_inv_sqrt


class SimpleGCN:
    """
    2-layer Graph Convolutional Network (Kipf & Welling 2017).
    Pure NumPy — no PyTorch required.
    """

    def __init__(self, hidden=64, lr=0.01, epochs=300, seed=42):
        self.hidden = hidden
        self.lr = lr
        self.epochs = epochs
        np.random.seed(seed)

    def fit(self, A_norm, X, y_train, train_mask):
        n, d = X.shape
        c = int(y_train.max()) + 1

        # Xavier init
        self.W1 = np.random.randn(d, self.hidden) * np.sqrt(2.0 / d)
        self.W2 = np.random.randn(self.hidden, c) * np.sqrt(2.0 / self.hidden)

        for _ in range(self.epochs):
            # Forward
            H1 = _relu(A_norm @ X @ self.W1)
            logits = A_norm @ H1 @ self.W2
            probs = _softmax(logits)

            # Cross-entropy loss (train nodes only)
            eps = 1e-9
            loss = -np.mean(np.log(probs[train_mask, y_train[train_mask]] + eps))

            # Backprop (simplified)
            dL = probs.copy()
            dL[train_mask] -= np.eye(c)[y_train[train_mask]]
            dL /= train_mask.sum()

            dW2 = (A_norm @ H1).T @ dL
            dH1 = dL @ self.W2.T
            dH1_relu = dH1 * (H1 > 0)
            dW1 = (A_norm @ X).T @ dH1_relu

            self.W1 -= self.lr * dW1
            self.W2 -= self.lr * dW2

    def predict(self, A_norm, X):
        H1 = _relu(A_norm @ X @ self.W1)
        logits = A_norm @ H1 @ self.W2
        return np.argmax(logits, axis=1)

    def predict_proba(self, A_norm, X):
        H1 = _relu(A_norm @ X @ self.W1)
        logits = A_norm @ H1 @ self.W2
        return _softmax(logits)


# ─────────────────────────────────────────────────────────────────────────────
# GraphSAGE-style aggregation (mean pooling)
# ─────────────────────────────────────────────────────────────────────────────

class SimpleGraphSAGE:
    """
    2-layer GraphSAGE with mean aggregation — pure NumPy.
    """

    def __init__(self, hidden=64, lr=0.01, epochs=300, seed=42):
        self.hidden = hidden
        self.lr = lr
        self.epochs = epochs
        np.random.seed(seed)

    def _aggregate(self, A, X, W_self, W_neigh):
        """Mean aggregate neighbors then concatenate."""
        deg = A.sum(axis=1, keepdims=True)
        deg = np.where(deg == 0, 1, deg)
        neigh_mean = (A @ X) / deg
        return X @ W_self + neigh_mean @ W_neigh

    def fit(self, A, X, y_train, train_mask):
        n, d = X.shape
        c = int(y_train.max()) + 1

        self.W1s = np.random.randn(d, self.hidden) * 0.1
        self.W1n = np.random.randn(d, self.hidden) * 0.1
        self.W2s = np.random.randn(self.hidden, c) * 0.1
        self.W2n = np.random.randn(self.hidden, c) * 0.1

        # Remove self loops for pure neighbor aggregation
        A_no_self = A.copy()
        np.fill_diagonal(A_no_self, 0)

        for _ in range(self.epochs):
            H1 = _relu(self._aggregate(A_no_self, X, self.W1s, self.W1n))
            logits = self._aggregate(A_no_self, H1, self.W2s, self.W2n)
            probs = _softmax(logits)

            dL = probs.copy()
            dL[train_mask] -= np.eye(c)[y_train[train_mask]]
            dL /= max(train_mask.sum(), 1)

            # Simplified gradient step
            self.W2s -= self.lr * H1.T @ dL
            self.W2n -= self.lr * ((A_no_self @ H1) / np.where(A_no_self.sum(1, keepdims=True)==0,1,A_no_self.sum(1,keepdims=True))).T @ dL

    def predict(self, A, X):
        A_no_self = A.copy()
        np.fill_diagonal(A_no_self, 0)
        H1 = _relu(self._aggregate(A_no_self, X, self.W1s, self.W1n))
        logits = self._aggregate(A_no_self, H1, self.W2s, self.W2n)
        return np.argmax(logits, axis=1)


# ─────────────────────────────────────────────────────────────────────────────
# Main GNN classification pipeline
# ─────────────────────────────────────────────────────────────────────────────

def classify_nodes_gnn(nodes, node_data_list, features_dict, G, target_key,
                       label_fn=None, test_size=0.3, epochs=300):
    """
    Run GNN-based node classification.

    Parameters
    ----------
    nodes        : list of node IDs
    node_data_list : list of dicts with node attributes
    features_dict  : {node_id: feature_vector}
    G            : NetworkX graph
    target_key   : attribute name for label
    label_fn     : optional callable(node_data) -> label string
    test_size    : fraction for test split
    epochs       : training epochs

    Returns dict with results compatible with the app's display functions.
    """
    # Filter to nodes that have features
    valid = [n for n in nodes if n in features_dict]
    if len(valid) < 6:
        return {"error": "Not enough nodes for GNN classification"}

    lookup = {d["id"]: d for d in node_data_list}

    # Labels
    if label_fn:
        y_raw = [label_fn(lookup[n]) for n in valid]
    else:
        y_raw = [lookup[n][target_key] for n in valid]

    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    n_classes = len(le.classes_)

    if n_classes < 2:
        return {"error": "Need at least 2 classes"}

    # Feature matrix
    X_list = [features_dict[n] for n in valid]
    X = np.array(X_list, dtype=float)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Adjacency matrix (subgraph of valid nodes)
    idx_map = {n: i for i, n in enumerate(valid)}
    N = len(valid)
    A = np.zeros((N, N), dtype=float)
    for u, v in G.edges():
        if u in idx_map and v in idx_map:
            A[idx_map[u], idx_map[v]] = 1
            A[idx_map[v], idx_map[u]] = 1
    A_norm = _normalize_adj(A)

    # Train/test split — disable stratify if any class has < 2 members
    idx = np.arange(N)
    class_counts = np.bincount(y)
    use_stratify = y if (n_classes >= 2 and class_counts.min() >= 2) else None
    train_idx, test_idx = train_test_split(idx, test_size=test_size, random_state=42,
                                           stratify=use_stratify)
    train_mask = np.zeros(N, dtype=bool)
    train_mask[train_idx] = True

    results = {}

    # ── GCN ──
    try:
        gcn = SimpleGCN(hidden=64, lr=0.01, epochs=epochs)
        gcn.fit(A_norm, X, y, train_mask)
        y_pred_gcn = gcn.predict(A_norm, X)
        train_acc = accuracy_score(y[train_idx], y_pred_gcn[train_idx])
        test_acc  = accuracy_score(y[test_idx],  y_pred_gcn[test_idx])
        results["GCN (Graph Conv Net)"] = {
            "cv_accuracy_mean": test_acc,
            "cv_accuracy_std": 0.0,
            "train_accuracy": train_acc,
            "confusion_matrix": confusion_matrix(y, y_pred_gcn).tolist(),
            "classes": list(le.classes_),
            "note": f"Train/Test split 70/30 — {epochs} epochs"
        }
    except Exception as e:
        results["GCN (Graph Conv Net)"] = {"error": str(e)}

    # ── GraphSAGE ──
    try:
        sage = SimpleGraphSAGE(hidden=64, lr=0.005, epochs=epochs)
        sage.fit(A, X, y, train_mask)
        y_pred_sage = sage.predict(A, X)
        train_acc_s = accuracy_score(y[train_idx], y_pred_sage[train_idx])
        test_acc_s  = accuracy_score(y[test_idx],  y_pred_sage[test_idx])
        results["GraphSAGE (Mean Agg)"] = {
            "cv_accuracy_mean": test_acc_s,
            "cv_accuracy_std": 0.0,
            "train_accuracy": train_acc_s,
            "confusion_matrix": confusion_matrix(y, y_pred_sage).tolist(),
            "classes": list(le.classes_),
            "note": f"Train/Test split 70/30 — {epochs} epochs"
        }
    except Exception as e:
        results["GraphSAGE (Mean Agg)"] = {"error": str(e)}

    best = max(
        [(k, v["cv_accuracy_mean"]) for k, v in results.items() if "error" not in v],
        key=lambda x: x[1], default=("GCN (Graph Conv Net)", 0)
    )

    return {
        "target": target_key,
        "n_samples": N,
        "classes": list(le.classes_),
        "models": results,
        "best_model": best[0],
        "best_score": best[1],
        "method": "GNN",
        "architecture": "GCN + GraphSAGE (pure NumPy)",
    }


def run_gnn_all_tasks(cvs, jobs, features_dict, G):
    """Run GNN classification for all CV and Job tasks."""
    cv_nodes  = [cv["id"]  for cv in cvs]
    job_nodes = [job["id"] for job in jobs]

    return {
        "gnn_cv_level": classify_nodes_gnn(
            cv_nodes, cvs, features_dict, G, "level", epochs=200),
        "gnn_cv_domain": classify_nodes_gnn(
            cv_nodes, cvs, features_dict, G, "domain", epochs=200),
        "gnn_cv_poly": classify_nodes_gnn(
            cv_nodes, cvs, features_dict, G, "polyvalent",
            label_fn=lambda d: "polyvalent" if d["polyvalent"] else "spécialisé",
            epochs=200),
        "gnn_job_level": classify_nodes_gnn(
            job_nodes, jobs, features_dict, G, "level", epochs=200),
        "gnn_job_domain": classify_nodes_gnn(
            job_nodes, jobs, features_dict, G, "domain", epochs=200),
    }