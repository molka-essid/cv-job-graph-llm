# cv-job-graph-llm

> Bipartite CV–Job Graph with LLM Link Prediction & Node Classification

A full graph-based recruitment intelligence pipeline — models the CV–Job matching problem as a bipartite graph, enriches it with LLM semantic embeddings (Google Gemini), predicts missing links using both LLM and classical algorithms, and classifies nodes using ML models and custom GNNs.

Built with an interactive Streamlit interface across 7 tabs.

---

## Features

- **Bipartite graph construction** — 40 CV nodes + 25 Job nodes with rich attributes
- **Structural analysis** — degree, density, betweenness/closeness/degree centrality
- **Community detection** — Louvain (greedy modularity) + Label Propagation at 3 levels
- **LLM link prediction** — Google Gemini `text-embedding-004` (768-dim → PCA 32) with TF-IDF fallback
- **Classical link prediction** — 5 bipartite-aware algorithms (Common Neighbors, Adamic-Adar, Jaccard, Preferential Attachment, Katz)
- **Node classification** — Random Forest, Gradient Boosting, SVM, Label Propagation (semi-supervised)
- **GNN** — GCN (Kipf & Welling) + GraphSAGE implemented in pure NumPy (no PyTorch required)
- **Graph enrichment analysis** — measures the impact of LLM-predicted links on classification accuracy

---

## Project Structure

```
cv-job-graph-llm/
├── app.py                  # Streamlit interface (7 tabs)
├── data.py                 # Synthetic CV & Job dataset generator
├── graph_analysis.py       # Bipartite graph construction & structural analysis
├── embeddings.py           # Gemini LLM embeddings + community detection
├── classification.py       # ML node classification (RF, GB, SVM, LP)
├── gnn_classification.py   # GCN + GraphSAGE (pure NumPy)
├── requirements.txt
└── .env                    # API keys (not committed)
```

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Gemini API (optional)

Create a `.env` file:

```env
GEMINI_API_KEY=your_api_key_here
```

> Without a key, the pipeline automatically falls back to TF-IDF + PCA embeddings. Everything still works.

### 3. Run the app

```bash
streamlit run app.py
```

---

## Streamlit Interface

| Tab | Content |
|-----|---------|
| 1 — Data | Synthetic dataset stats and node profiles |
| 2 — Graph | Bipartite graph visualization (spring layout) |
| 3 — Analysis | Structural metrics and centrality measures |
| 4 — Communities | Louvain + Label Propagation, modularity scores |
| 5 — Link Prediction | LLM vs classical algorithms comparison |
| 6 — Classification | RF / GB / SVM / GCN results |
| 7 — Enrichment | Before/after classification with enriched graph G' |

---

## Link Prediction — Bipartite-Aware Design

Standard link prediction algorithms (Common Neighbors, Adamic-Adar, etc.) **do not apply directly** to bipartite graphs — CV and Job nodes can never share direct common neighbors. This project handles this correctly by working through graph projections:

- **CV-projection** — two CVs are connected if they share ≥1 Job neighbor
- **Job-projection** — two Jobs are connected if they share ≥1 CV neighbor

Scores are then computed through these projections, giving meaningful non-zero values for all CV–Job pairs.

---

## GNN Implementation

Both GCN and GraphSAGE are implemented from scratch in NumPy — no PyTorch or PyG dependency.

**GCN propagation rule:**
```
H(l+1) = ReLU( D^{-1/2} A D^{-1/2} · H(l) · W(l) )
```

- 2-layer architecture with Xavier initialization
- SGD training, 300 epochs, lr=0.01
- Cross-entropy loss on labeled nodes only

**GraphSAGE** uses mean-aggregation over 1-hop neighbors.

---

## Node Features

Each node is represented by a concatenation of:

- **Structural** (7): degree, degree centrality, betweenness, closeness, years of experience, number of skills, polyvalence flag
- **Semantic** (32): Gemini PCA embeddings or TF-IDF PCA fallback
- **Community** (1): Louvain community ID

---

## Requirements

```
streamlit
networkx
numpy
pandas
scikit-learn
matplotlib
python-dotenv
google-generativeai   # optional, for real LLM embeddings
```

---

## Notes

- The dataset is **fully synthetic** — no real personal data is used
- All random seeds are fixed (`seed=42`) for reproducibility
- The pipeline runs end-to-end without any API key
