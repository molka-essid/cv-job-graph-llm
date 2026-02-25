"""
Graph Construction and Analysis Module
Builds bipartite CV-Job graph and computes structural properties.

Fix: structural link prediction now uses bipartite-aware approach —
     CV-Job pairs share "co-neighbors" via the other side of the graph,
     so we compute scores through the 2-hop neighbourhood correctly.
"""
import networkx as nx
import numpy as np
from collections import defaultdict


def build_bipartite_graph(cvs, jobs, edges):
    """Build a bipartite graph with CV and Job nodes."""
    G = nx.Graph()
    for cv in cvs:
        G.add_node(cv["id"],
                   node_type="CV",
                   level=cv["level"],
                   domain=cv["domain"],
                   skills=cv["skills"],
                   years_exp=cv["years_exp"],
                   diploma=cv["diploma"],
                   text=cv["text"],
                   polyvalent=cv["polyvalent"])
    for job in jobs:
        G.add_node(job["id"],
                   node_type="Job",
                   level=job["level"],
                   domain=job["domain"],
                   skills=job["skills_required"],
                   company_size=job["company_size"],
                   text=job["text"])
    for u, v, attr in edges:
        G.add_edge(u, v, **attr)
    return G


def compute_graph_properties(G):
    """Compute structural properties of the graph."""
    cv_nodes  = [n for n, d in G.nodes(data=True) if d.get("node_type") == "CV"]
    job_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "Job"]

    props = {
        "n_nodes":  G.number_of_nodes(),
        "n_cv":     len(cv_nodes),
        "n_jobs":   len(job_nodes),
        "n_edges":  G.number_of_edges(),
        "density":  nx.density(G),
        "avg_degree": np.mean([d for _, d in G.degree()]),
        "max_degree": max(d for _, d in G.degree()),
        "components": nx.number_connected_components(G),
        "largest_component_size": len(max(nx.connected_components(G), key=len)),
    }

    cv_degrees  = {n: G.degree(n) for n in cv_nodes}
    job_degrees = {n: G.degree(n) for n in job_nodes}
    props["avg_cv_degree"]  = np.mean(list(cv_degrees.values()))
    props["avg_job_degree"] = np.mean(list(job_degrees.values()))
    props["top_cv"]   = sorted(cv_degrees,  key=cv_degrees.get,  reverse=True)[:5]
    props["top_jobs"] = sorted(job_degrees, key=job_degrees.get, reverse=True)[:5]

    largest_cc = max(nx.connected_components(G), key=len)
    subG = G.subgraph(largest_cc)
    betweenness = nx.betweenness_centrality(subG, normalized=True)
    closeness   = nx.closeness_centrality(subG)
    degree_cent = nx.degree_centrality(subG)

    props["betweenness"]      = betweenness
    props["closeness"]        = closeness
    props["degree_centrality"]= degree_cent
    props["top_betweenness"]  = sorted(betweenness, key=betweenness.get, reverse=True)[:5]
    return props


def build_skill_graph(cv_data):
    skills = cv_data.get("skills", [])
    G_skills = nx.Graph()
    G_skills.add_nodes_from(skills)
    for i, s1 in enumerate(skills):
        for s2 in skills[i+1:]:
            G_skills.add_edge(s1, s2)
    return G_skills


def build_cv_skill_similarity_graph(cvs):
    G = nx.Graph()
    for cv in cvs:
        G.add_node(cv["id"], **{k: v for k, v in cv.items() if k != "text"})
    for i, cv1 in enumerate(cvs):
        for cv2 in cvs[i+1:]:
            s1 = set(cv1["skills"]); s2 = set(cv2["skills"])
            overlap = len(s1 & s2) / max(len(s1 | s2), 1)
            if overlap > 0.2:
                G.add_edge(cv1["id"], cv2["id"], weight=round(overlap, 3))
    return G


# ─────────────────────────────────────────────────────────────────────────────
# BIPARTITE-AWARE LINK PREDICTION
# ─────────────────────────────────────────────────────────────────────────────
# In a bipartite graph G = (CV ∪ Job, E), a CV and a Job NEVER share a direct
# common neighbor — they are on opposite sides.  The meaningful "co-neighbors"
# of a CV-Job pair (u, v) are the nodes reachable in exactly 2 hops:
#
#   • For CV u: its Job-neighbors  N_J(u)
#   • For Job v: its CV-neighbors  N_C(v)
#
# "Common neighbors" in the bipartite sense = |N_J(u) ∩ {v's other jobs}|
# but a more natural bipartite formulation uses the *projected* graph:
#
#   CV-projection  G_C : two CVs connected if they share ≥1 Job neighbor
#   Job-projection G_J : two Jobs connected if they share ≥1 CV neighbor
#
# For predicting link (CV_u, Job_v):
#   CN  = |{Jobs connected to u}  ∩  {Jobs connected to Jobs adj. to v}|
#        = number of Job-neighbors of u that are also Job-neighbors of v's CV-neighbors
#
# Simpler & correct formula used below:
#   shared_cv_neighbors(u, v) = |N(u) ∩ N(N(v))|   (2-hop, same side)
#   i.e. CVs that are connected to the same Jobs as v → proxy for similarity
#
# Practically we use:
#   score(CV_u, Job_v) based on the CV-projection neighborhood of u
#   and the Job-projection neighborhood of v.
# ─────────────────────────────────────────────────────────────────────────────

def _bipartite_projections(G, cv_nodes, job_nodes):
    """
    Build CV-projection and Job-projection from the bipartite graph.
    CV-projection: edge between two CVs if they share ≥1 Job.
    Job-projection: edge between two Jobs if they share ≥1 CV.
    Returns (G_cv_proj, G_job_proj).
    """
    cv_set  = set(cv_nodes)
    job_set = set(job_nodes)

    G_cv  = nx.Graph(); G_cv.add_nodes_from(cv_nodes)
    G_job = nx.Graph(); G_job.add_nodes_from(job_nodes)

    # For each Job, connect all CVs that share it
    for job in job_nodes:
        neighbors_cv = [n for n in G.neighbors(job) if n in cv_set]
        for i in range(len(neighbors_cv)):
            for j in range(i+1, len(neighbors_cv)):
                u, v = neighbors_cv[i], neighbors_cv[j]
                if G_cv.has_edge(u, v):
                    G_cv[u][v]["weight"] += 1
                else:
                    G_cv.add_edge(u, v, weight=1)

    # For each CV, connect all Jobs that share it
    for cv in cv_nodes:
        neighbors_job = [n for n in G.neighbors(cv) if n in job_set]
        for i in range(len(neighbors_job)):
            for j in range(i+1, len(neighbors_job)):
                u, v = neighbors_job[i], neighbors_job[j]
                if G_job.has_edge(u, v):
                    G_job[u][v]["weight"] += 1
                else:
                    G_job.add_edge(u, v, weight=1)

    return G_cv, G_job


def compute_link_prediction_scores(G, cv_nodes, job_nodes, method="adamic_adar"):
    """
    Compute bipartite-aware link prediction scores for non-existing CV-Job pairs.

    Strategy: use the Job-projection G_J to find which Jobs are "similar" to
    a given Job v (share CV neighbors), then score a CV-Job pair (u, v) by
    how many of u's current Job-neighbors are similar to v in G_J.
    This gives meaningful non-zero scores in a bipartite setting.
    """
    cv_set  = set(cv_nodes)
    job_set = set(job_nodes)

    # Non-existing pairs only
    non_existing = [
        (cv, job) for cv in cv_nodes for job in job_nodes
        if not G.has_edge(cv, job) and not G.has_edge(job, cv)
    ]

    if not non_existing:
        return {}

    # Build projections
    G_cv, G_job = _bipartite_projections(G, cv_nodes, job_nodes)

    # Precompute: for each CV, its set of connected Jobs
    cv_job_neighbors = {cv: set(n for n in G.neighbors(cv) if n in job_set)
                        for cv in cv_nodes}
    # For each Job, its set of connected CVs
    job_cv_neighbors = {job: set(n for n in G.neighbors(job) if n in cv_set)
                        for job in job_nodes}

    scores = {}

    if method == "common_neighbors":
        # # of Jobs adjacent to u that are also adjacent (in G_job) to v
        # = "how many of u's jobs are similar to v"
        for u, v in non_existing:
            u_jobs = cv_job_neighbors[u]
            # jobs similar to v (connected in job-projection)
            v_similar_jobs = set(G_job.neighbors(v)) if v in G_job else set()
            scores[(u, v)] = len(u_jobs & v_similar_jobs)

    elif method == "adamic_adar":
        # Weighted version: sum over shared-job-neighbors of 1/log(degree in G_job)
        for u, v in non_existing:
            u_jobs = cv_job_neighbors[u]
            v_similar_jobs = set(G_job.neighbors(v)) if v in G_job else set()
            shared = u_jobs & v_similar_jobs
            score = 0.0
            for w in shared:
                deg = G_job.degree(w) if w in G_job else 1
                if deg > 1:
                    score += 1.0 / np.log(deg)
            # Also add direct path score: CVs sharing jobs with v
            v_cvs = job_cv_neighbors[v]
            u_cvs = set(G_cv.neighbors(u)) if u in G_cv else set()
            shared_cv = v_cvs & u_cvs
            for w in shared_cv:
                deg = G_cv.degree(w) if w in G_cv else 1
                if deg > 1:
                    score += 1.0 / np.log(deg)
            scores[(u, v)] = score

    elif method == "jaccard":
        for u, v in non_existing:
            u_jobs = cv_job_neighbors[u]
            v_similar_jobs = set(G_job.neighbors(v)) if v in G_job else set()
            union = len(u_jobs | v_similar_jobs)
            inter = len(u_jobs & v_similar_jobs)
            scores[(u, v)] = inter / union if union > 0 else 0.0

    elif method == "preferential_attachment":
        # degree in original graph × degree in projection
        for u, v in non_existing:
            scores[(u, v)] = float(G.degree(u) * G.degree(v))

    elif method == "katz":
        # Katz index via 2-hop paths in the bipartite graph
        # score(u,v) = Σ_w [A_uw * A_wv] with damping β=0.5 (length-2 paths)
        beta = 0.5
        for u, v in non_existing:
            # All 2-hop paths from u to v  (u→w→v where w is intermediate)
            paths2 = len([w for w in G.neighbors(u) if G.has_edge(w, v)])
            # 3-hop paths approximation via projections
            u_jobs = cv_job_neighbors[u]
            v_cvs  = job_cv_neighbors[v]
            paths3 = sum(
                len(cv_job_neighbors.get(c, set()) & u_jobs)
                for c in v_cvs
            )
            scores[(u, v)] = beta * paths2 + beta**2 * paths3

    # Normalize to [0, 1]
    max_score = max(scores.values()) if scores else 1.0
    if max_score > 0:
        scores = {k: v / max_score for k, v in scores.items()}

    return scores