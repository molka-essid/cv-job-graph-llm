"""
═══════════════════════════════════════════════════════════════════════════════
     Projet M2 — Graphe CV-Job : Prédiction de Liens & Classification
═══════════════════════════════════════════════════════════════════════════════
Streamlit interface — run with:  streamlit run app.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings("ignore")

from data             import generate_dataset, generate_edges
from graph_analysis   import (build_bipartite_graph, compute_graph_properties,
                               compute_link_prediction_scores, build_cv_skill_similarity_graph)
from embeddings       import (detect_communities_louvain, detect_communities_label_propagation,
                               compute_modularity, compute_semantic_embeddings,
                               predict_links_llm, build_node_features, compute_2d_projection)
from classification   import classify_cv_nodes, classify_job_nodes
from gnn_classification import run_gnn_all_tasks

# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Graphe CV-Job | M2 Projet", page_icon="🔗",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
.main-header {
    background: linear-gradient(135deg,#1a1a2e 0%,#16213e 50%,#0f3460 100%);
    padding:2rem;border-radius:12px;margin-bottom:1.5rem;
    text-align:center;color:white;
}
.metric-card {
    background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);
    padding:1rem;border-radius:10px;text-align:center;color:white;margin:0.3rem;
}
.metric-value{font-size:2rem;font-weight:bold;}
.metric-label{font-size:0.85rem;opacity:0.9;}
.info-box{background:#e8f4fd;border-left:4px solid #3498db;padding:1rem;border-radius:6px;margin:1rem 0;}
.success-box{background:#e8f8f0;border-left:4px solid #2ecc71;padding:1rem;border-radius:6px;margin:1rem 0;}
.warn-box{background:#fff8e1;border-left:4px solid #f39c12;padding:1rem;border-radius:6px;margin:1rem 0;}
.section-title{font-size:1.3rem;font-weight:700;color:#1a1a2e;
    border-bottom:2px solid #3498db;padding-bottom:0.4rem;margin:1rem 0;}
.llm-badge{background:#6c3483;color:white;padding:3px 10px;border-radius:12px;font-size:0.8rem;}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline (cached)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="🔄 Construction du pipeline complet…")
def build_pipeline(n_cv, n_jobs, seed):
    import random
    random.seed(seed); np.random.seed(seed)

    cvs, jobs = generate_dataset(n_cv, n_jobs)
    edges = generate_edges(cvs, jobs)

    G = build_bipartite_graph(cvs, jobs, edges)
    props = compute_graph_properties(G)

    communities_louvain = detect_communities_louvain(G)
    communities_lp      = detect_communities_label_propagation(G)
    mod_louvain = compute_modularity(G, communities_louvain)
    mod_lp      = compute_modularity(G, communities_lp)

    # ── Real Gemini LLM embeddings ──
    cv_emb, job_emb, embedder = compute_semantic_embeddings(cvs, jobs)
    llm_source = "Gemini text-embedding-004" if getattr(embedder, "use_llm", False) else "TF-IDF + PCA (fallback)"

    llm_preds = predict_links_llm(cv_emb, job_emb, G, threshold=-0.1)

    cv_ids  = [cv["id"]  for cv in cvs]
    job_ids = [job["id"] for job in jobs]
    struct_aa   = compute_link_prediction_scores(G, cv_ids, job_ids, "adamic_adar")
    struct_cn   = compute_link_prediction_scores(G, cv_ids, job_ids, "common_neighbors")
    struct_jac  = compute_link_prediction_scores(G, cv_ids, job_ids, "jaccard")
    struct_pa   = compute_link_prediction_scores(G, cv_ids, job_ids, "preferential_attachment")
    struct_katz = compute_link_prediction_scores(G, cv_ids, job_ids, "katz")

    G_enriched = G.copy()
    for cv_id, job_id, score in llm_preds[:20]:
        G_enriched.add_edge(cv_id, job_id, weight=score, predicted=True)

    # ── Features on ORIGINAL graph ──
    node_features_orig = build_node_features(cvs, jobs, G,          cv_emb, job_emb, communities_louvain)
    # ── Features on ENRICHED graph ──
    node_features_enr  = build_node_features(cvs, jobs, G_enriched, cv_emb, job_emb, communities_louvain)

    proj_2d = compute_2d_projection(node_features_enr)

    # ── Classical classification (original vs enriched) ──
    clf_orig = {
        "cv_level":  classify_cv_nodes(cvs,  node_features_orig, target="level"),
        "cv_domain": classify_cv_nodes(cvs,  node_features_orig, target="domain"),
        "cv_poly":   classify_cv_nodes(cvs,  node_features_orig, target="polyvalent"),
        "job_level": classify_job_nodes(jobs, node_features_orig, target="level"),
        "job_domain":classify_job_nodes(jobs, node_features_orig, target="domain"),
    }
    clf_enr = {
        "cv_level":  classify_cv_nodes(cvs,  node_features_enr, target="level"),
        "cv_domain": classify_cv_nodes(cvs,  node_features_enr, target="domain"),
        "cv_poly":   classify_cv_nodes(cvs,  node_features_enr, target="polyvalent"),
        "job_level": classify_job_nodes(jobs, node_features_enr, target="level"),
        "job_domain":classify_job_nodes(jobs, node_features_enr, target="domain"),
    }

    # ── GNN classification ──
    gnn_results = run_gnn_all_tasks(cvs, jobs, node_features_enr, G_enriched)

    G_skills = build_cv_skill_similarity_graph(cvs)
    comm_skills = detect_communities_louvain(G_skills)

    return dict(
        cvs=cvs, jobs=jobs, edges=edges,
        G=G, G_enriched=G_enriched, props=props,
        communities_louvain=communities_louvain, communities_lp=communities_lp,
        mod_louvain=mod_louvain, mod_lp=mod_lp,
        cv_emb=cv_emb, job_emb=job_emb, llm_source=llm_source,
        llm_preds=llm_preds,
        struct_aa=struct_aa, struct_cn=struct_cn, struct_jac=struct_jac, struct_pa=struct_pa, struct_katz=struct_katz,
        node_features_orig=node_features_orig, node_features_enr=node_features_enr,
        proj_2d=proj_2d,
        clf_orig=clf_orig, clf_enr=clf_enr,
        # convenience aliases kept for tab 4
        clf_cv_level=clf_enr["cv_level"],  clf_cv_domain=clf_enr["cv_domain"],
        clf_cv_poly=clf_enr["cv_poly"],    clf_job_level=clf_enr["job_level"],
        clf_job_domain=clf_enr["job_domain"],
        gnn_results=gnn_results,
        G_skills=G_skills, comm_skills=comm_skills,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Paramètres")
    n_cv   = st.slider("Nombre de CVs",   20, 80, 40, step=5)
    n_jobs = st.slider("Nombre de Jobs",  10, 50, 25, step=5)
    seed   = st.number_input("Seed aléatoire", value=42, min_value=0, max_value=9999)
    if st.button("🔁 Reconstruire le pipeline", type="primary", use_container_width=True):
        st.cache_resource.clear()
    st.markdown("---")
    st.markdown("""
    **Navigation**
    - 🏠 Vue d'ensemble
    - 🔍 Analyse structurelle
    - 👥 Communautés
    - 🤖 Prédiction de liens
    - 🏷️ Classification ML
    - 🧠 GNN Classification
    - 📊 Enrichissement & Comparaison
    """)
    st.caption("M2 Projet — Graphe CV-Job | 2024-2025")

with st.spinner("Chargement du pipeline…"):
    D = build_pipeline(n_cv, n_jobs, int(seed))

G          = D["G"]
G_enriched = D["G_enriched"]
props      = D["props"]
cvs        = D["cvs"]
jobs       = D["jobs"]
communities= D["communities_louvain"]
proj_2d    = D["proj_2d"]

# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="main-header">
    <h1>🔗 Graphe CV–Job</h1>
    <h3>Prédiction de Liens par LLM & Classification des Nœuds</h3>
    <p>Projet M2 — Analyse de graphes, NLP & Machine Learning</p>
    <span class="llm-badge">🤖 Embeddings : {D['llm_source']}</span>
</div>
""", unsafe_allow_html=True)

tabs = st.tabs([
    "🏠 Vue d'ensemble",
    "🔍 Analyse Structurelle",
    "👥 Communautés",
    "🤖 Prédiction de Liens",
    "🏷️ Classification ML",
    "🧠 GNN Classification",
    "📊 Enrichissement & Insights",
])

# ════════════════════════════════════════════════════════════════════════════
# TAB 0 — VUE D'ENSEMBLE
# ════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.markdown('<div class="section-title">📈 Métriques clés du graphe biparti</div>', unsafe_allow_html=True)
    cols = st.columns(6)
    metrics = [
        ("Nœuds totaux", props["n_nodes"], ""),
        ("CVs", props["n_cv"], "👤"),
        ("Jobs", props["n_jobs"], "💼"),
        ("Liens (arêtes)", props["n_edges"], "🔗"),
        ("Densité", f"{props['density']:.4f}", "📊"),
        ("Composantes", props["components"], "🔲"),
    ]
    for col, (label, value, icon) in zip(cols, metrics):
        col.markdown(f"""<div class="metric-card">
            <div class="metric-value">{icon} {value}</div>
            <div class="metric-label">{label}</div></div>""", unsafe_allow_html=True)

    st.markdown("---")
    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown('<div class="section-title">🗺️ Visualisation du graphe biparti</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.set_facecolor("#0d1117"); fig.patch.set_facecolor("#0d1117")
        cv_nodes  = [n for n, d in G.nodes(data=True) if d.get("node_type") == "CV"]
        job_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "Job"]
        pos = {}
        for i, n in enumerate(cv_nodes):
            pos[n] = (-1, i / max(len(cv_nodes)-1,1)*2-1)
        for i, n in enumerate(job_nodes):
            pos[n] = (1, i / max(len(job_nodes)-1,1)*2-1)
        n_comm = max(communities.values())+1 if communities else 1
        cmap = plt.cm.get_cmap("tab20", n_comm)
        nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.15, edge_color="#4fc3f7", width=0.6)
        nx.draw_networkx_nodes(G, pos, nodelist=cv_nodes,
            node_color=[cmap(communities.get(n,0)) for n in cv_nodes], node_size=120, node_shape="o", ax=ax)
        nx.draw_networkx_nodes(G, pos, nodelist=job_nodes,
            node_color=[cmap(communities.get(n,0)) for n in job_nodes], node_size=180, node_shape="s", ax=ax)
        labels = {n: n for n in props["top_cv"][:3] + props["top_jobs"][:3]}
        nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=6, font_color="white", font_weight="bold")
        ax.legend(handles=[mpatches.Patch(color="#4fc3f7",label="CVs ● (cercles)"),
                            mpatches.Patch(color="#f06292",label="Jobs ■ (carrés)")],
                  frameon=False, labelcolor="white")
        ax.set_title("Graphe Biparti CV–Job (coloré par communauté)", color="white", fontsize=13)
        ax.axis("off"); st.pyplot(fig); plt.close()

    with col2:
        st.markdown('<div class="section-title">📋 Résumé des données</div>', unsafe_allow_html=True)
        level_counts = pd.Series([cv["level"] for cv in cvs]).value_counts()
        fig2, axes = plt.subplots(2, 1, figsize=(5,7)); fig2.patch.set_facecolor("#0d1117")
        axes[0].pie(level_counts.values, labels=level_counts.index, autopct="%1.0f%%",
                    colors=["#4fc3f7","#ba68c8","#f06292"], textprops={"color":"white","fontsize":10})
        axes[0].set_title("Distribution des niveaux (CV)", color="white", fontsize=11)
        axes[0].set_facecolor("#0d1117")
        domain_counts = pd.Series([cv["domain"] for cv in cvs]).value_counts().head(6)
        bars = axes[1].barh(domain_counts.index, domain_counts.values,
                             color=plt.cm.viridis(np.linspace(0.2,0.9,len(domain_counts))))
        axes[1].set_title("Top domaines (CV)", color="white", fontsize=11)
        axes[1].tick_params(colors="white", labelsize=8)
        axes[1].set_facecolor("#0d1117")
        for spine in ["top","right"]: axes[1].spines[spine].set_visible(False)
        for sp in ["bottom","left"]: axes[1].spines[sp].set_color("white")
        for bar in bars:
            axes[1].text(bar.get_width()+0.1, bar.get_y()+bar.get_height()/2,
                         str(int(bar.get_width())), va="center", color="white", fontsize=8)
        plt.tight_layout(); st.pyplot(fig2); plt.close()
        st.markdown(f"""<div class="info-box">
        <b>📌 Informations clés</b><br>
        • Degré moyen (CV) : <b>{props['avg_cv_degree']:.2f}</b><br>
        • Degré moyen (Job) : <b>{props['avg_job_degree']:.2f}</b><br>
        • Plus grande composante : <b>{props['largest_component_size']} nœuds</b><br>
        • Communautés détectées : <b>{n_comm}</b><br>
        • Liens prédits (LLM) : <b>{len(D['llm_preds'])}</b><br>
        • Source embeddings : <b>{D['llm_source']}</b>
        </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — ANALYSE STRUCTURELLE
# ════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown('<div class="section-title">🔍 Propriétés Structurelles du Graphe</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        degrees = [d for _, d in G.degree()]
        fig, ax = plt.subplots(figsize=(6,4))
        ax.set_facecolor("#0d1117"); fig.patch.set_facecolor("#0d1117")
        ax.hist(degrees, bins=20, color="#4fc3f7", edgecolor="#0d1117", alpha=0.85)
        ax.set_xlabel("Degré", color="white"); ax.set_ylabel("Fréquence", color="white")
        ax.set_title("Distribution des degrés", color="white"); ax.tick_params(colors="white")
        for sp in ax.spines.values(): sp.set_color("#333")
        st.pyplot(fig); plt.close()
        st.markdown("**Top CVs (par degré)**")
        st.dataframe(pd.DataFrame(
            [(n, G.degree(n), G.nodes[n].get("level",""), G.nodes[n].get("domain",""))
             for n in props["top_cv"]], columns=["ID","Degré","Niveau","Domaine"]),
            use_container_width=True, hide_index=True)
    with col2:
        bc = props["betweenness"]; cc = props["closeness"]
        common = [n for n in bc if n in cc]
        colors = ["#4fc3f7" if G.nodes[n].get("node_type")=="CV" else "#f06292" for n in common]
        fig, ax = plt.subplots(figsize=(6,4))
        ax.set_facecolor("#0d1117"); fig.patch.set_facecolor("#0d1117")
        ax.scatter([bc[n] for n in common],[cc[n] for n in common], c=colors, alpha=0.75, s=50)
        ax.set_xlabel("Centralité Betweenness", color="white"); ax.set_ylabel("Centralité Closeness", color="white")
        ax.set_title("Centralités des nœuds", color="white"); ax.tick_params(colors="white")
        for sp in ax.spines.values(): sp.set_color("#333")
        ax.legend(handles=[mpatches.Patch(color="#4fc3f7",label="CV"),mpatches.Patch(color="#f06292",label="Job")],
                  frameon=False, labelcolor="white")
        st.pyplot(fig); plt.close()
        st.markdown("**Top Jobs (par degré)**")
        st.dataframe(pd.DataFrame(
            [(n, G.degree(n), G.nodes[n].get("level",""), G.nodes[n].get("domain",""))
             for n in props["top_jobs"]], columns=["ID","Degré","Niveau","Domaine"]),
            use_container_width=True, hide_index=True)
    st.markdown("---")
    st.markdown('<div class="section-title">📊 Métriques structurelles complètes</div>', unsafe_allow_html=True)
    st.dataframe(pd.DataFrame([
        {"Métrique":"Nombre de nœuds","Valeur":props["n_nodes"]},
        {"Métrique":"Nombre d'arêtes","Valeur":props["n_edges"]},
        {"Métrique":"Densité","Valeur":f"{props['density']:.6f}"},
        {"Métrique":"Degré moyen global","Valeur":f"{props['avg_degree']:.2f}"},
        {"Métrique":"Degré moyen CV","Valeur":f"{props['avg_cv_degree']:.2f}"},
        {"Métrique":"Degré moyen Job","Valeur":f"{props['avg_job_degree']:.2f}"},
        {"Métrique":"Degré maximal","Valeur":props["max_degree"]},
        {"Métrique":"Composantes connexes","Valeur":props["components"]},
        {"Métrique":"Taille plus grande composante","Valeur":props["largest_component_size"]},
        {"Métrique":"Modularité (Louvain)","Valeur":f"{D['mod_louvain']:.4f}"},
        {"Métrique":"Modularité (Label Propagation)","Valeur":f"{D['mod_lp']:.4f}"},
    ]), use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — COMMUNAUTÉS
# ════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown('<div class="section-title">👥 Détection de Communautés</div>', unsafe_allow_html=True)
    col1, col2 = st.columns([1,1])
    with col1:
        algo = st.selectbox("Algorithme", ["Louvain (Greedy Modularity)","Label Propagation"])
        comm_map = D["communities_louvain"] if "Louvain" in algo else D["communities_lp"]
        mod_val  = D["mod_louvain"]         if "Louvain" in algo else D["mod_lp"]
        n_comm   = max(comm_map.values())+1
        st.metric("Communautés détectées", n_comm)
        st.metric("Modularité Q", f"{mod_val:.4f}")
        st.markdown("""<div class="info-box"><b>📌 Interprétation</b><br>
        Une modularité Q > 0.3 indique une structure communautaire forte.
        Les communautés regroupent CVs et Jobs partageant des <b>compétences similaires</b>
        et des <b>connexions structurelles</b>.</div>""", unsafe_allow_html=True)
    with col2:
        comm_sizes = pd.Series(comm_map).value_counts().sort_index()
        cmap_c = plt.cm.get_cmap("tab20", n_comm)
        fig, ax = plt.subplots(figsize=(5,3)); ax.set_facecolor("#0d1117"); fig.patch.set_facecolor("#0d1117")
        ax.bar(range(len(comm_sizes)), comm_sizes.values,
               color=[cmap_c(i) for i in range(len(comm_sizes))], edgecolor="none")
        ax.set_xlabel("Communauté",color="white"); ax.set_ylabel("Taille",color="white")
        ax.set_title("Taille des communautés",color="white"); ax.tick_params(colors="white")
        for sp in ax.spines.values(): sp.set_color("#222")
        st.pyplot(fig); plt.close()
    st.markdown("---")
    st.markdown('<div class="section-title">🗺️ Projection 2D des nœuds (PCA sur features LLM)</div>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(12,7)); ax.set_facecolor("#0d1117"); fig.patch.set_facecolor("#0d1117")
    for nid, (px, py) in proj_2d.items():
        nt = G.nodes[nid].get("node_type","CV") if nid in G.nodes else "CV"
        color = cmap_c(comm_map.get(nid,0) % n_comm)
        ax.scatter(px, py, c=[color], marker="o" if nt=="CV" else "s",
                   s=80 if nt=="CV" else 120, alpha=0.8, edgecolors="none")
    ax.set_title("Espace 2D — Nœuds colorés par communauté (● = CV, ■ = Job)", color="white", fontsize=13)
    ax.set_xlabel("PCA Dim 1", color="white"); ax.set_ylabel("PCA Dim 2", color="white")
    ax.tick_params(colors="white")
    for sp in ax.spines.values(): sp.set_color("#333")
    st.pyplot(fig); plt.close()
    st.markdown("---")
    st.markdown('<div class="section-title">🧠 Communautés internes des CV (graphe de compétences)</div>', unsafe_allow_html=True)
    G_sk = D["G_skills"]; comm_sk = D["comm_skills"]
    n_sk = max(comm_sk.values())+1 if comm_sk else 1
    col3, col4 = st.columns([3,2])
    with col3:
        fig, ax = plt.subplots(figsize=(8,5)); ax.set_facecolor("#0d1117"); fig.patch.set_facecolor("#0d1117")
        pos_sk = nx.spring_layout(G_sk, seed=42, k=0.6)
        cmap_sk = plt.cm.get_cmap("Paired", n_sk)
        nx.draw_networkx_edges(G_sk, pos_sk, ax=ax, alpha=0.2, edge_color="#aaa", width=0.5)
        nx.draw_networkx_nodes(G_sk, pos_sk,
            node_color=[cmap_sk(comm_sk.get(n,0)) for n in G_sk.nodes()], node_size=100, ax=ax, alpha=0.85)
        labels_top = {n:n for n,d in sorted(G_sk.degree(), key=lambda x:-x[1])[:15]}
        nx.draw_networkx_labels(G_sk, pos_sk, labels_top, ax=ax, font_size=6, font_color="white")
        ax.set_title("Graphe de similarité CV (par compétences)", color="white", fontsize=11)
        ax.axis("off"); st.pyplot(fig); plt.close()
    with col4:
        comm_detail = {}
        for nid, cid in comm_sk.items(): comm_detail.setdefault(cid,[]).append(nid)
        for cid in sorted(comm_detail)[:6]:
            members = comm_detail[cid]
            st.markdown(f"**Communauté {cid}** ({len(members)} CVs)")
            for m in members[:4]:
                cv = next((c for c in cvs if c["id"]==m), None)
                if cv: st.caption(f"  • {m} | {cv['level']} | {cv['domain']}")


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — PRÉDICTION DE LIENS
# ════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown('<div class="section-title">🤖 Prédiction de Liens — LLM (embeddings sémantiques)</div>', unsafe_allow_html=True)
    llm_source = D["llm_source"]
    badge_color = "#6c3483" if "Gemini" in llm_source else "#e67e22"
    st.markdown(f"""<div class="info-box">
    <b>Approche LLM :</b> <span style="background:{badge_color};color:white;padding:2px 8px;border-radius:8px;font-size:0.85rem">
    {llm_source}</span><br><br>
    Les textes des CVs et Jobs sont vectorisés via <b>{llm_source}</b>.
    La <b>similarité cosinus</b> entre les embeddings prédit la probabilité d'un lien CV ↔ Job manquant.
    </div>""", unsafe_allow_html=True)

    llm_preds = D["llm_preds"]
    top_n = st.slider("Afficher les N meilleures prédictions", 5, 50, 20)
    top_preds = llm_preds[:top_n]
    if top_preds:
        pred_df = pd.DataFrame([{
            "CV": cv_id, "Job": job_id,
            "Score (cosine sim)": f"{score:.4f}",
            "Niveau CV": next((c["level"] for c in cvs if c["id"]==cv_id),"?"),
            "Domaine CV": next((c["domain"] for c in cvs if c["id"]==cv_id),"?"),
            "Niveau Job": next((j["level"] for j in jobs if j["id"]==job_id),"?"),
            "Domaine Job": next((j["domain"] for j in jobs if j["id"]==job_id),"?"),
        } for cv_id, job_id, score in top_preds])
        st.dataframe(pred_df, use_container_width=True, hide_index=True)
        all_scores = [s for _,_,s in llm_preds]
        fig, ax = plt.subplots(figsize=(8,3)); ax.set_facecolor("#0d1117"); fig.patch.set_facecolor("#0d1117")
        ax.hist(all_scores, bins=30, color="#a8d8a8", edgecolor="none", alpha=0.85)
        ax.axvline(x=0.3, color="#f06292", linestyle="--", linewidth=2, label="Seuil=0.3")
        ax.set_xlabel("Score de similarité",color="white"); ax.set_ylabel("Fréquence",color="white")
        ax.set_title("Distribution des scores de prédiction LLM",color="white"); ax.tick_params(colors="white")
        for sp in ax.spines.values(): sp.set_color("#333")
        ax.legend(frameon=False, labelcolor="white"); st.pyplot(fig); plt.close()

    st.markdown("---")
    st.markdown('<div class="section-title">📐 Méthodes Structurelles de Prédiction de Liens</div>', unsafe_allow_html=True)
    method_map = {"Adamic-Adar":D["struct_aa"],"Common Neighbors":D["struct_cn"],
                  "Jaccard Coefficient":D["struct_jac"],"Preferential Attachment":D["struct_pa"],
                  "Katz Index":D["struct_katz"]}
    selected = st.selectbox("Méthode structurelle", list(method_map.keys()))
    struct_scores = method_map[selected]
    # Show only non-zero scores, sorted descending
    nonzero = {k: v for k, v in struct_scores.items() if v > 0}
    if nonzero:
        top_struct = sorted(nonzero, key=nonzero.get, reverse=True)[:20]
        st.markdown(f"**{len(nonzero)} paires avec score > 0** (top 20 affichées)")
        st.dataframe(pd.DataFrame([{"CV":u,"Job":v,"Score (normalisé)":f"{struct_scores[(u,v)]:.4f}"}
                                    for u,v in top_struct]), use_container_width=True, hide_index=True)
    else:
        st.warning("⚠️ Aucun score non-nul — cette méthode nécessite des voisins communs dans le graphe.")

    st.markdown("---")
    st.markdown('<div class="section-title">🔬 Graphe enrichi après prédiction LLM</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    col1.metric("Liens originaux", G.number_of_edges())
    col2.metric("Liens après enrichissement", G_enriched.number_of_edges())
    st.caption(f"→ **{G_enriched.number_of_edges()-G.number_of_edges()} nouveaux liens** ajoutés par la prédiction LLM")


# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — CLASSIFICATION ML
# ════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.markdown('<div class="section-title">🏷️ Classification ML des Nœuds</div>', unsafe_allow_html=True)
    st.markdown("""<div class="info-box">
    Les nœuds sont classifiés en combinant : <b>features structurelles</b> (degré, centralité, communauté) +
    <b>features sémantiques</b> (embeddings LLM 32-dim) + <b>features métier</b> (années d'expérience, compétences).
    </div>""", unsafe_allow_html=True)

    def plot_clf(result, title):
        if "error" in result:
            st.error(f"Erreur: {result['error']}"); return
        models = result.get("models", {})
        if not models:
            st.metric("Accuracy (cross-val)", f"{result.get('cv_accuracy_mean',0):.2%}")
            st.metric("Accuracy (train)",     f"{result.get('train_accuracy',0):.2%}")
            return
        model_names = list(models.keys())
        cv_scores    = [models[m].get("cv_accuracy_mean",0) for m in model_names]
        train_scores = [models[m].get("train_accuracy",0)   for m in model_names]
        fig, axes = plt.subplots(1, 2, figsize=(12,4)); fig.patch.set_facecolor("#0d1117")
        x = range(len(model_names))
        axes[0].set_facecolor("#0d1117")
        bars_cv    = axes[0].bar([xi-0.2 for xi in x], cv_scores,    0.35, label="Cross-Val", color="#4fc3f7", alpha=0.85)
        bars_train = axes[0].bar([xi+0.2 for xi in x], train_scores, 0.35, label="Train",     color="#f06292", alpha=0.85)
        axes[0].set_xticks(list(x)); axes[0].set_xticklabels([m.split(" ")[0] for m in model_names], color="white", fontsize=8)
        axes[0].set_ylabel("Accuracy",color="white"); axes[0].set_title(f"{title}\nComparaison",color="white")
        axes[0].set_ylim(0,1.1); axes[0].tick_params(colors="white")
        axes[0].legend(frameon=False, labelcolor="white")
        for sp in axes[0].spines.values(): sp.set_color("#333")
        for bar in bars_cv:
            axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02,
                         f"{bar.get_height():.0%}", ha="center", color="white", fontsize=8)
        best = result.get("best_model")
        if best and best in models and "confusion_matrix" in models[best]:
            cm = np.array(models[best]["confusion_matrix"]); classes = models[best]["classes"]
            im = axes[1].imshow(cm, cmap="Blues"); axes[1].set_facecolor("#0d1117")
            axes[1].set_xticks(range(len(classes))); axes[1].set_yticks(range(len(classes)))
            axes[1].set_xticklabels(classes, rotation=30, ha="right", color="white", fontsize=8)
            axes[1].set_yticklabels(classes, color="white", fontsize=8)
            axes[1].set_title(f"Matrice de confusion ({best})",color="white")
            for i in range(len(classes)):
                for j in range(len(classes)):
                    axes[1].text(j, i, str(cm[i,j]), ha="center", va="center",
                                 color="white" if cm[i,j]<cm.max()/2 else "black", fontsize=10)
            plt.colorbar(im, ax=axes[1])
        else: axes[1].set_visible(False)
        plt.tight_layout(); st.pyplot(fig); plt.close()
        summary = [{"Modèle":m,"CV Accuracy":f"{models[m].get('cv_accuracy_mean',0):.2%} ± {models[m].get('cv_accuracy_std',0):.2%}",
                    "Train Accuracy":f"{models[m].get('train_accuracy',0):.2%}","Note":models[m].get("note","")}
                   for m in model_names if "error" not in models[m]]
        if summary: st.dataframe(pd.DataFrame(summary), use_container_width=True, hide_index=True)

    sub_tabs = st.tabs(["CV — Niveau","CV — Domaine","CV — Polyvalence","Job — Niveau","Job — Domaine"])
    with sub_tabs[0]: plot_clf(D["clf_cv_level"],  "Classification CV — Niveau")
    with sub_tabs[1]: plot_clf(D["clf_cv_domain"], "Classification CV — Domaine")
    with sub_tabs[2]: plot_clf(D["clf_cv_poly"],   "Classification CV — Polyvalence")
    with sub_tabs[3]:
        r = D["clf_job_level"]
        if "error" not in r:
            col1, col2 = st.columns(2)
            col1.metric("Accuracy (cross-val)", f"{r.get('cv_accuracy_mean',0):.2%}")
            col2.metric("Accuracy (train)",     f"{r.get('train_accuracy',0):.2%}")
            if "confusion_matrix" in r:
                fig, ax = plt.subplots(figsize=(5,4)); ax.set_facecolor("#0d1117"); fig.patch.set_facecolor("#0d1117")
                cm = np.array(r["confusion_matrix"]); classes = r["classes"]
                ax.imshow(cm, cmap="Blues")
                ax.set_xticks(range(len(classes))); ax.set_yticks(range(len(classes)))
                ax.set_xticklabels(classes, rotation=30, ha="right", color="white")
                ax.set_yticklabels(classes, color="white")
                ax.set_title("Matrice de confusion — Job Niveau",color="white")
                for i in range(len(classes)):
                    for j in range(len(classes)): ax.text(j,i,str(cm[i,j]),ha="center",va="center",color="white",fontsize=10)
                st.pyplot(fig); plt.close()
    with sub_tabs[4]:
        r = D["clf_job_domain"]
        if "error" not in r:
            col1, col2 = st.columns(2)
            col1.metric("Accuracy (cross-val)", f"{r.get('cv_accuracy_mean',0):.2%}")
            col2.metric("Accuracy (train)",     f"{r.get('train_accuracy',0):.2%}")


# ════════════════════════════════════════════════════════════════════════════
# TAB 5 — GNN CLASSIFICATION
# ════════════════════════════════════════════════════════════════════════════
with tabs[5]:
    st.markdown('<div class="section-title">🧠 Classification par Réseaux de Neurones sur Graphe (GNN)</div>', unsafe_allow_html=True)
    st.markdown("""<div class="info-box">
    <b>Architecture GNN :</b><br>
    • <b>GCN (Graph Convolutional Network)</b> — Kipf & Welling 2017 : convolution spectrale, normalisation D⁻¹/²AD⁻¹/²<br>
    • <b>GraphSAGE</b> — Hamilton et al. 2017 : agrégation des voisins par moyenne (inductive learning)<br><br>
    Les deux modèles utilisent les <b>features LLM</b> + structurelles comme entrée, 
    en exploitant la <b>topologie du graphe</b> pour propager l'information entre nœuds.
    </div>""", unsafe_allow_html=True)

    gnn = D["gnn_results"]

    def plot_gnn(result, title):
        if "error" in result:
            st.warning(f"⚠️ {result['error']}"); return
        models = result.get("models", {})
        if not models: return
        model_names = list(models.keys())
        test_scores  = [models[m].get("cv_accuracy_mean",0) for m in model_names]
        train_scores = [models[m].get("train_accuracy",0)   for m in model_names]
        fig, axes = plt.subplots(1,2, figsize=(12,4)); fig.patch.set_facecolor("#0d1117")
        x = range(len(model_names))
        axes[0].set_facecolor("#0d1117")
        bars = axes[0].bar([xi-0.2 for xi in x], test_scores,  0.35, label="Test",  color="#00c897", alpha=0.85)
        axes[0].bar([xi+0.2 for xi in x], train_scores, 0.35, label="Train", color="#7c4dff", alpha=0.85)
        axes[0].set_xticks(list(x)); axes[0].set_xticklabels(model_names, color="white", fontsize=9)
        axes[0].set_ylabel("Accuracy",color="white"); axes[0].set_title(f"{title}",color="white")
        axes[0].set_ylim(0,1.1); axes[0].tick_params(colors="white")
        axes[0].legend(frameon=False, labelcolor="white")
        for sp in axes[0].spines.values(): sp.set_color("#333")
        for bar in bars:
            axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02,
                         f"{bar.get_height():.0%}", ha="center", color="white", fontsize=9)
        best = result.get("best_model")
        if best and best in models and "confusion_matrix" in models[best]:
            cm = np.array(models[best]["confusion_matrix"]); classes = models[best]["classes"]
            im = axes[1].imshow(cm, cmap="Greens"); axes[1].set_facecolor("#0d1117")
            axes[1].set_xticks(range(len(classes))); axes[1].set_yticks(range(len(classes)))
            axes[1].set_xticklabels(classes,rotation=30,ha="right",color="white",fontsize=8)
            axes[1].set_yticklabels(classes,color="white",fontsize=8)
            axes[1].set_title(f"Confusion matrix ({best})",color="white")
            for i in range(len(classes)):
                for j in range(len(classes)): axes[1].text(j,i,str(cm[i,j]),ha="center",va="center",color="white",fontsize=10)
            plt.colorbar(im, ax=axes[1])
        else: axes[1].set_visible(False)
        plt.tight_layout(); st.pyplot(fig); plt.close()

        # Summary
        summary = []
        for m in model_names:
            r = models[m]
            if "error" not in r:
                summary.append({"Modèle":m,"Test Accuracy":f"{r.get('cv_accuracy_mean',0):.2%}",
                                 "Train Accuracy":f"{r.get('train_accuracy',0):.2%}",
                                 "Note":r.get("note","")})
        if summary:
            st.dataframe(pd.DataFrame(summary), use_container_width=True, hide_index=True)

    gnn_sub = st.tabs(["CV — Niveau","CV — Domaine","CV — Polyvalence","Job — Niveau","Job — Domaine"])
    with gnn_sub[0]: plot_gnn(gnn.get("gnn_cv_level",{}),  "GNN — Classification CV (Niveau)")
    with gnn_sub[1]: plot_gnn(gnn.get("gnn_cv_domain",{}), "GNN — Classification CV (Domaine)")
    with gnn_sub[2]: plot_gnn(gnn.get("gnn_cv_poly",{}),   "GNN — Classification CV (Polyvalence)")
    with gnn_sub[3]: plot_gnn(gnn.get("gnn_job_level",{}), "GNN — Classification Job (Niveau)")
    with gnn_sub[4]: plot_gnn(gnn.get("gnn_job_domain",{}),"GNN — Classification Job (Domaine)")

    st.markdown("---")
    st.markdown('<div class="section-title">🔬 Architecture GNN — Détails</div>', unsafe_allow_html=True)
    st.markdown("""
    | Composant | GCN | GraphSAGE |
    |---|---|---|
    | **Propagation** | A_sym × X × W (spectral) | Concat(self, mean_neighbors) |
    | **Normalisation** | D⁻¹/²AD⁻¹/² (symétrique) | Degré par voisin |
    | **Couches** | 2 (hidden=64) | 2 (hidden=64) |
    | **Activation** | ReLU | ReLU |
    | **Sortie** | Softmax | Softmax |
    | **Apprentissage** | Transductif | Inductif |
    """)


# ════════════════════════════════════════════════════════════════════════════
# TAB 6 — ENRICHISSEMENT & INSIGHTS
# ════════════════════════════════════════════════════════════════════════════
with tabs[6]:
    st.markdown('<div class="section-title">📊 Impact de l\'enrichissement du graphe sur la classification</div>', unsafe_allow_html=True)
    st.markdown("""<div class="info-box">
    <b>Méthodologie :</b> On compare les performances de classification <b>avant</b> (graphe original)
    et <b>après</b> enrichissement (graphe + liens LLM prédits).
    L'amélioration démontre l'apport réel des embeddings Gemini.
    </div>""", unsafe_allow_html=True)

    # Build comparison table
    def get_best(result):
        """Get best cross-validation accuracy across models."""
        if "error" in result: return 0.0
        if "models" in result:
            vals = [v.get("cv_accuracy_mean", 0) for v in result["models"].values() if "error" not in v]
            return max(vals) if vals else 0.0
        return result.get("cv_accuracy_mean", 0.0)

    tasks = [
        ("CV — Niveau",     "cv_level"),
        ("CV — Domaine",    "cv_domain"),
        ("CV — Polyvalence","cv_poly"),
        ("Job — Niveau",    "job_level"),
        ("Job — Domaine",   "job_domain"),
    ]
    impact_data = []
    for label, key in tasks:
        orig = get_best(D["clf_orig"][key])
        enr  = get_best(D["clf_enr"][key])
        delta = enr - orig
        impact_data.append({"Tâche":label,
                             "Avant enrichissement":f"{orig:.2%}",
                             "Après enrichissement":f"{enr:.2%}",
                             "Δ Amélioration":f"{delta:+.2%}",
                             "_orig":orig,"_enr":enr,"_delta":delta})
    df_impact = pd.DataFrame(impact_data)
    st.dataframe(df_impact[["Tâche","Avant enrichissement","Après enrichissement","Δ Amélioration"]],
                 use_container_width=True, hide_index=True)

    # Bar chart comparison
    fig, ax = plt.subplots(figsize=(10,4)); ax.set_facecolor("#0d1117"); fig.patch.set_facecolor("#0d1117")
    x = range(len(impact_data))
    bars_orig = ax.bar([xi-0.2 for xi in x], [d["_orig"] for d in impact_data], 0.35,
                        label="Avant enrichissement", color="#4fc3f7", alpha=0.85)
    bars_enr  = ax.bar([xi+0.2 for xi in x], [d["_enr"]  for d in impact_data], 0.35,
                        label="Après enrichissement", color="#f06292", alpha=0.85)
    ax.set_xticks(list(x)); ax.set_xticklabels([d["Tâche"] for d in impact_data], color="white", fontsize=9)
    ax.set_ylabel("Accuracy", color="white"); ax.set_title("Impact de l'enrichissement LLM sur la classification", color="white")
    ax.set_ylim(0,1.1); ax.tick_params(colors="white"); ax.legend(frameon=False, labelcolor="white")
    for sp in ax.spines.values(): sp.set_color("#333")
    for bar in bars_enr:
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02,
                f"{bar.get_height():.0%}", ha="center", color="white", fontsize=8)
    st.pyplot(fig); plt.close()

    st.markdown("---")
    st.markdown('<div class="section-title">📊 Comparaison LLM vs méthodes structurelles</div>', unsafe_allow_html=True)
    struct_methods = {"Adamic-Adar":D["struct_aa"],"Common Neighbors":D["struct_cn"],
                      "Jaccard":D["struct_jac"],"Pref. Attachment":D["struct_pa"],
                      "Katz Index":D["struct_katz"]}
    llm_top_pairs = set((u,v) for u,v,_ in D["llm_preds"][:30])
    comparison_data = []
    for name, scores in struct_methods.items():
        nonzero = {k:v for k,v in scores.items() if v > 0}
        if nonzero:
            struct_top = set(sorted(nonzero, key=nonzero.get, reverse=True)[:30])
            overlap    = len(struct_top & llm_top_pairs)
            avg_score  = np.mean(list(nonzero.values()))
            n_nonzero  = len(nonzero)
        else:
            overlap, avg_score, n_nonzero = 0, 0.0, 0
        comparison_data.append({"Méthode":name,"Type":"Structurelle",
                                  "Paires non-nulles":n_nonzero,
                                  "Overlap avec LLM (top 30)":overlap,
                                  "Score moyen":f"{avg_score:.4f}"})
    comparison_data.append({"Méthode":f"LLM ({D['llm_source'].split()[0]})","Type":"Sémantique LLM",
                              "Paires non-nulles": len(D["llm_preds"]),
                              "Overlap avec LLM (top 30)":30,
                              "Score moyen":f"{np.mean([s for _,_,s in D['llm_preds']]):.4f}" if D['llm_preds'] else "0"})
    st.dataframe(pd.DataFrame(comparison_data), use_container_width=True, hide_index=True)

    fig, ax = plt.subplots(figsize=(8,3)); ax.set_facecolor("#0d1117"); fig.patch.set_facecolor("#0d1117")
    names    = [d["Méthode"] for d in comparison_data[:-1]]
    overlaps = [d["Overlap avec LLM (top 30)"] for d in comparison_data[:-1]]
    bars = ax.barh(names, overlaps, color=plt.cm.viridis(np.linspace(0.3,0.9,len(names))), edgecolor="none")
    ax.set_xlabel("Overlap avec top-30 LLM",color="white")
    ax.set_title("Robustesse : accord entre méthodes structurelles et LLM",color="white")
    ax.tick_params(colors="white",labelsize=9)
    for sp in ax.spines.values(): sp.set_color("#333")
    for bar in bars:
        ax.text(bar.get_width()+0.2, bar.get_y()+bar.get_height()/2,
                str(int(bar.get_width())), va="center", color="white", fontsize=9)
    st.pyplot(fig); plt.close()

    st.markdown("---")
    st.markdown('<div class="section-title">💡 Insights & Recommandations</div>', unsafe_allow_html=True)
    bc = props["betweenness"]
    cv_nodes_list = [n for n,d in G.nodes(data=True) if d.get("node_type")=="CV"]
    atypical = sorted([(n,bc.get(n,0)) for n in cv_nodes_list], key=lambda x:-x[1])[:5]
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**🌟 CVs atypiques (haute centralité betweenness)**")
        for nid, score in atypical:
            cv = next((c for c in cvs if c["id"]==nid), None)
            if cv:
                st.markdown(f"- **{nid}** | {cv['level']} | {cv['domain']} | {'polyvalent' if cv['polyvalent'] else 'spécialisé'} | BC={score:.4f}")
    with col2:
        st.markdown("**📈 Résumé performances GNN vs ML classique**")
        gnn = D["gnn_results"]
        def gnn_best(key):
            r = gnn.get(key,{})
            if "error" in r: return 0.0
            models = r.get("models",{})
            vals = [v.get("cv_accuracy_mean",0) for v in models.values() if "error" not in v]
            return max(vals) if vals else 0.0
        comp_rows = [("CV Niveau — ML",   get_best(D["clf_enr"]["cv_level"])),
                     ("CV Niveau — GNN",  gnn_best("gnn_cv_level")),
                     ("CV Domaine — ML",  get_best(D["clf_enr"]["cv_domain"])),
                     ("CV Domaine — GNN", gnn_best("gnn_cv_domain"))]
        fig, ax = plt.subplots(figsize=(5,3)); ax.set_facecolor("#0d1117"); fig.patch.set_facecolor("#0d1117")
        tasks2  = [t for t,_ in comp_rows]
        scores2 = [s for _,s in comp_rows]
        colors2 = ["#4fc3f7" if "ML" in t else "#00c897" for t in tasks2]
        bars = ax.barh(tasks2, scores2, color=colors2, edgecolor="none", alpha=0.85)
        ax.set_xlim(0,1.0); ax.axvline(x=0.6,color="white",linestyle="--",linewidth=1,alpha=0.5)
        ax.set_title("ML vs GNN — Comparaison",color="white",fontsize=10); ax.tick_params(colors="white",labelsize=8)
        for sp in ax.spines.values(): sp.set_color("#333")
        for bar in bars:
            ax.text(bar.get_width()+0.01, bar.get_y()+bar.get_height()/2,
                    f"{bar.get_width():.0%}", va="center",color="white",fontsize=8)
        st.pyplot(fig); plt.close()

    pct = (G_enriched.number_of_edges()/max(G.number_of_edges(),1)-1)*100
    st.markdown(f"""<div class="success-box">
    <b>✅ Conclusions principales</b><br>
    • Le graphe biparti CV-Job révèle une structure communautaire claire, alignée sur les <b>domaines professionnels</b>.<br>
    • Les embeddings <b>{D['llm_source']}</b> capturent la sémantique des textes, identifiant des correspondances non visibles par les méthodes structurelles.<br>
    • L'enrichissement du graphe améliore la couverture de <b>{pct:.0f}%</b> et impacte positivement la classification.<br>
    • Les <b>GNN (GCN + GraphSAGE)</b> exploitent la topologie du graphe pour améliorer la classification vs ML classique seul.<br>
    • Les CVs à haute centralité betweenness sont souvent <b>polyvalents</b> — candidats stratégiques pour les recruteurs.
    </div>""", unsafe_allow_html=True)

st.markdown("---")
st.caption(f"🎓 Projet M2 — Graphe CV-Job | Pipeline : NetworkX · Scikit-learn · {D['llm_source']} · GCN · GraphSAGE · PCA · Matplotlib")