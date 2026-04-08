"""LogicSplat – Streamlit scene understanding app."""
import os
import streamlit as st
import numpy as np
import plotly.graph_objects as go

from src.colmap.loader import load_scene_points
from src.clustering.objects import cluster_to_objects
from src.logic.rules import build_scene_graph

# ── config ────────────────────────────────────────────────────────────────────

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "processed")
RELATION_COLORS = {
    "on_top_of":   "#00C853",
    "inside":      "#2979FF",
    "occludes":    "#FF6D00",
    "adjacent_to": "#AA00FF",
}
RELATION_ICONS = {
    "on_top_of":   "⬆️",
    "inside":      "📦",
    "occludes":    "🫣",
    "adjacent_to": "↔️",
}


# ── helpers ───────────────────────────────────────────────────────────────────

def available_scenes() -> list[str]:
    if not os.path.isdir(DATA_DIR):
        return []
    return sorted([
        d for d in os.listdir(DATA_DIR)
        if os.path.isdir(os.path.join(DATA_DIR, d, "colmap", "sparse", "0"))
        and os.path.exists(os.path.join(DATA_DIR, d, "colmap", "sparse", "0", "points3D.bin"))
    ])


@st.cache_data(show_spinner=False)
def run_pipeline(scene_id: str, min_cluster_size, min_samples: int, sat_threshold):
    scene_path = os.path.join(DATA_DIR, scene_id)
    points, colors = load_scene_points(scene_path)
    objects, params = cluster_to_objects(
        points, colors,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        sat_threshold=sat_threshold,
        infer_table=True,
    )
    graph = build_scene_graph(scene_id, objects)
    return points, colors, graph, params


def point_cloud_fig(points: np.ndarray, colors: np.ndarray, objects) -> go.Figure:
    r, g, b = colors[:, 0], colors[:, 1], colors[:, 2]
    color_strs = [f"rgb({ri},{gi},{bi})" for ri, gi, bi in zip(r, g, b)]

    fig = go.Figure()

    # raw point cloud
    fig.add_trace(go.Scatter3d(
        x=points[:, 0], y=points[:, 1], z=points[:, 2],
        mode="markers",
        marker=dict(size=1.2, color=color_strs, opacity=0.4),
        name="Point Cloud",
        hoverinfo="skip",
    ))

    # cluster centroids
    cx = [o.centroid[0] for o in objects]
    cy = [o.centroid[1] for o in objects]
    cz = [o.centroid[2] for o in objects]
    obj_colors = [f"rgb({o.color[0]},{o.color[1]},{o.color[2]})" for o in objects]
    labels = [f"Object_{o.uid}<br>pts={o.point_count}" for o in objects]

    fig.add_trace(go.Scatter3d(
        x=cx, y=cy, z=cz,
        mode="markers+text",
        marker=dict(size=8, color=obj_colors, line=dict(width=2, color="white")),
        text=[f"Obj {o.uid}" for o in objects],
        textposition="top center",
        hovertext=labels,
        hoverinfo="text",
        name="Objects",
    ))

    fig.update_layout(
        scene=dict(
            xaxis_title="X", yaxis_title="Y", zaxis_title="Z",
            bgcolor="#0e1117",
            xaxis=dict(gridcolor="#333"), yaxis=dict(gridcolor="#333"), zaxis=dict(gridcolor="#333"),
        ),
        paper_bgcolor="#0e1117",
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(bgcolor="#0e1117", font=dict(color="white")),
        height=520,
    )
    return fig


def relation_graph_fig(graph) -> go.Figure:
    """Simple 2D spring-layout graph of the scene graph relations."""
    import math
    objects = graph.objects
    n = len(objects)
    if n == 0:
        return go.Figure()

    # circular layout
    angles = [2 * math.pi * i / n for i in range(n)]
    pos = {o.uid: (math.cos(a), math.sin(a)) for o, a in zip(objects, angles)}

    fig = go.Figure()

    # edges
    for rel in graph.relations:
        x0, y0 = pos[rel.subject_id]
        x1, y1 = pos[rel.object_id]
        color = RELATION_COLORS.get(rel.relation, "#888")
        fig.add_trace(go.Scatter(
            x=[x0, x1, None], y=[y0, y1, None],
            mode="lines",
            line=dict(width=2, color=color),
            name=rel.relation,
            showlegend=False,
            hoverinfo="skip",
        ))

    # nodes
    nx_ = [pos[o.uid][0] for o in objects]
    ny_ = [pos[o.uid][1] for o in objects]
    node_colors = [f"rgb({o.color[0]},{o.color[1]},{o.color[2]})" for o in objects]
    fig.add_trace(go.Scatter(
        x=nx_, y=ny_,
        mode="markers+text",
        marker=dict(size=22, color=node_colors, line=dict(width=2, color="white")),
        text=[f"Obj {o.uid}" for o in objects],
        textposition="middle center",
        textfont=dict(color="white", size=11),
        hoverinfo="skip",
        showlegend=False,
    ))

    # legend entries for relation types present
    seen = set()
    for rel in graph.relations:
        if rel.relation not in seen:
            seen.add(rel.relation)
            color = RELATION_COLORS.get(rel.relation, "#888")
            icon  = RELATION_ICONS.get(rel.relation, "")
            fig.add_trace(go.Scatter(
                x=[None], y=[None], mode="lines",
                line=dict(width=3, color=color),
                name=f"{icon} {rel.relation}",
            ))

    fig.update_layout(
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(bgcolor="#0e1117", font=dict(color="white")),
        height=420,
    )
    return fig


# ── UI ────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="LogicSplat", page_icon="🧠", layout="wide")

st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; }
    .metric-card {
        background: #1e1e2e; border-radius: 10px;
        padding: 1rem 1.5rem; margin-bottom: 0.5rem;
    }
    .rel-pill {
        display: inline-block; border-radius: 20px;
        padding: 3px 12px; margin: 3px; font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

st.title("🧠 LogicSplat")
st.caption("3D Scene Understanding from Monocular Video via COLMAP + Geometric Reasoning")

scenes = available_scenes()
if not scenes:
    st.error("No scenes with COLMAP data found in data/processed/")
    st.stop()

# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    scene_id = st.selectbox("Scene", scenes)
    st.divider()
    st.subheader("Clustering")
    auto_mode = st.toggle("Auto-tune parameters", value=True,
                          help="HDBSCAN auto-detects cluster scale. No eps needed.")
    if auto_mode:
        min_cluster_size = None  # auto-computed inside cluster_to_objects as pts/30
        min_samples = 3
        sat_threshold = None
    else:
        min_cluster_size = st.slider("Min cluster size", 3, 30, 5,
                                     help="Minimum points to form an object")
        min_samples = st.slider("Min samples", 1, 10, 3,
                                help="Controls noise sensitivity")
        sat_threshold = st.slider("Background saturation filter", 0, 80, 20,
                                  help="Strip white/grey background points")
    st.divider()
    run = st.button("▶ Run Analysis", use_container_width=True, type="primary")

# ── main ──────────────────────────────────────────────────────────────────────
if run or "graph" in st.session_state:
    if run:
        with st.spinner("Loading point cloud and running pipeline..."):
            try:
                points, colors, graph, params = run_pipeline(scene_id, min_cluster_size, min_samples, sat_threshold)
                st.session_state["graph"]   = graph
                st.session_state["points"]  = points
                st.session_state["colors"]  = colors
                st.session_state["params"]  = params
                st.session_state["scene_id"] = scene_id
            except FileNotFoundError as e:
                st.error(str(e))
                st.stop()

    graph  = st.session_state["graph"]
    points = st.session_state["points"]
    colors = st.session_state["colors"]
    params = st.session_state.get("params", {})

    # ── metrics row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Points",    f"{len(points):,}")
    c2.metric("Objects Found",   len(graph.objects))
    c3.metric("Relations Found", len(graph.relations))
    c4.metric("Scene",           graph.scene_id)

    if params:
        with st.expander("🔧 Auto-detected parameters", expanded=False):
            p1, p2, p3, p4 = st.columns(4)
            p1.metric("min_cluster_size", params.get("min_cluster_size", "—"))
            p2.metric("min_samples",      params.get("min_samples", "—"))
            p3.metric("sat_threshold",    params.get("sat_threshold", "—"))
            p4.metric("Points (filtered)", f"{params.get('points_after_filter', 0):,}")

    st.divider()

    # ── tabs
    tab1, tab2, tab3 = st.tabs(["🌐 3D Point Cloud", "🕸️ Scene Graph", "📝 Explanation"])

    with tab1:
        st.plotly_chart(
            point_cloud_fig(points, colors, graph.objects),
            use_container_width=True,
        )
        if graph.objects:
            st.subheader("Discovered Objects")
            cols = st.columns(min(len(graph.objects), 4))
            for i, obj in enumerate(graph.objects):
                with cols[i % 4]:
                    r, g, b = int(obj.color[0]), int(obj.color[1]), int(obj.color[2])
                    size = obj.size
                    st.markdown(f"""
                    <div class="metric-card">
                        <b style="color:rgb({r},{g},{b})">● Object {obj.uid}</b><br>
                        <small>
                        Points: {obj.point_count}<br>
                        Centroid: ({obj.centroid[0]:.2f}, {obj.centroid[1]:.2f}, {obj.centroid[2]:.2f})<br>
                        Size: {size[0]:.2f} × {size[1]:.2f} × {size[2]:.2f}
                        </small>
                    </div>
                    """, unsafe_allow_html=True)

    with tab2:
        if not graph.relations:
            st.info("No relations found. Try adjusting the DBSCAN parameters.")
        else:
            st.plotly_chart(relation_graph_fig(graph), use_container_width=True)

            st.subheader("All Relations")
            for rel_type, color in RELATION_COLORS.items():
                rels = [r for r in graph.relations if r.relation == rel_type]
                if not rels:
                    continue
                icon = RELATION_ICONS.get(rel_type, "")
                st.markdown(f"**{icon} {rel_type}** ({len(rels)})")
                for r in rels:
                    st.markdown(
                        f'<span class="rel-pill" style="background:{color}22;border:1px solid {color}">'
                        f'Object_{r.subject_id} → Object_{r.object_id}</span>',
                        unsafe_allow_html=True,
                    )

    with tab3:
        st.subheader("Natural Language Scene Description")
        if not graph.relations:
            st.info("No relations to describe.")
        else:
            for rel in graph.relations:
                icon = RELATION_ICONS.get(rel.relation, "•")
                st.write(f"{icon} {rel.to_text(graph.objects)}")

else:
    st.info("👈 Select a scene and click **Run Analysis** to begin.")
    st.markdown("""
    **Pipeline:**
    1. Load COLMAP sparse point cloud
    2. Cluster points with DBSCAN → discover objects
    3. Infer spatial relations (support, containment, occlusion, adjacency)
    4. Build scene graph + natural language explanation
    """)
