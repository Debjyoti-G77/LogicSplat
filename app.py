"""
LogicSplat — Streamlit App (TASK 4)

Pipeline:
  1. Select or upload a scene (splat.ply)
  2. Load & filter Gaussians → show 3D point cloud
  3. Run YOLO labeling to name objects
  4. Run GNN relation inference
  5. Show scene graph (nodes + edges)
  6. Show natural language description

Run:
    streamlit run app.py
"""
import os
import sys
import json
import glob
import tempfile
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

import numpy as np
import streamlit as st
import plotly.graph_objects as go

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LogicSplat",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── constants ─────────────────────────────────────────────────────────────────
DATA_DIR   = "data/processed"
MODELS_DIR = "models"
RELATION_COLORS = {
    "on_top_of":   "#2ecc71",
    "under":       "#27ae60",
    "inside":      "#3498db",
    "attached_to": "#9b59b6",
    "hanging_from":"#8e44ad",
    "adjacent_to": "#f39c12",
    "surrounding":  "#e67e22",
    "left_of":     "#e74c3c",
    "right_of":    "#c0392b",
    "in_front_of": "#1abc9c",
    "behind":      "#16a085",
    "higher_than": "#f1c40f",
    "lower_than":  "#d4ac0d",
    "occludes":    "#95a5a6",
}


# ── helpers ───────────────────────────────────────────────────────────────────

@st.cache_resource
def _load_model(model_path: str, node_dim: int = 8, edge_dim: int = 4):
    """Load RelationGNN — cached so it stays in memory across reruns."""
    import torch
    from src.models.relation_gnn import RelationGNN
    from src.relations.schema import NUM_RELATIONS
    model = RelationGNN(
        node_feat_dim=node_dim,
        edge_feat_dim=edge_dim,
        hidden_dim=128,
        num_relations=NUM_RELATIONS,
        dropout=0.0,
    )
    model.load_state_dict(
        torch.load(model_path, map_location="cpu", weights_only=True)
    )
    model.eval()
    return model


def _discover_scenes() -> list:
    """Return list of scene IDs that have a splat.ply."""
    scenes = []
    if os.path.isdir(DATA_DIR):
        for name in sorted(os.listdir(DATA_DIR)):
            ply = os.path.join(DATA_DIR, name, "splat.ply")
            if os.path.exists(ply):
                scenes.append(name)
    return scenes


def _discover_models() -> dict:
    """Return {display_name: path} for all .pt files in models/."""
    models = {}
    if os.path.isdir(MODELS_DIR):
        for pt in sorted(glob.glob(os.path.join(MODELS_DIR, "relation_gnn_*.pt"))):
            name = os.path.splitext(os.path.basename(pt))[0]
            name = name.replace("relation_gnn_", "")
            models[name] = pt
    return models


def _infer_model_dims(model_path: str):
    """Auto-detect node/edge feature dims from saved weights."""
    import torch
    sd = torch.load(model_path, map_location="cpu", weights_only=True)
    node_dim, edge_dim, hidden_dim = 8, 4, 128
    for k, v in sd.items():
        if "node_encoder.0.weight" in k:
            node_dim  = v.shape[1]
            hidden_dim = v.shape[0]
        if "edge_classifier.0.weight" in k:
            edge_dim = v.shape[1] - 2 * hidden_dim
    return node_dim, edge_dim


@st.cache_data(show_spinner=False)
def _load_and_cluster(splat_path: str, opacity_thresh: float):
    """Load splat.ply, filter, cluster → objects + params. Cached by path."""
    from src.gaussian.loader import load_gaussian_ply, filter_gaussians
    from src.gaussian.clustering import gaussian_to_objects
    cloud = load_gaussian_ply(splat_path)
    filtered = filter_gaussians(cloud, opacity_threshold=opacity_thresh)
    objects, params = gaussian_to_objects(filtered)
    # return serialisable data (numpy arrays are fine for st.cache_data)
    return objects, params, filtered.xyz, filtered.rgb


def _build_full_graph(objects):
    """
    Build a complete directed graph over all object pairs.
    Inlined here to avoid the src.colmap.loader import in predict.py.
    """
    import torch

    def _node_feat(obj):
        size = obj.size
        vol = max(obj.volume, 1e-6)
        vol_norm = min(vol / 10.0, 1.0)
        height_ratio = size[2] / max(max(size[0], size[1]), 1e-6)
        is_flat = 1.0 if height_ratio < 0.3 else 0.0
        is_tall = 1.0 if height_ratio > 2.0 else 0.0
        pts_per_vol = obj.point_count / vol
        is_hollow = 1.0 if pts_per_vol < 10 else 0.0
        return torch.tensor(
            [vol_norm, 1.0, is_flat, is_tall, 1.0, 0.0, 0.0, is_hollow],
            dtype=torch.float32,
        )

    def _edge_feat(a, b):
        vol_a = max(a.volume, 1e-6)
        vol_b = max(b.volume, 1e-6)
        size_sim = min(vol_a, vol_b) / max(vol_a, vol_b)
        size_a_norm = min(vol_a / 10.0, 1.0)
        size_b_norm = min(vol_b / 10.0, 1.0)
        same_size = 1.0 if size_sim > 0.8 else 0.0
        return torch.tensor(
            [same_size, size_a_norm, size_b_norm, size_sim],
            dtype=torch.float32,
        )

    n = len(objects)
    src, dst, edge_attrs = [], [], []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            src.append(i)
            dst.append(j)
            edge_attrs.append(_edge_feat(objects[i], objects[j]))

    x = torch.stack([_node_feat(o) for o in objects])
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_attr  = torch.stack(edge_attrs)
    return x, edge_index, edge_attr


def _run_gnn(objects, model_path: str, threshold: float):
    """Run GNN on objects → list of relation dicts."""
    import torch
    from src.relations.schema import RELATION_NAMES, RELATION_DESCRIPTIONS, Relation

    node_dim, edge_dim = _infer_model_dims(model_path)
    model = _load_model(model_path, node_dim, edge_dim)

    x, edge_index, edge_attr = _build_full_graph(objects)

    # pad features if model expects more dims than the 8/4 defaults
    import torch.nn.functional as F
    if x.shape[1] < node_dim:
        x = F.pad(x, (0, node_dim - x.shape[1]))
    if edge_attr.shape[1] < edge_dim:
        edge_attr = F.pad(edge_attr, (0, edge_dim - edge_attr.shape[1]))
    with torch.no_grad():
        logits = model(x, edge_index, edge_attr)
        probs  = torch.softmax(logits, dim=-1)
        preds  = logits.argmax(dim=-1)

    src_nodes = edge_index[0].tolist()
    dst_nodes = edge_index[1].tolist()
    relations = []
    seen = set()
    for s, d, pred, prob in zip(src_nodes, dst_nodes, preds.tolist(), probs):
        conf = float(prob[pred])
        if conf < threshold:
            continue
        key = (s, d, pred)
        if key in seen:
            continue
        seen.add(key)
        rel_name = RELATION_NAMES[pred]
        relations.append({
            "subject_id":   s,
            "object_id":    d,
            "relation":     rel_name,
            "confidence":   conf,
            "description":  RELATION_DESCRIPTIONS.get(Relation(pred), rel_name),
        })
    return relations


def _run_rules(objects):
    """Run geometric rule baseline → list of relation dicts."""
    from src.logic.rules import infer_relations
    raw = infer_relations(objects)
    return [
        {
            "subject_id":  r.subject_id,
            "object_id":   r.object_id,
            "relation":    r.relation,
            "confidence":  1.0,
            "description": r.relation.replace("_", " "),
        }
        for r in raw
    ]


def _natural_language(objects, relations) -> str:
    """Generate a short natural-language scene description."""
    if not objects:
        return "No objects detected."
    obj_names = {o.uid: o.label for o in objects}
    lines = [f"The scene contains {len(objects)} object(s): "
             + ", ".join(f"**{o.label}**" for o in objects) + "."]
    if relations:
        lines.append("")
        for r in relations[:12]:  # cap at 12 sentences
            s = obj_names.get(r["subject_id"], f"Object {r['subject_id']}")
            o = obj_names.get(r["object_id"],  f"Object {r['object_id']}")
            lines.append(f"- The **{s}** {r['description']} the **{o}**.")
        if len(relations) > 12:
            lines.append(f"- *(and {len(relations) - 12} more relations…)*")
    else:
        lines.append("No spatial relations were detected above the confidence threshold.")
    return "\n".join(lines)


# ── 3D visualisations ─────────────────────────────────────────────────────────

def _fig_point_cloud(xyz: np.ndarray, rgb: np.ndarray, objects) -> go.Figure:
    """Plotly 3D scatter of the Gaussian point cloud with cluster colours."""
    # subsample for performance
    N = len(xyz)
    max_pts = 50_000
    if N > max_pts:
        idx = np.random.choice(N, max_pts, replace=False)
        xyz_s, rgb_s = xyz[idx], rgb[idx]
    else:
        xyz_s, rgb_s = xyz, rgb

    colors = [f"rgb({r},{g},{b})" for r, g, b in rgb_s]

    fig = go.Figure(go.Scatter3d(
        x=xyz_s[:, 0], y=xyz_s[:, 1], z=xyz_s[:, 2],
        mode="markers",
        marker=dict(size=1.5, color=colors, opacity=0.7),
        name="Gaussians",
        hoverinfo="skip",
    ))

    # overlay cluster centroids
    palette = [
        "#e74c3c","#3498db","#2ecc71","#f39c12","#9b59b6",
        "#1abc9c","#e67e22","#34495e","#e91e63","#00bcd4",
    ]
    for i, obj in enumerate(objects):
        c = palette[i % len(palette)]
        cx, cy, cz = obj.centroid
        fig.add_trace(go.Scatter3d(
            x=[cx], y=[cy], z=[cz],
            mode="markers+text",
            marker=dict(size=8, color=c, symbol="diamond",
                        line=dict(color="white", width=1)),
            text=[f"{obj.label} ({obj.uid})"],
            textposition="top center",
            name=f"{obj.label} ({obj.uid})",
        ))

    fig.update_layout(
        scene=dict(
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(showgrid=False, zeroline=False),
            zaxis=dict(showgrid=False, zeroline=False),
            bgcolor="#0e1117",
        ),
        paper_bgcolor="#0e1117",
        font_color="white",
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(bgcolor="rgba(0,0,0,0.4)", font_size=11),
        height=520,
    )
    return fig


def _fig_scene_graph(objects, relations) -> go.Figure:
    """Plotly 2D scene graph: objects as nodes, relations as labelled edges."""
    if not objects:
        return go.Figure()

    # layout: arrange objects in a circle
    n = len(objects)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pos = {obj.uid: (np.cos(a), np.sin(a)) for obj, a in zip(objects, angles)}
    palette = [
        "#e74c3c","#3498db","#2ecc71","#f39c12","#9b59b6",
        "#1abc9c","#e67e22","#34495e","#e91e63","#00bcd4",
    ]
    uid_to_color = {obj.uid: palette[i % len(palette)]
                    for i, obj in enumerate(objects)}

    fig = go.Figure()

    # edges
    for r in relations:
        if r["subject_id"] not in pos or r["object_id"] not in pos:
            continue
        x0, y0 = pos[r["subject_id"]]
        x1, y1 = pos[r["object_id"]]
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        rel_color = RELATION_COLORS.get(r["relation"], "#aaaaaa")
        fig.add_trace(go.Scatter(
            x=[x0, x1, None], y=[y0, y1, None],
            mode="lines",
            line=dict(color=rel_color, width=max(1, int(r["confidence"] * 4))),
            hoverinfo="skip",
            showlegend=False,
        ))
        # relation label at midpoint
        fig.add_trace(go.Scatter(
            x=[mx], y=[my],
            mode="text",
            text=[f"<b>{r['relation']}</b><br>{r['confidence']:.2f}"],
            textfont=dict(size=9, color=rel_color),
            hoverinfo="skip",
            showlegend=False,
        ))

    # nodes
    for obj in objects:
        x, y = pos[obj.uid]
        c = uid_to_color[obj.uid]
        fig.add_trace(go.Scatter(
            x=[x], y=[y],
            mode="markers+text",
            marker=dict(size=28, color=c,
                        line=dict(color="white", width=2)),
            text=[f"<b>{obj.label}</b><br>({obj.uid})"],
            textposition="middle center",
            textfont=dict(size=10, color="white"),
            name=obj.label,
            hovertemplate=(
                f"<b>{obj.label}</b><br>"
                f"uid={obj.uid}<br>"
                f"pts={obj.point_count}<br>"
                f"centroid=({obj.centroid[0]:.2f}, "
                f"{obj.centroid[1]:.2f}, {obj.centroid[2]:.2f})"
                "<extra></extra>"
            ),
        ))

    fig.update_layout(
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                   range=[-1.6, 1.6]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                   range=[-1.6, 1.6]),
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        font_color="white",
        margin=dict(l=0, r=0, t=0, b=0),
        height=480,
        showlegend=False,
    )
    return fig


# ── sidebar ───────────────────────────────────────────────────────────────────

def _sidebar():
    st.sidebar.image(
        "https://raw.githubusercontent.com/streamlit/streamlit/develop/lib/streamlit/static/favicon.png",
        width=32,
    )
    st.sidebar.title("LogicSplat 🔮")
    st.sidebar.caption("3D Scene Graph from Monocular Video")
    st.sidebar.divider()

    # ── scene source ──────────────────────────────────────────────────────────
    st.sidebar.subheader("Scene")
    source = st.sidebar.radio(
        "Source", ["Existing scene", "Upload splat.ply"], label_visibility="collapsed"
    )

    splat_path = None
    scene_id   = None

    if source == "Existing scene":
        scenes = _discover_scenes()
        if scenes:
            scene_id = st.sidebar.selectbox("Select scene", scenes)
            splat_path = os.path.join(DATA_DIR, scene_id, "splat.ply")
        else:
            st.sidebar.warning("No scenes found in data/processed/")
    else:
        uploaded = st.sidebar.file_uploader("Upload splat.ply", type=["ply"])
        if uploaded:
            tmp = tempfile.NamedTemporaryFile(suffix=".ply", delete=False)
            tmp.write(uploaded.read())
            tmp.flush()
            splat_path = tmp.name
            scene_id   = uploaded.name.replace(".ply", "")

    st.sidebar.divider()

    # ── clustering params ─────────────────────────────────────────────────────
    st.sidebar.subheader("Clustering")
    opacity_thresh = st.sidebar.slider(
        "Opacity threshold", 0.01, 0.5, 0.1, 0.01,
        help="Remove Gaussians below this opacity (background filter)"
    )

    st.sidebar.divider()

    # ── model selection ───────────────────────────────────────────────────────
    st.sidebar.subheader("GNN Model")
    models = _discover_models()
    if models:
        model_name = st.sidebar.selectbox("Select model", list(models.keys()))
        model_path = models[model_name]
        node_dim, edge_dim = _infer_model_dims(model_path)
        st.sidebar.caption(f"node_dim={node_dim}  edge_dim={edge_dim}")
    else:
        st.sidebar.warning("No trained models found in models/")
        model_path = None

    gnn_threshold = st.sidebar.slider(
        "Confidence threshold", 0.1, 0.9, 0.35, 0.05,
        help="Only show relations above this softmax confidence"
    )

    st.sidebar.divider()

    # ── YOLO labeling ─────────────────────────────────────────────────────────
    st.sidebar.subheader("YOLO Labeling")
    run_yolo = st.sidebar.checkbox(
        "Run YOLO labeling", value=False,
        help="Projects cluster centroids into video frames and votes on labels. "
             "Requires transforms.json and images in the scene folder."
    )
    n_frames = st.sidebar.slider("Frames to sample", 5, 60, 20, 5,
                                  disabled=not run_yolo)

    st.sidebar.divider()

    # ── inference mode ────────────────────────────────────────────────────────
    st.sidebar.subheader("Inference")
    inference_mode = st.sidebar.radio(
        "Mode",
        ["GNN only", "Rules only", "Ensemble (rules + GNN)"],
        index=0,
    )

    return dict(
        splat_path=splat_path,
        scene_id=scene_id,
        opacity_thresh=opacity_thresh,
        model_path=model_path,
        gnn_threshold=gnn_threshold,
        run_yolo=run_yolo,
        n_frames=n_frames,
        inference_mode=inference_mode,
    )


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    cfg = _sidebar()

    st.title("LogicSplat — 3D Scene Graph Generation")

    if cfg["splat_path"] is None:
        st.info("👈 Select a scene or upload a splat.ply to get started.")
        return

    if not os.path.exists(cfg["splat_path"]):
        st.error(f"splat.ply not found: `{cfg['splat_path']}`")
        return

    # ── run button ────────────────────────────────────────────────────────────
    run_btn = st.button("▶  Run Pipeline", type="primary", use_container_width=True)

    # keep results in session state so they survive sidebar interactions
    if run_btn:
        st.session_state.pop("results", None)

    if run_btn or "results" in st.session_state:

        if "results" not in st.session_state or run_btn:
            with st.spinner("Loading & clustering Gaussians…"):
                try:
                    objects, params, xyz, rgb = _load_and_cluster(
                        cfg["splat_path"], cfg["opacity_thresh"]
                    )
                except Exception as e:
                    st.error(f"Failed to load splat.ply: {e}")
                    return

            # ── YOLO labeling ─────────────────────────────────────────────────
            if cfg["run_yolo"] and cfg["scene_id"]:
                transforms_path = os.path.join(
                    DATA_DIR, cfg["scene_id"], "ns_data", "transforms.json"
                )
                images_dir = os.path.join(
                    DATA_DIR, cfg["scene_id"], "ns_data", "images"
                )
                if os.path.exists(transforms_path) and os.path.isdir(images_dir):
                    with st.spinner("Running YOLO labeling…"):
                        try:
                            from src.labeling.yolo_labeler import label_objects_with_yolo
                            scene_dir = os.path.join(DATA_DIR, cfg["scene_id"])
                            objects = label_objects_with_yolo(
                                objects,
                                transforms_path,
                                images_dir,
                                n_frames=cfg["n_frames"],
                                scene_dir=scene_dir,
                            )
                        except Exception as e:
                            st.warning(f"YOLO labeling failed: {e}")
                else:
                    st.warning(
                        "YOLO labeling skipped — "
                        f"`transforms.json` or `images/` not found in "
                        f"`{DATA_DIR}/{cfg['scene_id']}/ns_data/`"
                    )

            # ── relation inference ────────────────────────────────────────────
            relations = []
            mode = cfg["inference_mode"]

            if mode in ("GNN only", "Ensemble (rules + GNN)"):
                if cfg["model_path"] and os.path.exists(cfg["model_path"]):
                    with st.spinner("Running GNN inference…"):
                        try:
                            gnn_rels = _run_gnn(
                                objects, cfg["model_path"], cfg["gnn_threshold"]
                            )
                        except Exception as e:
                            st.warning(f"GNN inference failed: {e}")
                            gnn_rels = []
                else:
                    st.warning("No model selected — skipping GNN.")
                    gnn_rels = []

            if mode in ("Rules only", "Ensemble (rules + GNN)"):
                with st.spinner("Running geometric rules…"):
                    try:
                        rule_rels = _run_rules(objects)
                    except Exception as e:
                        st.warning(f"Rules failed: {e}")
                        rule_rels = []

            if mode == "GNN only":
                relations = gnn_rels
            elif mode == "Rules only":
                relations = rule_rels
            else:
                # ensemble: merge, deduplicate, prefer rules for vertical rels
                rule_dominant = {"on_top_of", "under", "inside", "hanging_from"}
                seen = set()
                merged = []
                for r in rule_rels:
                    key = (r["subject_id"], r["object_id"], r["relation"])
                    if key not in seen:
                        seen.add(key)
                        merged.append({**r, "source": "rules"})
                for r in gnn_rels:
                    key = (r["subject_id"], r["object_id"], r["relation"])
                    if key not in seen:
                        if r["relation"] not in rule_dominant:
                            seen.add(key)
                            merged.append({**r, "source": "gnn"})
                relations = merged

            st.session_state["results"] = dict(
                objects=objects,
                params=params,
                xyz=xyz,
                rgb=rgb,
                relations=relations,
            )

        # ── render results ────────────────────────────────────────────────────
        res = st.session_state["results"]
        objects   = res["objects"]
        params    = res["params"]
        xyz       = res["xyz"]
        rgb       = res["rgb"]
        relations = res["relations"]

        # ── metrics row ───────────────────────────────────────────────────────
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Gaussians (raw)",     f"{params.get('n_gaussians_raw', 0):,}")
        c2.metric("After filter",        f"{params.get('n_after_filter', 0):,}")
        c3.metric("Objects found",       len(objects))
        c4.metric("Relations predicted", len(relations))

        st.divider()

        # ── tabs ──────────────────────────────────────────────────────────────
        tab_cloud, tab_graph, tab_nlp, tab_raw = st.tabs([
            "☁ Point Cloud", "🕸 Scene Graph", "📝 Description", "📊 Raw Data"
        ])

        with tab_cloud:
            st.subheader("Gaussian Point Cloud")
            st.caption(
                f"Opacity threshold: {params.get('sat_threshold', '—')}  |  "
                f"min_cluster_size: {params.get('min_cluster_size', '—')}  |  "
                f"noise fraction: {params.get('noise_fraction', '—')}"
            )
            if len(xyz) > 0:
                fig_cloud = _fig_point_cloud(xyz, rgb, objects)
                st.plotly_chart(fig_cloud, use_container_width=True)
            else:
                st.warning("No Gaussians remain after filtering. "
                           "Try lowering the opacity threshold.")

            # object table
            if objects:
                st.subheader("Detected Objects")
                rows = []
                for o in objects:
                    rows.append({
                        "ID":    o.uid,
                        "Label": o.label,
                        "Points": o.point_count,
                        "Centroid X": round(float(o.centroid[0]), 3),
                        "Centroid Y": round(float(o.centroid[1]), 3),
                        "Centroid Z": round(float(o.centroid[2]), 3),
                        "Size X": round(float(o.size[0]), 3),
                        "Size Y": round(float(o.size[1]), 3),
                        "Size Z": round(float(o.size[2]), 3),
                    })
                st.dataframe(rows, use_container_width=True, hide_index=True)

        with tab_graph:
            st.subheader("Scene Graph")
            if objects and relations:
                fig_graph = _fig_scene_graph(objects, relations)
                st.plotly_chart(fig_graph, use_container_width=True)

                # relation table
                st.subheader("Relations")
                rel_rows = []
                uid_to_label = {o.uid: o.label for o in objects}
                for r in sorted(relations, key=lambda x: -x["confidence"]):
                    rel_rows.append({
                        "Subject": uid_to_label.get(r["subject_id"],
                                                     f"Obj {r['subject_id']}"),
                        "Relation": r["relation"],
                        "Object":  uid_to_label.get(r["object_id"],
                                                     f"Obj {r['object_id']}"),
                        "Confidence": round(r["confidence"], 3),
                        "Source": r.get("source", cfg["inference_mode"]),
                    })
                st.dataframe(rel_rows, use_container_width=True, hide_index=True)
            elif objects and not relations:
                st.info("No relations found above the confidence threshold. "
                        "Try lowering the threshold in the sidebar.")
                fig_graph = _fig_scene_graph(objects, [])
                st.plotly_chart(fig_graph, use_container_width=True)
            else:
                st.warning("No objects detected.")

        with tab_nlp:
            st.subheader("Natural Language Description")
            desc = _natural_language(objects, relations)
            st.markdown(desc)

            if relations:
                st.divider()
                st.subheader("Relation breakdown by type")
                from collections import Counter
                counts = Counter(r["relation"] for r in relations)
                rel_names = list(counts.keys())
                rel_vals  = [counts[k] for k in rel_names]
                fig_bar = go.Figure(go.Bar(
                    x=rel_names, y=rel_vals,
                    marker_color=[RELATION_COLORS.get(r, "#aaaaaa") for r in rel_names],
                    text=rel_vals, textposition="outside",
                ))
                fig_bar.update_layout(
                    paper_bgcolor="#0e1117",
                    plot_bgcolor="#0e1117",
                    font_color="white",
                    xaxis_tickangle=-35,
                    margin=dict(l=0, r=0, t=20, b=0),
                    height=320,
                    showlegend=False,
                )
                st.plotly_chart(fig_bar, use_container_width=True)

        with tab_raw:
            st.subheader("Clustering Parameters")
            st.json(params)

            st.subheader("Objects (JSON)")
            obj_data = []
            for o in objects:
                obj_data.append({
                    "uid":      o.uid,
                    "label":    o.label,
                    "centroid": o.centroid.tolist(),
                    "bbox_min": o.bbox_min.tolist(),
                    "bbox_max": o.bbox_max.tolist(),
                    "size":     o.size.tolist(),
                    "volume":   round(o.volume, 5),
                    "point_count": o.point_count,
                })
            st.json(obj_data)

            st.subheader("Relations (JSON)")
            st.json(relations)

            # download button
            export = json.dumps(
                {"objects": obj_data, "relations": relations}, indent=2
            )
            st.download_button(
                "⬇ Download scene graph JSON",
                data=export,
                file_name=f"{cfg.get('scene_id', 'scene')}_graph.json",
                mime="application/json",
            )


if __name__ == "__main__":
    main()
