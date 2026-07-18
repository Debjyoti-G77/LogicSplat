"""
LogicSplat — Annotation Tool

Annotate spatial relations on OCTScenes by looking at the preview image
and confirming/rejecting auto-generated relations.

Run:
    streamlit run scripts/annotation_tool.py
"""
import os
import sys
import json
import math

sys.path.insert(0, ".")
import streamlit as st
import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LogicSplat Annotator",
    page_icon="🏷️",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_DIR = "D:/logicsplat_data/octscenes"
ALL_SCENES = [f"oct_{i:02d}" for i in range(1, 21) if i != 15]

AUTO_KEEP = {"left_of", "right_of"}
MANUAL_TYPES = {"on_top_of", "under", "higher_than", "lower_than",
                "in_front_of", "behind", "adjacent_to"}

INVERSE = {
    "on_top_of": "under", "under": "on_top_of",
    "higher_than": "lower_than", "lower_than": "higher_than",
    "in_front_of": "behind", "behind": "in_front_of",
    "adjacent_to": "adjacent_to",
}


# ── Data I/O ──────────────────────────────────────────────────────────────────

def load_scene(scene_id):
    path = os.path.join(DATA_DIR, scene_id, "annotation_template.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_scene(scene_id, data):
    path = os.path.join(DATA_DIR, scene_id, "annotation_template.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def is_verified(scene_id):
    d = load_scene(scene_id)
    return d is not None and d.get("annotation_status") == "verified"


def count_done():
    return sum(1 for s in ALL_SCENES if is_verified(s))


# ── Geometry helpers ──────────────────────────────────────────────────────────

def xy_dist(a, b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)


def rgb_to_hex(rgb):
    return "#{:02x}{:02x}{:02x}".format(*rgb)


# ── Re-clustering ────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Re-clustering scene...")
def recluster_scene(scene_id, n_objects):
    """
    Re-run clustering on the scene's splat.ply targeting exactly n_objects clusters.
    Returns a new objects list (same format as annotation_template.json objects).
    """
    from src.gaussian.loader import load_gaussian_ply, filter_gaussians, prune_isolated_gaussians
    from src.gaussian.clustering import gaussian_to_objects

    ply_path = os.path.join(DATA_DIR, scene_id, "splat.ply")
    if not os.path.exists(ply_path):
        return None

    cloud = load_gaussian_ply(ply_path)
    cloud = filter_gaussians(cloud, opacity_threshold=0.1)
    cloud = prune_isolated_gaussians(cloud, nb_neighbors=20, std_ratio=2.0)

    objects, params = gaussian_to_objects(
        cloud,
        target_min=n_objects,
        target_max=n_objects,
    )

    # Z-flip (same as prepare_annotation.py)
    for o in objects:
        o.centroid = o.centroid.copy()
        o.centroid[2] *= -1
        o.bbox_min = o.bbox_min.copy()
        o.bbox_min[2] *= -1
        o.bbox_max = o.bbox_max.copy()
        o.bbox_max[2] *= -1
        o.bbox_min[2], o.bbox_max[2] = min(o.bbox_min[2], o.bbox_max[2]), max(o.bbox_min[2], o.bbox_max[2])

    # Convert to JSON-compatible format
    obj_list = []
    for o in objects:
        obj_list.append({
            "id": o.uid,
            "name": f"object_{o.uid}",
            "centroid": [round(float(c), 4) for c in o.centroid],
            "point_count": o.point_count,
            "color": [int(c) for c in o.color],
        })
    return obj_list


def make_color_legend_html(objects):
    """Build an HTML color legend that maps object IDs to their colors."""
    rows = ""
    for obj in objects:
        hex_color = rgb_to_hex(obj["color"])
        name = obj["name"]
        rows += (
            f'<div style="display:flex;align-items:center;margin:4px 0;">'
            f'<div style="width:24px;height:24px;background:{hex_color};'
            f'border:2px solid #fff;border-radius:4px;margin-right:8px;'
            f'box-shadow:0 0 3px rgba(0,0,0,0.5);"></div>'
            f'<span style="font-size:14px;"><b>ID {obj["id"]}</b> = {name} '
            f'({obj["point_count"]} pts)</span>'
            f'</div>'
        )
    return f'<div style="padding:8px;background:#1e1e1e;border-radius:8px;border:1px solid #444;">{rows}</div>'


def pair_label(a, b, rel_text):
    """Plain text label for a checkbox — no HTML needed."""
    a_hex = rgb_to_hex(a["color"])
    b_hex = rgb_to_hex(b["color"])
    return f"[{a['name']}] {rel_text} [{b['name']}]"


# ── Sidebar ───────────────────────────────────────────────────────────────────

def sidebar():
    with st.sidebar:
        st.markdown("## 🏷️ Scene Annotator")

        done = count_done()
        st.markdown(f"**Progress: {done} / 19** scenes done")
        st.progress(done / 19)

        st.markdown("---")

        if "scene" not in st.session_state:
            st.session_state.scene = ALL_SCENES[0]

        idx = ALL_SCENES.index(st.session_state.scene)
        pick = st.selectbox(
            "Scene",
            ALL_SCENES,
            index=idx,
            format_func=lambda s: f"{s}  ✅" if is_verified(s) else s,
        )
        st.session_state.scene = pick

        st.markdown("---")

        if st.button("💾  Save & Next", use_container_width=True, type="primary"):
            do_save()
            go_next()
            st.rerun()

        if st.button("⏭️  Skip Scene", use_container_width=True):
            go_next()
            st.rerun()

        st.markdown("---")
        st.caption("When all 19 are done, run:")
        st.code("python scripts/finetune_tabletop.py")


def go_next():
    cur = ALL_SCENES.index(st.session_state.scene)
    for i in range(1, len(ALL_SCENES)):
        nxt = ALL_SCENES[(cur + i) % len(ALL_SCENES)]
        if not is_verified(nxt):
            st.session_state.scene = nxt
            return


# ── Save logic ────────────────────────────────────────────────────────────────

def do_save():
    scene_id = st.session_state.scene
    data = load_scene(scene_id)
    if data is None:
        return

    objects = data["objects"]
    old_relations = data.get("relations", [])

    # Keep auto left/right
    kept = [r for r in old_relations if r["relation"] in AUTO_KEEP]

    # Collect user decisions
    manual = []
    prefix = f"chk|{scene_id}|"
    for key, val in st.session_state.items():
        if not key.startswith(prefix):
            continue
        if not val:
            continue
        parts = key[len(prefix):].split("|")
        if len(parts) != 3:
            continue
        subj_id, rel_type, obj_id = int(parts[0]), parts[1], int(parts[2])
        subj_name = next((o["name"] for o in objects if o["id"] == subj_id), None)
        obj_name = next((o["name"] for o in objects if o["id"] == obj_id), None)
        if subj_name and obj_name:
            manual.append({"subject": subj_name, "relation": rel_type,
                           "object": obj_name, "auto": False, "verified": True})

    # Add inverses
    inverses = []
    for r in manual:
        inv = INVERSE.get(r["relation"])
        if inv and inv != r["relation"]:
            inverses.append({"subject": r["object"], "relation": inv,
                             "object": r["subject"], "auto": False, "verified": True})
        elif inv == r["relation"]:  # adjacent_to
            inverses.append({"subject": r["object"], "relation": r["relation"],
                             "object": r["subject"], "auto": False, "verified": True})

    # Deduplicate
    all_rels = kept + manual + inverses
    seen = set()
    final = []
    for r in all_rels:
        k = (r["subject"], r["relation"], r["object"])
        if k not in seen:
            seen.add(k)
            final.append(r)

    # Update names
    name_prefix = f"name|{scene_id}|"
    for obj in objects:
        nk = f"{name_prefix}{obj['id']}"
        if nk in st.session_state and st.session_state[nk]:
            new_name = st.session_state[nk]
            old_name = obj["name"]
            if new_name != old_name:
                for r in final:
                    if r["subject"] == old_name:
                        r["subject"] = new_name
                    if r["object"] == old_name:
                        r["object"] = new_name
                obj["name"] = new_name

    data["objects"] = objects
    data["relations"] = final
    data["annotation_status"] = "verified"
    save_scene(scene_id, data)


# ── Main content ──────────────────────────────────────────────────────────────

def main_area():
    scene_id = st.session_state.scene
    data = load_scene(scene_id)

    if data is None:
        st.error(f"No annotation data for {scene_id}")
        return

    objects = data["objects"]
    relations = data.get("relations", [])
    status = data.get("annotation_status", "?")

    # Title
    st.markdown(f"# {scene_id}")
    if status == "verified":
        st.success("Already verified — you can re-edit and save again.")
    else:
        st.info("Look at the image + color legend. Then check/uncheck relations below.")

    # ══════════════════════════════════════════════════════════════════════════
    # IMAGE + COLOR LEGEND side by side
    # ══════════════════════════════════════════════════════════════════════════
    col_img, col_legend = st.columns([3, 2])

    with col_img:
        img_path = os.path.join(DATA_DIR, scene_id, "preview.png")
        if os.path.exists(img_path):
            st.image(img_path, use_container_width=True)
        else:
            st.warning("preview.png not found")

    with col_legend:
        st.markdown("### 🎨 Color Legend")
        st.caption("Match these colors to the objects in the image:")
        st.markdown(make_color_legend_html(objects), unsafe_allow_html=True)

        # ── Re-cluster control
        st.markdown("---")
        st.markdown("**Wrong number of objects?**")
        ply_exists = os.path.exists(os.path.join(DATA_DIR, scene_id, "splat.ply"))
        if ply_exists:
            n_current = len(objects)
            n_desired = st.number_input(
                "How many objects do you see?",
                min_value=2, max_value=20, value=n_current,
                key=f"nobj|{scene_id}",
            )
            if n_desired != n_current:
                if st.button("🔄 Re-cluster", key=f"recluster|{scene_id}"):
                    new_objects = recluster_scene(scene_id, n_desired)
                    if new_objects and len(new_objects) >= 2:
                        # Update the template file
                        data["objects"] = new_objects
                        data["relations"] = []  # reset relations — need re-derivation
                        data["annotation_status"] = "re-clustered"
                        save_scene(scene_id, data)
                        st.rerun()
                    else:
                        st.error(f"Clustering produced {len(new_objects) if new_objects else 0} objects. Try a different number.")
        else:
            st.caption("(splat.ply not found — can't re-cluster)")

        # Editable names
        st.markdown("---")
        st.markdown("**Rename objects** (optional):")
        for obj in objects:
            hex_c = rgb_to_hex(obj["color"])
            st.text_input(
                f"🟢 ID {obj['id']}",
                value=obj["name"],
                key=f"name|{scene_id}|{obj['id']}",
                help=f"Color: {hex_c} | {obj['point_count']} points",
            )

    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════════════
    # RELATION TABS
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("### ✏️ Relations")
    st.markdown(
        "**✅ = relation exists** &nbsp;&nbsp; "
        "**⬜ = relation does NOT exist** &nbsp;&nbsp; "
        "Pre-filled from auto-detection. Fix mistakes."
    )

    tab_stack, tab_prox, tab_depth, tab_height, tab_lr = st.tabs([
        "🧱 Stacking", "📐 Proximity", "🔭 Depth", "📏 Height", "↔️ Left/Right"
    ])

    with tab_stack:
        st.markdown("**Question: Is A sitting on top of B?**")
        render_directed(scene_id, objects, relations, "on_top_of",
                        "is ON TOP OF",
                        filter_fn=lambda a, b: a["centroid"][2] > b["centroid"][2],
                        sort_key=lambda a, b: -(a["centroid"][2] - b["centroid"][2]),
                        extra=lambda a, b: f"height diff: {a['centroid'][2]-b['centroid'][2]:.2f}")

    with tab_prox:
        st.markdown("**Question: Are A and B touching / right next to each other?**")
        render_symmetric(scene_id, objects, relations, "adjacent_to",
                         "is ADJACENT TO",
                         sort_key=lambda a, b: xy_dist(a["centroid"], b["centroid"]),
                         extra=lambda a, b: f"distance: {xy_dist(a['centroid'], b['centroid']):.2f}")

    with tab_depth:
        st.markdown("**Question: Is A closer to the camera (in front of) B?**")
        render_directed(scene_id, objects, relations, "in_front_of",
                        "is IN FRONT OF",
                        filter_fn=lambda a, b: (b["centroid"][1] - a["centroid"][1]) > 0.05,
                        sort_key=lambda a, b: -(b["centroid"][1] - a["centroid"][1]),
                        extra=lambda a, b: f"depth diff: {b['centroid'][1]-a['centroid'][1]:.2f}")

    with tab_height:
        st.markdown("**Question: Is A higher up than B?**")
        render_directed(scene_id, objects, relations, "higher_than",
                        "is HIGHER THAN",
                        filter_fn=lambda a, b: (a["centroid"][2] - b["centroid"][2]) > 0.05,
                        sort_key=lambda a, b: -(a["centroid"][2] - b["centroid"][2]),
                        extra=lambda a, b: f"height diff: {a['centroid'][2]-b['centroid'][2]:.2f}")

    with tab_lr:
        st.markdown("**Auto-generated (not editable) — just for reference**")
        lr = [r for r in relations if r["relation"] in AUTO_KEEP]
        if not lr:
            st.write("None.")
        for r in lr:
            arrow = "⬅️" if r["relation"] == "left_of" else "➡️"
            st.write(f"{arrow}  {r['subject']}  {r['relation']}  {r['object']}")

    # Done?
    if count_done() == 19:
        st.balloons()
        st.success("🎉 All 19 scenes annotated! Run: python scripts/finetune_tabletop.py")


# ── Relation renderers ────────────────────────────────────────────────────────

def render_directed(scene_id, objects, relations, rel_type, rel_text,
                    filter_fn, sort_key, extra):
    """Checkboxes for directed relations (A rel B)."""
    pairs = []
    for a in objects:
        for b in objects:
            if a["id"] == b["id"]:
                continue
            if filter_fn(a, b):
                pairs.append((a, b))
    pairs.sort(key=lambda p: sort_key(p[0], p[1]))

    if not pairs:
        st.write("No candidate pairs.")
        return

    # Bulk buttons
    c1, c2, _ = st.columns([1, 1, 6])
    with c1:
        if st.button("✅ All", key=f"all|{scene_id}|{rel_type}"):
            for a, b in pairs:
                st.session_state[f"chk|{scene_id}|{a['id']}|{rel_type}|{b['id']}"] = True
            st.rerun()
    with c2:
        if st.button("❌ None", key=f"none|{scene_id}|{rel_type}"):
            for a, b in pairs:
                st.session_state[f"chk|{scene_id}|{a['id']}|{rel_type}|{b['id']}"] = False
            st.rerun()

    for a, b in pairs:
        key = f"chk|{scene_id}|{a['id']}|{rel_type}|{b['id']}"
        default = any(
            r["subject"] == a["name"] and r["relation"] == rel_type and r["object"] == b["name"]
            for r in relations
        )
        if key not in st.session_state:
            st.session_state[key] = default

        # Build a readable label with color squares
        a_hex = rgb_to_hex(a["color"])
        b_hex = rgb_to_hex(b["color"])
        label = f"{a['name']} {rel_text} {b['name']}"

        left, mid, right = st.columns([1, 5, 2])
        with left:
            # Show color squares as a visual hint
            st.markdown(
                f'<div style="display:flex;align-items:center;height:38px;">'
                f'<div style="width:18px;height:18px;background:{a_hex};border:1px solid #aaa;border-radius:3px;margin-right:4px;"></div>'
                f'<span style="margin:0 4px;">→</span>'
                f'<div style="width:18px;height:18px;background:{b_hex};border:1px solid #aaa;border-radius:3px;"></div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with mid:
            st.checkbox(label, key=key)
        with right:
            st.caption(extra(a, b))


def render_symmetric(scene_id, objects, relations, rel_type, rel_text,
                     sort_key, extra):
    """Checkboxes for symmetric relations (unordered pairs)."""
    pairs = []
    for i, a in enumerate(objects):
        for j, b in enumerate(objects):
            if a["id"] >= b["id"]:
                continue
            pairs.append((a, b))
    pairs.sort(key=lambda p: sort_key(p[0], p[1]))

    if not pairs:
        st.write("No candidate pairs.")
        return

    c1, c2, _ = st.columns([1, 1, 6])
    with c1:
        if st.button("✅ All", key=f"all|{scene_id}|{rel_type}"):
            for a, b in pairs:
                st.session_state[f"chk|{scene_id}|{a['id']}|{rel_type}|{b['id']}"] = True
            st.rerun()
    with c2:
        if st.button("❌ None", key=f"none|{scene_id}|{rel_type}"):
            for a, b in pairs:
                st.session_state[f"chk|{scene_id}|{a['id']}|{rel_type}|{b['id']}"] = False
            st.rerun()

    for a, b in pairs:
        key = f"chk|{scene_id}|{a['id']}|{rel_type}|{b['id']}"
        default = any(
            (r["subject"] == a["name"] and r["relation"] == rel_type and r["object"] == b["name"]) or
            (r["subject"] == b["name"] and r["relation"] == rel_type and r["object"] == a["name"])
            for r in relations
        )
        if key not in st.session_state:
            st.session_state[key] = default

        a_hex = rgb_to_hex(a["color"])
        b_hex = rgb_to_hex(b["color"])
        label = f"{a['name']} {rel_text} {b['name']}"

        left, mid, right = st.columns([1, 5, 2])
        with left:
            st.markdown(
                f'<div style="display:flex;align-items:center;height:38px;">'
                f'<div style="width:18px;height:18px;background:{a_hex};border:1px solid #aaa;border-radius:3px;margin-right:4px;"></div>'
                f'<span style="margin:0 4px;">↔</span>'
                f'<div style="width:18px;height:18px;background:{b_hex};border:1px solid #aaa;border-radius:3px;"></div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with mid:
            st.checkbox(label, key=key)
        with right:
            st.caption(extra(a, b))


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    sidebar()
    main_area()


if __name__ == "__main__":
    main()
