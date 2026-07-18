"""
LogicSplat Live -- optional live-inference server.

Exposes GET /predict/<scene_id>, which re-runs the exact same precompute.py
scene function (same model, same clustering, same repair) on demand and
returns the same JSON shape the frontend already reads from the cached
demo/data/<scene_id>.json. If this server isn't running (or is slow), the
UI silently falls back to the cached files -- this server is a bonus, not
a requirement.

Run: python demo/server.py
"""
import sys
import os
_DEMO_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_DEMO_DIR)
sys.path.insert(0, _DEMO_DIR)
sys.path.insert(0, _PROJECT_ROOT)
os.chdir(_PROJECT_ROOT)  # eval_geokan_tabletop.py loads models/*.pt via relative paths

import json
from flask import Flask, jsonify, Response

import precompute

app = Flask(__name__)

SCENE_LOOKUP = {}
for sid, cfg in precompute.TABLETOP_SCENES.items():
    SCENE_LOOKUP[sid] = ("tabletop", cfg)
for sid, cfg in precompute.LERF_SCENES.items():
    SCENE_LOOKUP[sid] = ("lerf", cfg)


@app.after_request
def add_cors(resp):
    resp.headers["Access-Control-Allow-Origin"] = "*"
    return resp


@app.route("/predict/<scene_id>")
def predict(scene_id):
    if scene_id not in SCENE_LOOKUP:
        return jsonify({"error": "unknown scene_id"}), 404
    kind, cfg = SCENE_LOOKUP[scene_id]
    try:
        if kind == "tabletop":
            precompute.process_tabletop_scene(scene_id, cfg)
        else:
            precompute.process_lerf_scene(scene_id, cfg)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    out_path = os.path.join(precompute.OUT_DIR, f"{scene_id}.json")
    with open(out_path) as f:
        data = json.load(f)
    return Response(json.dumps(data), mimetype="application/json")


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    print("LogicSplat Live server starting on http://localhost:8731")
    print("(this is optional -- the demo works fully from cached data without it)")
    app.run(host="127.0.0.1", port=8731, debug=False)
