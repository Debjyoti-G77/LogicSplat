"""Run full evaluation with --compare flag."""
import sys
sys.path.insert(0, ".")
sys.argv = ["evaluate_scenes.py", "--compare", "--scenes",
            "scene_06", "scene_07", "scene_08", "scene_09",
            "scene_10", "scene_11", "scene_12", "scene_13"]

exec(open("scripts/evaluate_scenes.py").read())
