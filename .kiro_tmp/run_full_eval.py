"""Run full evaluation with --compare flag on scenes 06-13."""
import sys
import os

# Force UTF-8 for stdout/stderr
os.environ["PYTHONIOENCODING"] = "utf-8"
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

sys.path.insert(0, r"c:\Users\Debjyoti\Desktop\LogicSplat")
os.chdir(r"c:\Users\Debjyoti\Desktop\LogicSplat")

sys.argv = [
    "evaluate_scenes.py", "--compare",
    "--scenes", "scene_06", "scene_07", "scene_08", "scene_09",
    "scene_10", "scene_11", "scene_12", "scene_13"
]

from scripts.evaluate_scenes import main
main()
