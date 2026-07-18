"""
Master orchestration script for the full RIO10 test scene pipeline.

Runs all 5 steps in sequence:
  1. Download sequences + mesh files
  2. Extract & convert to NerfStudio format
  3. Train Gaussian splats via NerfStudio (longest step: ~11-23 hours)
  4. Transfer instance labels
  5. Build test graphs

Each step has resume support — you can interrupt and restart safely.

Usage:
    python scripts/run_test_pipeline.py                    # Run full pipeline
    python scripts/run_test_pipeline.py --start-step 3     # Resume from step 3
    python scripts/run_test_pipeline.py --end-step 2       # Only run steps 1-2
    python scripts/run_test_pipeline.py --max-scenes 5     # Test with 5 scenes
    python scripts/run_test_pipeline.py --skip-training    # Skip NerfStudio (if using pre-trained)
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path


SCRIPTS_DIR = Path(__file__).parent


def run_step(step_num: int, name: str, script: str, extra_args: list = None):
    """Run a pipeline step and report status."""
    print(f"\n{'='*70}")
    print(f"STEP {step_num}: {name}")
    print(f"{'='*70}\n")

    cmd = [sys.executable, str(SCRIPTS_DIR / script)]
    if extra_args:
        cmd.extend(extra_args)

    start = time.time()
    result = subprocess.run(cmd)
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"\n  ⚠ Step {step_num} exited with code {result.returncode}")
        print(f"  You can resume from this step with: --start-step {step_num}")
        return False

    print(f"\n  ✓ Step {step_num} complete ({elapsed/60:.1f} min)")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Run full RIO10 test scene pipeline"
    )
    parser.add_argument("--start-step", type=int, default=1,
                        help="Start from this step (1-5)")
    parser.add_argument("--end-step", type=int, default=5,
                        help="End at this step (1-5)")
    parser.add_argument("--max-scenes", type=int, default=None,
                        help="Limit scenes (passed to all steps)")
    parser.add_argument("--skip-training", action="store_true",
                        help="Skip step 3 (NerfStudio training)")
    parser.add_argument("--workers", type=int, default=4,
                        help="Download workers")
    parser.add_argument("--max-iterations", type=int, default=30000,
                        help="NerfStudio training iterations")
    args = parser.parse_args()

    print("=" * 70)
    print("RIO10 TEST SCENE PIPELINE")
    print("=" * 70)
    print(f"  Steps: {args.start_step} → {args.end_step}")
    print(f"  Max scenes: {args.max_scenes or 'all (46)'}")
    print(f"  Skip training: {args.skip_training}")
    print()

    # Build common args
    common_args = []
    if args.max_scenes:
        common_args.extend(["--max-scenes", str(args.max_scenes)])

    pipeline_start = time.time()

    # Step 1: Download
    if args.start_step <= 1 <= args.end_step:
        extra = common_args + ["--workers", str(args.workers)]
        if not run_step(1, "Download sequences + meshes",
                        "download_3rscan_test_sequences.py", extra):
            return

    # Step 2: Extract & Convert
    if args.start_step <= 2 <= args.end_step:
        if not run_step(2, "Extract & convert to NerfStudio format",
                        "convert_3rscan_to_nerfstudio.py", common_args):
            return

    # Step 3: Train splats
    if args.start_step <= 3 <= args.end_step and not args.skip_training:
        extra = common_args + ["--max-iterations", str(args.max_iterations)]
        if not run_step(3, "Train Gaussian splats (NerfStudio)",
                        "train_3rscan_test_splats.py", extra):
            return

    # Step 4: Transfer labels
    if args.start_step <= 4 <= args.end_step:
        if not run_step(4, "Transfer instance labels",
                        "transfer_3rscan_test_labels.py", common_args):
            return

    # Step 5: Build graphs
    if args.start_step <= 5 <= args.end_step:
        if not run_step(5, "Build test graphs",
                        "build_3rscan_test_graphs.py", common_args):
            return

    total_elapsed = time.time() - pipeline_start
    print(f"\n{'='*70}")
    print(f"PIPELINE COMPLETE")
    print(f"{'='*70}")
    print(f"  Total time: {total_elapsed/3600:.1f} hours")
    print(f"\n  Test graphs saved to: data/3rscan_test_graph_cache/")
    print(f"  Ready for evaluation!")


if __name__ == "__main__":
    main()
