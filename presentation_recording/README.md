# Presentation Recording — scripts for the defense screen recording

## Fastest path: one script, fully automatic

```powershell
powershell -ExecutionPolicy Bypass -File presentation_recording\00_run_all.ps1
```

Start your screen recording, run this one command, and everything happens in order automatically: reconstruction → the verified scoring run → the interactive demo server starts and opens in your browser already on Tabletop 06. It stops there so you can do the live, on-camera part yourself (hovering relations, toggling repair, clicking the tabs) — that part is deliberately not automated, since it's the actual live demo you want the supervisor to watch you drive.

Everything below explains what `00_run_all.ps1` does under the hood, in case you want to run the stages individually instead.

---

Three scripts, covering the whole pipeline: **splat generation → clustering → inference → repair → scoring → interactive demo.** All print clear `STAGE n/N` banners as they run, so the terminal narrates itself on camera — you don't have to explain every line out loud.

Nothing here touches or overwrites any of the real, verified project data. All three were test-run before being handed to you.

## What's active vs. stale (checked before writing these)

`eval_geokan_tabletop.py` is the confirmed **active** script behind every tabletop number in the report and defense deck — verified two ways: its `MODEL_PATH` points at `models/geokan_relation_gamma.pt` (the real production checkpoint), and its own fresh log (`results/tabletop_fresh_log.txt`) matches the report's Recall@5 = 92.7% exactly. `demo/precompute.py` (the LogicSplat Live demo) already imports its functions directly, for the same reason.

Both scripts below **only import from that active file** — nothing is reimplemented. Files like `finetune_tabletop_gt.py`, `finetune_geokan_tabletop.py`, `adapt_tabletop_ttbn.py` are older, pre-final-checkpoint experiments (their model outputs are not `geokan_relation_gamma.pt`) and are not touched or used here.

## 1. `01_reconstruct_splat.ps1` — the slow, upstream part

Reconstructs **scene_06's real footage** (76 already-extracted frames from `scene_06.mov`) from scratch: COLMAP feature extraction → matching → sparse reconstruction → `ns-train splatfacto` (30,000 iterations) → export. Writes everything to a scratch folder (`D:\logicsplat_data\recording_scratch\scene_06_live`) — **the real `splat.ply` used in your report is never touched.**

```powershell
powershell -ExecutionPolicy Bypass -File presentation_recording\01_reconstruct_splat.ps1
```

**Record the browser, not just the terminal**, once `ns-train` starts — it opens a live viewer at `http://localhost:7007` where you can actually watch the splat resolve from noise into a clean reconstruction. That's the most visually compelling part of the whole recording.

**Expect this to take 10–20 minutes real time** (COLMAP is quick; `ns-train` is the slow part). Per the recording plan in `DEFENSE_SCRIPT.md`, capture it in real time and speed it up in editing afterward, with an on-screen caption saying so — never cut the step out.

**Say on camera when this segment ends:** *"That was a genuine, from-scratch reconstruction of this scene's real footage. For the results I'll show next, I'm switching to this same scene's previously-verified reconstruction — a fresh training run has harmless run-to-run randomness in it, and the report's published numbers were measured on the verified version."* This is honest, not a shortcut — say it plainly rather than skipping past it.

## 2. `02_run_pipeline_scene06.py` — the fast, downstream part (the payoff)

Runs the **real, verified pipeline** end to end on the already-verified `splat.ply`: load model → load ground truth → cluster into objects → build the graph → run GeoKANRelationGNN → apply SceneGraphRepair → score. Already test-run — reproduces the exact **66/84** correct relations from the report.

```powershell
python presentation_recording\02_run_pipeline_scene06.py
```

This runs in seconds. Every stage prints what it's doing, how long it took, and a plain-language description of the sub-steps involved — a viewer can follow along from the terminal output alone.

## Suggested recording order

1. `01_reconstruct_splat.ps1` — sped up in editing, captioned honestly.
2. Say the "switching to verified data" line above.
3. `02_run_pipeline_scene06.py` — real time, it's fast enough not to need editing.
4. Switch to the LogicSplat Live demo UI (`demo/`) for the interactive payoff — arrows on the photo, Symbolic Repair toggle, the All/Correct/Wrong/Missed tabs.

This matches segments 1–5 of the recording plan already written in `DEFENSE_SCRIPT.md` at the project root.
