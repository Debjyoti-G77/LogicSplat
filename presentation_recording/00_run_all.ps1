# LogicSplat -- ONE script, full pipeline, fully automatic.
#
# Runs, in order: reconstruction -> verified scoring run -> launches the
# interactive demo and opens it in your browser, already on Tabletop 06.
# Just start recording your screen, then run this one script.
#
# Run from the project root:
#     powershell -ExecutionPolicy Bypass -File presentation_recording\00_run_all.ps1

$ErrorActionPreference = "Stop"
$ROOT = "C:\Users\Debjyoti\Desktop\LogicSplat"
Set-Location $ROOT

function Phase($n, $title) {
    Write-Host ""
    Write-Host ("#" * 78)
    Write-Host "PHASE $n -- $title"
    Write-Host ("#" * 78)
}

Write-Host "LogicSplat -- full pipeline, splat generation to interactive scene graph"
Write-Host "Scene: scene_06 (Tabletop 06 -- router, hair-dryer box, water bottle, watch, pen)"

# ── Phase 0: clean up anything left over from a previous run ────────────────
Write-Host ""
Write-Host "Cleaning up before starting fresh..."
$SCRATCH_ROOT = "D:\logicsplat_data\recording_scratch"
if (Test-Path $SCRATCH_ROOT) {
    Remove-Item -Recurse -Force $SCRATCH_ROOT
    Write-Host "  -> removed old scratch reconstruction folder"
}
Get-NetTCPConnection -LocalPort 8790 -ErrorAction SilentlyContinue |
    ForEach-Object {
        Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue
        Write-Host "  -> stopped stale demo server (PID $($_.OwningProcess))"
    }

# ── Phase 1: reconstruct the real footage from scratch ──────────────────────
Phase 1 "Reconstruct scene_06 from its real footage (splat generation)"
& powershell -ExecutionPolicy Bypass -File "$ROOT\presentation_recording\01_reconstruct_splat.ps1"

Write-Host ""
Write-Host "That was a genuine, from-scratch reconstruction of this scene's real"
Write-Host "footage. Switching now to this same scene's previously-verified"
Write-Host "reconstruction for the results below -- a fresh training run has"
Write-Host "harmless run-to-run randomness, and the report's published numbers"
Write-Host "were measured on the verified version."
Start-Sleep -Seconds 2

# ── Phase 2: run the real, verified downstream pipeline ─────────────────────
Phase 2 "Cluster, predict, and repair -- the verified pipeline"
python "$ROOT\presentation_recording\02_run_pipeline_scene06.py"

# ── Phase 3: launch the interactive demo ─────────────────────────────────────
Phase 3 "Launch the interactive scene-graph demo"
Write-Host "Starting the static file server on http://localhost:8790 ..."
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$ROOT\demo'; python -m http.server 8790" -WindowStyle Minimized
Start-Sleep -Seconds 2
Start-Process "http://localhost:8790/index.html?scene=tabletop_06"

Write-Host ""
Write-Host ("#" * 78)
Write-Host "READY -- the demo is open on Tabletop 06."
Write-Host "Live, on camera from here: hover a relation row to draw its arrow,"
Write-Host "toggle Symbolic Repair, and click through the All / Correct (TP) /"
Write-Host "Wrong (FP) / Missed (FN) tabs."
Write-Host ("#" * 78)
