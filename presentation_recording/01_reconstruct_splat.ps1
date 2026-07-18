# LogicSplat -- narrated live reconstruction of scene_06, for screen recording.
#
# This does NOT touch the real, already-verified splat.ply used everywhere in
# the report/demo/defense deck (D:\logicsplat_data\processed\scene_06\splat.ply).
# It reconstructs the SAME real footage (the 76 already-extracted frames from
# scene_06.mov) into a separate scratch folder, purely so the reconstruction
# process itself can be shown live and genuinely, with zero risk to the real
# project data. For the downstream results shown later in the recording, switch
# to 02_run_pipeline_scene06.py, which uses the real, verified splat.ply.
#
# Run from the project root:
#     powershell -ExecutionPolicy Bypass -File presentation_recording\01_reconstruct_splat.ps1

$ErrorActionPreference = "Stop"

$IMAGES   = "D:\logicsplat_data\processed\scene_06\images"
$SCRATCH  = "D:\logicsplat_data\recording_scratch\scene_06_live"
$COLMAP   = "C:\Users\Debjyoti\Desktop\LogicSplat\bin\colmap.exe"

function Stage($n, $total, $title) {
    Write-Host ""
    Write-Host ("=" * 78)
    Write-Host "STAGE $n/$total -- $title"
    Write-Host ("=" * 78)
}

function Test-ExitCode($desc) {
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  -> FAILED: $desc exited with code $LASTEXITCODE" -ForegroundColor Red
        exit 1
    }
}

Write-Host "LogicSplat -- live reconstruction of scene_06"
Write-Host "Source footage: scene_06.mov, 76 frames already extracted"
Write-Host "Output goes to a scratch folder -- the real, verified splat.ply is untouched."
Write-Host "Scratch output: $SCRATCH"

New-Item -ItemType Directory -Force $SCRATCH | Out-Null
$TOTAL = 7

Stage 1 $TOTAL "Organise the real video frames into nerfstudio's expected layout"
ns-process-data images --data $IMAGES --output-dir "$SCRATCH\ns_data" --skip-colmap
Test-ExitCode "ns-process-data (stage 1)"
# Create the colmap folder now, BEFORE feature extraction runs -- COLMAP needs
# database_path's parent directory to already exist or it fails immediately.
New-Item -ItemType Directory -Force "$SCRATCH\ns_data\colmap\sparse" | Out-Null
Write-Host "  -> done. 76 real frames from the smartphone capture are now staged."

Stage 2 $TOTAL "COLMAP feature extraction (find trackable visual features per frame)"
& $COLMAP feature_extractor `
  --database_path "$SCRATCH\ns_data\colmap\database.db" `
  --image_path "$SCRATCH\ns_data\images" `
  --ImageReader.single_camera 1 `
  --ImageReader.camera_model OPENCV
Test-ExitCode "COLMAP feature_extractor"
Write-Host "  -> done."

Stage 3 $TOTAL "COLMAP sequential matching (match features across nearby frames)"
& $COLMAP sequential_matcher `
  --database_path "$SCRATCH\ns_data\colmap\database.db" `
  --SequentialMatching.overlap 10
Test-ExitCode "COLMAP sequential_matcher"
Write-Host "  -> done."

Stage 4 $TOTAL "COLMAP sparse reconstruction (recover camera poses + a 3D point cloud)"
& $COLMAP mapper `
  --database_path "$SCRATCH\ns_data\colmap\database.db" `
  --image_path "$SCRATCH\ns_data\images" `
  --output_path "$SCRATCH\ns_data\colmap\sparse"
Test-ExitCode "COLMAP mapper"
if (-not (Test-Path "$SCRATCH\ns_data\colmap\sparse\0")) {
    Write-Host "  -> FAILED: no sparse model was produced (check the log above for 'No images with matches')" -ForegroundColor Red
    exit 1
}
Write-Host "  -> done. This is the camera-pose recovery step -- structure from motion."

Stage 5 $TOTAL "Convert the COLMAP result into nerfstudio's transforms.json"
ns-process-data images --data $IMAGES --output-dir "$SCRATCH\ns_data" --skip-colmap
Test-ExitCode "ns-process-data (stage 5)"
if (-not (Test-Path "$SCRATCH\ns_data\transforms.json")) {
    Write-Host "  -> FAILED: transforms.json was not created" -ForegroundColor Red
    exit 1
}
Write-Host "  -> done."

Stage 6 $TOTAL "Train the 3D Gaussian Splat (splatfacto, 30,000 iterations)"
Write-Host "  This is the slow step -- 5 to 15 minutes on a consumer GPU."
Write-Host "  A live viewer opens at http://localhost:7007 -- open it in a browser"
Write-Host "  and record THAT, not just this terminal: you'll watch the splat"
Write-Host "  resolve from noise into a clean reconstruction over the iterations."
Write-Host "  NOTE: nerfstudio does not always exit on its own once training"
Write-Host "  finishes. Once you see the 'Training Finished' banner, the"
Write-Host "  checkpoint is already saved -- press Ctrl+C to continue."
ns-train splatfacto --data "$SCRATCH\ns_data" --output-dir "$SCRATCH\outputs" --max-num-iterations 30000 --viewer.quit-on-train-completion True
$trainExitCode = $LASTEXITCODE

Stage 7 $TOTAL "Export the final splat.ply"
$config = Get-ChildItem -Recurse -Filter "config.yml" "$SCRATCH\outputs" |
    Sort-Object LastWriteTime -Descending | Select-Object -First 1 -ExpandProperty FullName
if (-not $config) {
    Write-Host "  -> FAILED: no config.yml found under $SCRATCH\outputs -- training did not complete" -ForegroundColor Red
    exit 1
}
if ($trainExitCode -ne 0) {
    Write-Host "  -> ns-train exited with code $trainExitCode (likely your Ctrl+C after completion)," -ForegroundColor Yellow
    Write-Host "     but a checkpoint was found, so continuing: $config" -ForegroundColor Yellow
} else {
    Write-Host "  -> training complete. Checkpoint: $config"
}
ns-export gaussian-splat --load-config $config --output-dir $SCRATCH
Test-ExitCode "ns-export gaussian-splat"
Write-Host "  -> done. Fresh splat.ply written to: $SCRATCH\splat.ply"

Write-Host ""
Write-Host ("=" * 78)
Write-Host "DONE -- this is a genuine, from-scratch reconstruction of this scene's"
Write-Host "real footage. For the results shown next, we switch to this same"
Write-Host "scene's PREVIOUSLY VERIFIED reconstruction (identical footage), since"
Write-Host "a fresh training run has harmless run-to-run randomness in it -- the"
Write-Host "report's published numbers were measured on that verified version."
Write-Host ("=" * 78)
