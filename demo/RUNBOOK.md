# LogicSplat Live — Defense Runbook

## Starting it

Double-click **`demo/run.bat`**. It starts two local servers and opens the
app in your default browser at `http://localhost:8790/index.html`.

If double-click doesn't work (e.g. Python not on PATH in that shell),
open a terminal in `demo/` and run:

```
python -m http.server 8790
```

then open `http://localhost:8790/index.html` yourself. The app works fully
without the second (optional) server — see "About the LIVE/CACHED pill" below.

**No internet required.** Everything is self-contained and local.

## Presenter keyboard shortcuts

These work once you've clicked anywhere on the page:

| Key | Action |
|---|---|
| `→` / `←` | Next / previous scene |
| `R` | Toggle symbolic repair on/off |
| `G` | Toggle ground-truth ✓/✗ chips |
| `F` | Fullscreen |
| `Esc` | Un-pin a pinned relation |

Mouse: hover any relation row to draw its arrow on the photo; click to pin
it (stays drawn until you click elsewhere or press Esc). Click an object's
box to see all of its relations highlight at once.

## Suggested 3-minute walkthrough

1. **Start on a tabletop scene** (e.g. Tabletop 06). Say: *"This is a scene
   I captured myself on my phone — just a video, reconstructed into a 3D
   Gaussian Splat."*
2. **Hover 2–3 relation rows** to show the arrow-on-photo interaction.
   Point out the confidence bar and the ✓/✗ ground-truth chip.
3. **Flip the Symbolic Repair toggle on.** Say: *"The neural network's raw
   predictions can be logically inconsistent — this module fixes that
   with zero learned parameters."* Point out the `+N added` summary and
   the newly-tagged relations in the list.
4. **Switch to a LERF scene** (press `→` until you reach one, or click it
   in the left rail). Say: *"This is the same model, same checkpoint, zero
   fine-tuning — evaluated on LERF, a public benchmark it's never seen
   during training. Same interaction, same repair mechanism."*
5. **Close on the numbers**: point at the latency strip (bottom-left) —
   inference in milliseconds, ~1 million parameters, no foundation model
   anywhere in the pipeline.

## About the LIVE / CACHED pill

Each scene's numbers were computed by actually running the real pipeline
(`demo/precompute.py`) — nothing in the UI is hardcoded. The pill in the
bottom-left rail tells you which path served the *current view*:

- **CACHED** (grey): loaded from the precomputed JSON. This is the normal,
  reliable state — expect this for the whole presentation.
- **LIVE** (green): the optional server (`demo/server.py`) actually
  re-ran the model for that click. It's a genuine bonus feature, but the
  full pipeline (loading + clustering + inference) takes several seconds,
  well past the UI's 500ms patience window, so it will almost always show
  CACHED in practice — this is expected, not a bug. If someone asks
  whether it's "just a slideshow," you can open a terminal and run:
  ```
  curl http://localhost:8731/predict/tabletop_06
  ```
  to show it recomputing the real result live (~15–20 seconds).

## If something goes wrong

- **Blank page / stuck on "Loading":** the static server (port 8790)
  probably isn't running. Re-run `run.bat`, or check Task Manager for a
  stray `python.exe` holding the port and end it, then retry.
- **A scene photo is missing:** `demo/data/<scene>.jpg` wasn't generated.
  Re-run `python demo/precompute.py` from the project root.
- **Numbers look stale after you changed something:** re-run
  `python demo/precompute.py`, then hard-refresh the browser (Ctrl+Shift+R).
- **Nothing else works:** open `demo/index.html` directly as a file — most
  interactions will still render, though the manifest fetch needs an
  actual HTTP server (browsers block `fetch()` on `file://`).

## Regenerating scene data

If you want to add a scene or refresh the numbers:

```
cd C:\Users\Debjyoti\Desktop\LogicSplat
python demo\precompute.py
```

This re-runs the real model on all configured scenes (tabletop_06/08/10,
lerf_ramen, lerf_teatime) and rewrites `demo/data/*.json` + `.jpg` +
`manifest.json`. Takes under a minute total.

## What's real vs. what's a design choice

- **Real, verified**: every relation, confidence score, repair diff, and
  timing number — all computed live in `precompute.py` from the actual
  checkpoint (`models/geokan_relation_gamma.pt`) and the actual clustering/
  repair pipeline. Tabletop_06 was cross-checked against the manuscript's
  own verified qualitative figure (66/84 correct relations) and matches
  exactly.
- **Design choice, not projection math**: object bounding boxes on the
  tabletop photos were hand-placed by visually matching the photo (the
  "calibration fallback" from the build plan) rather than computed via
  camera projection, since tabletop scenes use nerfstudio's coordinate
  convention which didn't cleanly match up in the time available. LERF
  scene boxes come directly from LERF-OVS's own published pixel labels,
  with model objects matched to those labels via a verified 3D→2D
  projection (visually checked — see the projection note in
  `demo/lerf_project.py`). Box *positions* are accurate either way; only
  the *method* used to place them differs between the two datasets.

## Known limitation

One LERF scene (teatime) labels a sheep's leg as "hooves" — that's the
public benchmark's own label vocabulary, not something this project
invented. If asked, it's a fine, honest thing to just say.
