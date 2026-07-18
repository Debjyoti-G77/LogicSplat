# LogicSplat — Defense Script

Rehearsal copy for the M.Sc. thesis defense. Every line below is grounded in what's already on the actual slide in `LogicSplat_Defense.pptx` — nothing new is being claimed. Say it like you know it, not like you're reading it.

The deck is now **21 slides** — slide 17 (LERF qualitative) is built and in the file. Slides 17A and 17B below are proposed additions, not yet built into the `.pptx` — say the word and I'll add them.

---

## 01 — Title

**Say:** "Good [morning/afternoon]. My thesis is LogicSplat — neuro-symbolic 3D scene graph generation using Geometric Kolmogorov–Arnold Networks. I'm Debjyoti Sengupta, supervised by Mr. Sahil Shah at the Symbiosis Institute of Geoinformatics."

*Beat: ten seconds, confident, then move.*

## 02 — Context

**Say:** "A robot setting a table doesn't just need to know a cup exists — it needs to know the cup rests *on* the table. An AR assistant needs to know a router sits *behind* a laptop, not beside it. A scene graph captures exactly that: objects as nodes, spatial relations as directed edges. Gaussian Splatting has already made the geometry side cheap — a photorealistic, room-scale reconstruction from a short phone video, five to fifteen minutes, consumer hardware. So the question this thesis asks is simple: is that geometry, on its own, enough to recover the relational structure of a scene? Reconstruction isn't the bottleneck anymore — understanding what's been reconstructed is."

## 03 — Motivation

**Say:** "Three obstacles sit between a raw splat and a reliable scene graph. One — semantic grounding is expensive: GaussianGraph, ReLaGS, ConceptGraphs all treat 'knowing what an object is' as a prerequisite for reasoning about how it relates to others — hundreds of millions to billions of parameters, sometimes ten-plus minutes of per-scene optimisation, for relations that are actually fully determined by centroids and bounding volumes. Two — 3DSSG is severely under-annotated: labelers mark only 80 to 120 pairs per scene out of roughly 870 valid ones, so about 90% go unannotated, and standard training treats every one of those as a hard negative. Three — predicted graphs are logically inconsistent: relations are predicted independently per pair, so higher_than(A,B) and higher_than(B,A) can both come out true, and nothing in the literature applies a formal mechanism to fix that."

## 04 — Objectives

**Say:** "Four objectives, each answered by a controlled experiment. Predict relations directly from splat geometry — no CLIP, SAM, LLM, or per-scene optimisation at inference. Repair the training signal, so silence in the annotations stops being treated as a negative. Guarantee logical consistency after prediction, with a zero-parameter symbolic step. And test fairly whether a learnable-metric architecture actually beats a matched MLP, in-distribution and under domain shift."

## 05 — Related Work

**Say:** "Walking the table: ReLaGS needs CLIP and SAM — about 1.5 billion parameters — for 87% Recall@5. RelationField distills GPT-4o into a per-scene NeRF optimisation, 60 to 90 minutes on an A100, for 82%. ConceptGraphs and Open3DSG lean on GPT-4 or an LLM co-embedding for 79 and 65%. GaussianGraph stacks four foundation models for 63% on LERF. Every one of these treats semantic grounding as a prerequisite — whether the geometry already sitting inside the splat is sufficient on its own has not been systematically tested. That's the gap."

## 06 — The Key Idea

**Say:** "Treat spatial relations as what they are: geometric facts. on_top_of, left_of, higher_than are fully determined by centroids and bounding volumes, independent of what the objects semantically are — no grounding needed to state them, so none should be needed to predict them. Three decisions follow: learn with GeoKANRelationGNN, a graph network with a learnable geometric metric, 1.02 million parameters; supervise with rule-based label injection wherever annotation is silent; guarantee consistency with SceneGraphRepair, zero parameters. Every result today tests one of these three."

## 07 — System Overview

**Say:** "End to end: a smartphone video, 30,000 iterations of splat reconstruction, cleaning — opacity, outliers, plane removal — HDBSCAN clustering into objects, 10-D node and 22-D edge features, GeoKANRelationGNN, and SceneGraphRepair. No foundation model anywhere in that chain at inference."

## 08 — Method: Learning

**Say:** "A standard basis expansion treats every input dimension as equally informative. A GeoKAN layer first learns a per-dimension scaling — a diagonal Riemannian metric — that stretches useful dimensions and compresses the rest, *before* any basis function is applied: z equals u times the square root of that learned metric, then each dimension is expanded against 12 fixed centres. Three variants, differing in exactly two places: Gamma — one learnable scalar per dimension, input-independent. RBF — that metric becomes input-dependent through a small two-layer network. Wavelet — same input-dependent metric, but the basis becomes a Mexican-hat wavelet with signed side-lobes."

## 09 — Method: Architecture

**Say:** "Judging contact and judging direction need different evidence, so they get separate heads. A shared backbone projects the 10-D node features to 128-D and runs two GATv2 layers with the 22-D edge features feeding attention directly. The contact head — on_top_of, under, attached_to, adjacent_to — sees all 22 edge dimensions, because physical support needs vertical-gap and containment signals. The directional head — left, right, front, behind, higher, lower — sees only the first 10, because direction is fully determined by relative position; the rest is noise for that question. All told: 1,018,414 parameters for Gamma — roughly three orders of magnitude below the foundation-model systems it's compared against."

## 10 — Method: Supervision

**Say:** "The premise: an unannotated pair is not evidence of absence. If one centroid is measurably left of another, that relation holds whether or not a labeler wrote it down. So — inverse completion: annotators record on_top_of(A,B) but rarely under(B,A), so every annotation's inverse gets injected at the same confidence. And confidence-calibrated geometric rules fill the unannotated 90%: directional relations at 1.00, on_top_of and under at 0.75, attached_to at 0.60, adjacent_to at 0.55 — weakest evidence, lowest weight. Net effect: just over one million directed edges become 2.38 million positive labels, and human labels are never overwritten — rules only speak where annotators were silent."

## 11 — Method: Logical Repair

**Say:** "A network predicts each pair independently; physics doesn't work pair-by-pair. Four constraints, applied deterministically. Inverse completeness — the missing inverse is added at 0.95 times the source confidence. Mutual exclusion — contradictory relations can't both survive; the less confident is dropped. Asymmetry — a directional relation can't hold both ways. Transitivity — the implied relation is added at 0.9 times the weaker link. Fixed-point iteration, at most 10 rounds, though every scene I've tested converges in two or three. These are physical facts, not dataset conventions — that's why it costs zero learned parameters."

## 12 — Evaluation Setup

**Say:** "One training run, three settings of increasing difficulty. Training plus in-domain validation: 565 3RScan scenes, 480 train, 85 held out, room-scale. Dataset shift: four held-out LERF scenes — ramen, teatime, waldo_kitchen, figurines — 1,644 triples, never seen in training. Zero-shot scale shift: eight scenes I captured myself, tabletop scale, a tenth the size of training, 508 triples — three needed a Z-axis correction before evaluation. Every one of these reuses the exact same weights and thresholds from the single 3RScan training run — nothing is retuned per setting."

## 13 — Results: In-Domain

**Say:** "98.3% Recall@5 on 3DSSG, 1.02 million parameters, no foundation model — against ReLaGS's 87%, RelationField's 82%, ConceptGraphs' 79%, Open3DSG's 65%. Here's the honest part: against a matched-capacity MLP on the same features and labels, across 85 validation scenes, they're nearly indistinguishable — 0.9325 Macro F1 for the MLP versus 0.9257 for GeoKAN-Gamma. So in-distribution, this headline number is a property of the features, the injected labels, and the repair logic — not the classification head. The head only starts to matter under domain shift, which is the next two slides."

## 14 — Results: Scale Shift

**Say:** "At a tenth of training scale, zero-shot: 92.7% Recall@5. But the real story is the comparison — every GeoKAN variant beats the matched MLP on every metric here, 5 to nearly 10 Micro-F1 points, 4 to nearly 6 Recall@5 points after repair. The in-distribution ranking actually inverts: Wavelet, the weakest variant in-domain, leads under scale shift, because its input-conditioned metric generalises further than Gamma's fixed scalar. And repair adds almost every time, removes almost never — 88, 54, 42 relations added across the three variants, converging within two iterations."

## 15 — Results: Dataset Shift

**Say:** "On four held-out LERF scenes: 95.0% Recall@5, 31.8 points above GaussianGraph's best published configuration — a direct comparison, since these are entirely positional relations. Repair alone, no retraining, takes Recall@3 from 71.5% to 89.9% — an 18.4 point jump — adding 39 relations and removing none, so predictions stay internally consistent even fully off-distribution. Per scene: 99% on ramen down to 91.5% on teatime."

## 16 — Results: Qualitative (Tabletop)

**Say:** "To make this concrete rather than a table: one real scene I captured myself — router, hair-dryer box, water bottle, watch, pen, all close together on a desk. In the clustered splat, colour is what separates these objects, not distance — they genuinely sit that close. Of the 84 relations predicted after repair, 66 were verified correct by hand against the real scene."

*Beat: this is the scene that slide 17's live demo card and the recording both return to — say so explicitly: "this exact scene, now live."*

## 17 — Results: Qualitative — LERF *(built)*

The dataset-shift counterpart to slide 16 — same three-panel layout, LERF's "ramen" scene, 346 of 360 predicted relations verified correct.

**Say:** "Same pass, but now on a held-out public benchmark scene instead of one I captured myself — ten items in a single ramen bowl: bowl, noodles, egg, kamaboko, nori, and five more. Ten objects packed that close together are genuinely harder to separate cleanly than the tabletop scene — you can see it in the clustered splat, there's more colour overlap here than in the router-and-desk scene. But the same checkpoint, zero fine-tuning, still recovers this graph: 346 of 360 predicted relations verified correct after repair."

*Beat: if asked why the splat panel looks messier here, that's the honest answer — say it plainly, don't dodge it. It's a real property of small, tightly-packed objects, not a rendering artefact.*

## 17A — Live Demo Scene *(proposed — not yet built)*

Shows the same tabletop scene from slide 16, now as the interactive LogicSplat Live UI — arrows on the actual photo, confidence bars, ground-truth chips.

**Say:** "I built an interactive view around this exact scene — arrows drawn live between objects on the actual photo, a confidence bar and a ground-truth check on every relation. Every one of its predictions is filterable — what the model got right, what it got wrong, and what it missed outright — and symbolic repair can be toggled live to watch it add relations in real time."

*Beat: a static screenshot works if the recording (17B) is the main event; a 15-second live click-through works if you have a stable connection to the room's display.*

## 17B — Pipeline Walkthrough — Recorded *(proposed — not yet built)*

A title card that leads into playing the screen recording, since reconstruction can't run live in the room.

**Say:** "Reconstruction itself takes several minutes, so it isn't something I can run live in this room. I recorded the full process end to end instead — raw phone video, through reconstruction, clustering, inference, and repair, to this same interactive view. [Play recording — see the plan below.]"

## 18 — Analysis: Ablation

**Say:** "This is the controlled experiment behind the last two results slides — same features, labels, augmentation, loss, schedule, thresholds; only the layer changes, and the MLP gets *more* capacity, not less: 1.57 million parameters against GeoKAN's 1.02 million. In-distribution, no advantage — 0.9325 versus 0.9257, consistent with Yu et al.'s prior KAN-versus-MLP findings. Under scale shift, GeoKAN leads on every metric. Under dataset shift, GeoKAN leads again, by 2.2 points in both Micro F1 and Recall@5. The metric warp is an inductive bias that costs nothing in-distribution and pays for itself exactly where the system is meant to operate — beyond its training domain."

## 19 — Contributions

**Say:** "Five findings. Semantic grounding isn't a prerequisite — geometry-only prediction beats every foundation-model or per-scene-optimised system compared here. Annotation sparsity has a principled fix — pseudo-labels calibrated to each rule's actual certainty, not silence-as-negative. GeoKAN buys generalisation, not raw accuracy — no in-distribution edge, but a consistent gain under both kinds of domain shift, which the existing literature hadn't tested. Symbolic repair works mainly by completing missing relations, not resolving contradictions — out-of-distribution errors are dominated by omission. And scale, not dataset identity, drives most of the degradation under shift — in_front_of and behind degrade most."

## 20 — Limitations & Future Work

**Say:** "Every limitation here was a deliberate scope choice. Oracle clustering for held-out settings — the standard PredCls convention — isolates relation prediction, this thesis's actual contribution, from instance segmentation, a separate problem. Scenes span 4 to 13 objects, comfortably covering tabletop and single-room use. attached_to gets no cross-domain test because it doesn't occur naturally in the tabletop benchmark. One training run per configuration, consistent with a first systematic comparison. And the geometric assumptions target static, rigid, indoor scenes — the deployment scenario this work is built for. Next: exploit the splat's own per-Gaussian covariance, already sitting there unused; denser multi-room environments; a proper attached_to cross-domain test; multiple seeds on the closest margin; and whether GeoKAN's shift-advantage generalises beyond this task."

## 21 — Closing

**Say:** "Geometry, a learnable metric, and deterministic logic turn out to be sufficient — no foundation model required. 98.3% Recall@5 in-domain, 95.0% held-out on LERF, 92.7% zero-shot at a tenth of training scale, all from 1.02 million parameters — no VLM, no LLM — five to six seconds end-to-end per scene. Thank you — happy to take questions."

---

## Recording the pipeline walkthrough

Goal: an honest, watchable 3–4 minute video — raw video to the interactive final view — that plays under slide 17B. Slow parts are sped up on screen, never hidden.

> **Ground rule:** compress time with a visible on-screen caption ("sped up 15×"), never by cutting the step out. If reconstruction is asked about in Q&A, you want to be able to say "yes, you saw all of it, just faster."

**Tools** (Windows 11, already installed, free):
- **Capture** — Xbox Game Bar (Win+G), records a window or full screen with mic audio, zero install. Use this unless OBS is already installed.
- **Edit** — Clipchamp, already on Windows 11. Trims clips, speeds up a segment (e.g. 15×), adds burned-in captions, layers a separate narration track, exports 1080p.
- **Optional** — if OBS Studio is already installed, prefer it for capture — scene-switching between terminal / browser / viewer without alt-tabbing on camera looks more deliberate.

### Segments

1. **Cold open — raw footage** *(real time, ~10–15s)* — Play a few seconds of the actual phone video of the tabletop scene. Caption: "Raw smartphone video, ~20–30 seconds of footage." *Voice: "This is where every result today starts — a walk-around video, nothing else."*

2. **Reconstruction — sped up** *(captured real time, played back 10–15×, burned-in caption required)* — Start the COLMAP + nerfstudio/splatfacto training command. Nerfstudio's own web viewer shows the splat resolve from noise to clean geometry live — record that panel. Speed the clip up in Clipchamp afterward; keep the on-screen timer/iteration counter visible so the compression is legible, not hidden. *Voice: "Thirty thousand iterations, five to fifteen minutes on a consumer GPU — sped up here, nothing skipped."*

3. **Cleaning + clustering** *(real time, ~30–45s)* — Run the clustering step; capture the coloured point-cloud visualisation appearing (the same figure as slide 16's "clustered splat, colour = cluster"). Fast enough to show at real speed. *Voice: "Opacity filtering, outlier and plane removal, then HDBSCAN splits this into five to thirteen object clusters, depending on the scene."*

4. **Inference + repair** *(real time, ~10–15s)* — Run the eval script or hit `demo/server.py`'s `/predict` endpoint from a terminal. Let the actual millisecond timing print on screen — it's already fast enough to not need speeding up. *Voice: "Inference and repair together: single-digit to low-tens of milliseconds. This is the part that's genuinely real-time."*

5. **The payoff — LogicSplat Live** *(real time, ~60–90s, rehearse the click-path 2–3× first)* — Switch to the demo UI on the same scene. Hover two or three relations to draw arrows, toggle Symbolic Repair on and watch the "+N added" count and new rows appear, click through the All / Correct (TP) / Wrong (FP) / Missed (FN) tabs. *Voice: "Same scene, same checkpoint — every relation here is genuinely computed, nothing in this view is hardcoded."*

6. **Optional closer — second dataset** *(real time, ~10–15s, cut if time is tight)* — Quick cut to a LERF scene in the same UI, to back up the "same pipeline, two datasets" claim from earlier in the talk. *Voice: "Same model, zero fine-tuning, a public benchmark it's never seen."*

### Before you hit record

- [ ] Record narration as a separate pass over silent screen footage — much easier to get clean audio than narrating live while clicking.
- [ ] Rehearse the demo click-path (hover → toggle repair → tab through Correct/Wrong/Missed) two or three times so the mouse moves with intent, not hesitation.
- [ ] Turn on Focus Assist / disable notification popups; close unrelated tabs and windows; set the browser to a fixed 100% zoom and a consistent window size.
- [ ] Keep every raw per-segment clip after recording — don't overwrite them once the stitched cut is exported, in case a re-cut is needed later.
- [ ] Export at 1080p. Target 3–4 minutes total; if a fuller, unedited cut exists as backup, have it ready on a second file in case Q&A specifically probes "was any of that sped up more than shown."

---
*logicsplat / DEFENSE_SCRIPT.md — grounded in `LogicSplat_Defense.pptx` (21 slides), 2026-07-11*
