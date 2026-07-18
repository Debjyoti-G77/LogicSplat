# RIO10 Test Scene Annotations - Investigation Report

## Key Finding: Relationship annotations DO NOT EXIST publicly for the 46 RIO10 test scenes

### What We Found

1. **objects.json** (from `3DSSG.zip` at campar.in.tum.de) contains **all 1482 scans** including all 46 RIO10 test scenes
   - Each scene has labeled objects with: label, nyu40 class, attributes, affordances
   - Example: scene `00d42bed-...` has 39 objects (floor, wall, chair, etc.)

2. **relationships.json** contains only **1335 scans** — the 46 RIO10 scenes are NOT included
   - 147 scans have objects but no relationships (46 are RIO10, 101 are other test/val scenes)
   - Relationships are stored as `[subject_id, object_id, relationship_id, "predicate_text"]`

3. **3RScan.json** confirms all 46 RIO10 scenes are `type=test` reference scans

### How ReLaGS (and Open3DSG, RelationField) Evaluate

From the paper (Sec. 7, Implementation Details):
> "We train our graph neural network on 3DSSG dataset, with all testing sequences in RIO10 subset excluded to prevent data leakage"

From Sec. 4.1:
> "We evaluate our method on the RIO10 subset of 3DSSG for 3D scene graph prediction. This dataset provides semantic 3D scene graphs for pre-segmented 3D point clouds with 160 object classes and 27 relationship categories."

This means the **relationship ground truth for RIO10 test scenes exists** but is held privately by the dataset authors (Johanna Wald et al. at TUM). The evaluation protocol works as follows:

- **Training**: Use the 1335 scenes in relationships.json (excluding RIO10)
- **Testing**: The RIO10 test annotations are available to researchers who request access through the 3DSSG/3RScan data access form

### How to Get the Test Annotations

**Option A: Request from dataset authors**
- Fill out the 3RScan Terms of Use form: https://forms.gle/NvL5dvB4tSFrHfQH6
- Contact Johanna Wald (the 3RScan/3DSSG/RIO10 author) for the test split annotations
- The test annotations are likely distributed separately to prevent leakage

**Option B: Use what we have**
- We HAVE object annotations for all 46 RIO10 scenes (from objects.json)
- We can use these for object-level evaluation (Object Recall@K)
- For relationship evaluation, we need the held-out test annotations

**Option C: Use the 3DSSG test split (157 scenes) instead**
- The 3DSSG GitHub repo has `files/cvpr/test_scans.txt` with 157 test scene IDs
- These 157 scenes ARE in our relationships.json (152/157 found)
- This is a DIFFERENT test split from RIO10

### Data Available Now

| File | Scenes | RIO10 Coverage | Source |
|------|--------|----------------|--------|
| objects.json (full) | 1482 | 46/46 ✓ | 3DSSG.zip |
| relationships.json | 1335 | 0/46 ✗ | 3DSSG.zip |
| relationships_train.json | ? | ? | 3DSSG_subset server |
| 3RScan.json (full) | 1482 scans, 478 locations | 46/46 ✓ | 3RScan server |

### Recommended Next Steps

1. **Download the full objects.json** → already done, copy to `data/3DSSG/`
2. **Download the full 3RScan.json** from `https://www.campar.in.tum.de/public_datasets/3RScan/3RScan.json` (3.1MB, has all 1482 scans)
3. **Contact the 3DSSG authors** for the test relationship annotations
4. **Alternative**: Train on the 1335 annotated scenes and evaluate using the 3DSSG cvpr test split (157 scenes that DO have relationship annotations in our data)
