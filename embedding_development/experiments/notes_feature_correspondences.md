# Notes: Feature Correspondences Analysis
*Script: `find_feature_correspondences.py`*

## What This Script Does

Investigates whether MLP attribution methods (Global PV, IG, Cond-PV) correctly
identify **which behavioral features actually correlate** with each neural ensemble's
activity in each recording session.

### Section 1: Behavioral Correspondence Matrix
- For every (session, ensemble, semantic group) triple: computes max |Spearman ρ|
  between ensemble z-scored activity and any column in the semantic group,
  using **all timepoints** in the session (not just test set — more statistical power)
- This is fully model-agnostic: it asks "does this ensemble fire more when behavior X occurs?"
- Shape: `correspondence_matrix.npy` — (29 sessions, 23 ensembles, 12 groups)
- Output: `correspondence_heatmap.png`, `correspondence_per_session_top4.png`

### Section 2: Attribution-Correspondence Alignment
- For each method × group pair: Spearman ρ between behavioral |ρ| and attribution score
- Tells us: does the method assign high scores to features that are genuinely behaviorally correlated?
- A good method should have high alignment ρ across all groups
- Output: `alignment_heatmap.png`, `alignment_scatterplots.png`, `alignment_stats.csv`

### Section 3: Top-Pair Showcase
- Scores each (session, ensemble, group) triple by: correspondence × GPV × peak R²
- Greedy diversity selection (max 2 per ensemble, max 2 per group → 9 triples)
- Each showcase figure has 3 panels:
  - **Left**: time series overlay of ensemble activity (blue) and behavioral var (red)
  - **Center**: attribution bar chart (GPV, IG, CPV for all 12 groups; focal group highlighted gold)
  - **Right**: scatter plot of ensemble activity vs behavioral variable with Spearman ρ
- Output: `showcase_XX_ENxx_Sxx_groupname.png` (9 files)

### Section 4: Conditional Attribution by Categorical Level
- For the **top 3 (session, ensemble) pairs** by mean attribution:
  - Loads the seed-42 trained model
  - Splits test data by each categorical variable level (e.g., cue_visible=0 vs 1)
  - Computes GPV and IG **within each subset** separately
  - Shows: does the model rely on different features when the animal is in different conditions?
- Output: `conditional_gpv_ENxx_Sxx.png`, `conditional_ig_ENxx_Sxx.png`

### Section 5: Summary CSV
- Full table of all valid (session, ensemble, group) triples with:
  behavioral ρ, GPV, IG, CPV, mean R², composite score
- Sorted by composite score for easy inspection

---

## Key Findings (Run: 2026-05-17)

### Section 1: Correspondence Matrix
- **E03** is most behaviorally active: correlates with lick_detected (showcases 1-2)
- **E18** has strongest head_angle correspondence across multiple sessions (S28, S29)
- **E02, E04** encode track_zone_int (spatial position)
- **E05, E04** encode frame_raw / YawPitch features (visual flow)
- The correspondence matrix reveals at least 4 distinct encoding clusters

### Section 2: Attribution-Correspondence Alignment

Best and worst alignment (attribution vs behavioral correlation):

| Feature Group | GPV ρ | IG ρ | Cond-PV ρ |
|--------------|-------|------|-----------|
| head_angle_vel | **0.674** | **0.700** | **0.721** |
| track_zone_int | 0.433 | 0.466 | 0.423 |
| lick_detected  | 0.529 | 0.554 | 0.401 |
| reward_window  | 0.580 | 0.273 | 0.554 |
| **head_angle**   | 0.268 | 0.055 | 0.121 |
| cue_visible    | -0.004 | -0.266 | 0.156 |

**Critical finding**: `head_angle_vel` has the BEST alignment (ρ≈0.70) for all methods — the
methods reliably identify which ensembles are tuned to head angular velocity.

**But `head_angle` itself has poor IG alignment (ρ=0.055, p=0.44)** — IG assigns high importance
to head_angle even for ensembles that aren't actually strongly correlated with it. This suggests
IG may be over-attributing to head_angle in a non-selective way. GPV has moderate alignment
(0.27) but IG effectively ignores the distinction.

`cue_visible` has negative IG alignment (ρ=-0.266) — IG systematically under-attributes to
features that are actually correlated with ensemble activity when cue_visible is involved.

### Section 3: Showcase Top Pairs
| # | Ensemble | Session | Group | Behavioral ρ | GPV | R² |
|---|----------|---------|-------|-------------|-----|----|
| 1 | E03 | S1  | lick_detected | high | high | 0.462 |
| 2 | E03 | S2  | lick_detected | high | - | high |
| 3 | E05 | S25 | frame_raw | high | - | - |
| 4 | E18 | S29 | head_angle | high | 0.308 | 0.222 |
| 5 | E18 | S28 | head_angle | high | 0.266 | 0.236 |
| 6 | E04 | S7  | frame_raw | high | - | - |
| 7 | E02 | S23 | track_zone_int | high | - | - |
| 8 | E08 | S29 | frame_YawPitch | high | - | - |
| 9 | E04 | S2  | track_zone_int | high | - | - |

Showcase reveals: the dataset has at least 4 interpretable ensemble types:
- **Lick-sensitive** (E03): highest R², encodes lick events
- **Head-direction** (E18): consistent head_angle encoding, ~70-80% of R² from HA alone
- **Spatial/position** (E02, E04): encode track_zone, spatial navigation
- **Visual-flow** (E05, E08): encode video-based motion features

### Section 4: Conditional Attribution
Generated for top-3 (session, ensemble) pairs (E01/S9, E02/S9, E16/S19):
- Shows GPV and IG per categorical level for each variable
- Key question: does the model use different features for cue_visible=0 vs 1?
- Outputs: `conditional_gpv_ENxx_Sxx.png`, `conditional_ig_ENxx_Sxx.png`

---

## Key Outputs (Desktop folder: `feature_correspondence_YYYYMMDD_HHMM/`)
| File | Description |
|------|-------------|
| `correspondence_heatmap.png` | Overall: which ensembles correlate with which behaviors |
| `correspondence_per_session_top4.png` | Session-level variability for top 4 ensembles |
| `alignment_heatmap.png` | Methods × groups: how well does attribution track correlation |
| `alignment_scatterplots.png` | Scatter: behavioral ρ vs attribution (top 3 groups) |
| `showcase_XX_*.png` | 9 individual case studies (time series + bars + scatter) |
| `conditional_gpv_*.png` | Attribution within each categorical level |
| `conditional_ig_*.png` | IG within each categorical level |
| `correspondence_summary.csv` | All results in one table |
| `alignment_stats.csv` | Alignment stats per method × group |

## Design Decisions & Why

### Why use all session data (not just test) for correspondence?
Behavioral correlations are estimated from data, and the test set (~20% of trials) may
have too few points for stable Spearman estimates, especially for rare categorical levels.
Using all data gives more reliable correlation estimates without data leakage concerns —
we're not evaluating model performance here, just measuring behavioral tuning.

### Why max |ρ| across columns in a group (not mean)?
For multi-level categorical groups (e.g., track_zone has 6 one-hot columns), different
levels may encode in opposite directions. Max |ρ| captures the strongest relationship.
For singleton continuous groups, there's only one column anyway.

### Conditional attribution rationale
If the model sees different features as important depending on the categorical condition,
it suggests the model has learned **interactions** between categorical context and other features.
For example: if head_angle attribution is higher when cue_visible=1 vs cue_visible=0, it
suggests the model uses head angle specifically during cue periods.

## Interpretation Notes
- **High alignment ρ for a method** = that method correctly identifies which features the
  ensemble is tuned to. Low alignment = the method finds "important" features that don't
  actually correlate with the ensemble activity.
- **Showcase triples** are the clearest examples of attribution methods "working" —
  ensembles with strong behavioral correspondence that's picked up by the method.
- **Conditional analysis** shows within-condition feature importance, useful for understanding
  multi-way interactions that the global attribution masks.
