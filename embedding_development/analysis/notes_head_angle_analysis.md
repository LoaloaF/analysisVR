# Notes: Head Angle Analysis
*Script: `head_angle_analysis.py`*

## What This Script Does

Deep dive into why `head_angle` is the dominant attribution feature across most MLP ensembles.
Tests whether this reflects genuine neural tuning or is an artifact of confounds.

---

## Key Findings (Run: 2026-05-17)

### Section 1: Head Angle Characterization
- Head angle varies substantially across sessions (different distributions per session)
- **Temporal autocorrelation is high** — head_angle is a slow, smooth signal that persists
  across many timepoints, making it easier for MLPs to fit (more informative per sample)
- **head_angle ~ movement_energy: low correlation** (Spearman ρ ≈ small)
- **head_angle ~ track_zone_int: ρ = -0.008** — essentially no correlation with spatial zone!
  This rules out head_angle being a proxy for the animal's position in the maze.

### Section 2: Attribution Ranking

Top ensembles by head_angle GPV (mean across sessions):
| Ensemble | GPV   | IG    | CPV   | Fraction of total | max R² |
|----------|-------|-------|-------|-------------------|--------|
| E06      | 0.135 | 0.239 | 0.058 | 47%               | 0.151  |
| E18      | 0.128 | 0.197 | 0.040 | 37%               | 0.236  |
| E15      | 0.099 | 0.242 | 0.048 | 32%               | 0.178  |
| E11      | 0.093 | 0.224 | 0.043 | 40%               | 0.127  |
| E23      | 0.087 | 0.229 | 0.039 | 33%               | 0.179  |

**Note**: E01, E07, E09, E12 show NaN (low R², excluded by R²≥0.01 threshold).

All 3 methods (GPV, IG, CPV) agree that head_angle is top or near-top for these ensembles.

### Section 4: Zero-Ablation Test
Run existing MLP (trained on all features) with:
- All features → full R²
- Only head_angle active (others zeroed) → tests how model responds to head_angle alone
- Without head_angle (zeroed) → tests impact of removing HA

| Session | Ensemble | Full R² | HA only | HA+vel only | No HA |
|---------|----------|---------|---------|-------------|-------|
| S29     | E18      | 0.246   | 0.121   | 0.120       | 0.064 |
| S1      | E03      | 0.449   | 0.005   | 0.006       | 0.334 |
| S20     | E23      | 0.188   | 0.034   | 0.061       | -0.084 |
| S21     | E06      | 0.143   | 0.040   | 0.054       | -0.069 |
| S20     | E15      | 0.196   | 0.044   | 0.086       | -0.029 |

**Important caveat**: Zero-ablation is misleading for multi-feature models — the model wasn't
trained with zeroed features, so activations break internal representations. Interpret with care.
**Notable exception**: E03 at S1 gets R²=0.449 (full) but only 0.005 with HA alone. Without HA:
still 0.334! This ensemble is NOT head-angle driven — it encodes lick_detected (confirmed by showcase).

### Section 5: Single-Feature MLP Training
**Cleanest test**: train a fresh 2-layer MLP (32 hidden units) with ONLY head_angle as input.
3 random seed replicates per pair.

| Session | Ensemble | Full R² | HA-only MLP | HA+vel MLP | % full explained |
|---------|----------|---------|-------------|------------|------------------|
| S29     | E18      | 0.222   | 0.155±0.00  | 0.187      | **70%**          |
| S1      | E03      | 0.462   | 0.193±0.00  | 0.265      | 42%              |
| S20     | E23      | 0.170   | 0.149±0.00  | 0.163      | **88%**          |
| S21     | E06      | 0.151   | 0.123±0.00  | 0.146      | **81%**          |
| S20     | E15      | 0.178   | 0.144±0.00  | 0.179      | **81%**          |

**Key result**: For most head-angle-attributed ensembles, **head_angle alone explains 70-88%
of the full model's R²** when given its own optimised model. This is strong evidence that
head_angle is genuinely the primary driver — not an attribution artifact.

**E03 at S1 is the exception**: only 42% — consistent with it being a lick_detected ensemble
(confirmed in the showcase analysis).

### Section 6: Confound Analysis — Partial Correlation
- Mean raw |Spearman ρ| between head_angle and ensemble: **0.085**
- Mean partial |r| controlling for movement_energy: **0.083**
- Drop: < 3% — head_angle's predictive value is essentially **independent of movement_energy**

- head_angle ~ track_zone: **ρ = -0.008** — no spatial confound

**Conclusion**: head_angle provides unique information about ensemble activity that is NOT
explained by the animal's speed/movement or its spatial position in the maze.

---

## Why Is Head Angle So Strong? — Summary Explanation

1. **It's genuinely informative**: 70-88% of full model R² is captured by head_angle alone
   with a dedicated single-feature MLP. The feature truly drives the ensemble.

2. **It's not a confound**: Partial correlation controlling for movement_energy barely changes.
   Head_angle ~ track_zone ρ ≈ 0, so it's not just a proxy for spatial location.

3. **It's temporally smooth (high autocorrelation)**: This makes it easier for MLPs to fit —
   nearby timepoints carry similar information, effectively increasing sample size.

4. **It has high variance across and within sessions**: Wide dynamic range gives the model
   more signal to exploit. Sessions with larger head_angle range tend to have higher R².

5. **Multiple ensembles are independently tuned to head_angle**: E06, E18, E23, E15, E11 all
   show independent head_angle encoding. This is consistent with head direction being a
   fundamental variable in hippocampal/entorhinal circuits.

---

## Interesting Individual Cases

### E18 (strongest head_angle encoder):
- GPV = 0.128, 37% of total attribution
- Best sessions S28, S29 (late sessions → possible learning effect)
- Single-feature MLP: 70% of full R² from head_angle alone

### E03 (the exception):
- High R² (0.462) but low head_angle attribution
- Showcase: lick_detected is the dominant feature
- Zero-ablation confirms: without head_angle, R² drops only from 0.449 to 0.334
- **E03 represents a DIFFERENT encoding type: lick-sensitive, not head-direction**

### E04, E02 (track_zone ensembles):
- Showcase analysis found these correlate with track_zone_int
- Attribution also reflects spatial encoding, not head_angle dominated
- These are position-encoding ensembles rather than direction-encoding

---

## Outputs (Desktop: `head_angle_analysis_YYYYMMDD_HHMM/`)
| File | Description |
|------|-------------|
| `ha_distribution_per_session.png` | Violin plot of head_angle per session |
| `ha_correlations.png` | Head_angle ρ with all other behavioral vars |
| `ha_autocorrelation.png` | Temporal autocorrelation (Session 1) |
| `ha_range_vs_r2.png` | Head angle range vs encoding quality per session |
| `ha_ranking_bar.png` | All ensembles ranked by head_angle attribution |
| `ha_fraction_of_total.png` | Head_angle attribution vs mean total attribution |
| `ha_scatter_grid.png` | 9-panel scatter: ensemble activity vs head_angle |
| `ha_corr_over_sessions_E18.png` | Session-by-session ρ for best ensemble |
| `ablation_r2_bars.png` | Zero-ablation R² comparison |
| `single_feat_mlp_scatter.png` | Full R² vs head_angle-only R² scatter |
| `single_feat_pct_explained.png` | % of full R² captured by head_angle alone |
| `partial_corr_scatter.png` | Raw ρ vs partial r (controlling movement) |
| `ha_by_track_zone.png` | Head_angle distribution by track zone |
| `single_feature_mlp_results.csv` | All single-feature MLP results |
| `ablation_results.csv` | All zero-ablation results |

---

## Follow-Up Questions / Future Work
1. **Head direction cells?** The E18/E06/E23 clusters may correspond to classical head-direction
   cells. Worth checking if they have consistent preferred head angles.
2. **Session-to-session drift**: Does head_angle encoding drift or remain stable across 29 sessions?
   The `evolution_heatmap_gpv.png` from `eval_mlp_attribution.py` addresses this.
3. **E03 lick ensemble**: Why does one ensemble encode licking? Is it a reward-related signal?
4. **Cond-PV is consistently lower than GPV/IG for head_angle**: This may indicate head_angle
   encodes information that the k-NN conditional already partially removes.
