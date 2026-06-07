# Evaluation Plan

Covers three levels:
1. **Computation correctness** — is the analysis computing the right thing?
2. **Figure correctness** — does the figure faithfully represent the computation?
3. **Slide correctness** — does the slide render correctly against the manifest?

Each check is marked as AUTOMATED (can be scripted) or VISUAL (requires human inspection).
Run in order: computation → figure → slide. Do not proceed to the next level if the current level fails.

---

## Level 1 — Computation Correctness

These checks run before any figure is generated, directly on the output arrays.

### 1.1 Validity masks
- [ ] AUTOMATED: For every `all_r2.npy` loaded, assert shape is `(5, 29, 23)` (seeds × sessions × ensembles)
- [ ] AUTOMATED: `valid_mask = np.all(np.isnan(all_r2), axis=0)` — confirm mask uses `np.all` not `np.any` (the known bug). Check `(~valid_mask).sum()` > 0 for all models
- [ ] AUTOMATED: Valid pair count per model: MLP ~363, TempConv-Cont ~same, TempConv-Pred ~same. Flag if any model has >20% fewer valid pairs than MLP

### 1.2 Attribution arrays
- [ ] AUTOMATED: GPV shape: `(29, 23, 11)`. Values in `[0, 1]` for valid pairs, NaN for invalid. No negative values (permutation importance is a loss; negative = model is worse without the feature, which is the expected direction — but confirm sign convention)
- [ ] AUTOMATED: IG shape: `(29, 23, 11)`. Values should be positive (we use |IG|). Check `np.nanmin(ig_array) >= 0`
- [ ] AUTOMATED: Feature ordering: assert `semantic_groups.pkl` feature order matches `FEATURE_NAMES` key order in `utils/figure_style.py`. Print both and compare manually once

### 1.3 Head angle tuning (S23, S26)
- [ ] AUTOMATED: For each (session, ensemble) pair, confirm decile bins span the actual data range. Check `bin_edges[0] ≈ data.min()` and `bin_edges[-1] ≈ data.max()` within 5%
- [ ] AUTOMATED: Confirm head angle range in data. Expected: roughly −2 to 2 cm based on S36 x-axes. Print `np.percentile(head_angle_all, [1, 99])` and flag if outside [−3, 3]
- [ ] AUTOMATED: Each tuning curve has ≥ 5 data points per bin (otherwise the curve is unreliable). Flag bins with n < 5
- [ ] VISUAL: Do the 6 selected panels for S23 show visually distinct shapes? Must include at least: 1 inverted-U (centre-tuning), 1 U-shape (edge-tuning), 1 monotone. If all curves look flat → binning or session selection is wrong

### 1.4 Head angle 2D scatter (S25)
- [ ] AUTOMATED: x-axis (head angle) range matches 1.3 above
- [ ] AUTOMATED: y-axis (head angular velocity) — print range, confirm it is not suspiciously narrow (e.g., all zeros would indicate a data loading error)
- [ ] AUTOMATED: Color (ensemble activation) — confirm values come from the z-scored ensemble activity, not raw. Check `np.nanstd(activation) ≈ 1.0` for the session used
- [ ] VISUAL: Scatter shows a 2D structure (not a uniform cloud). If the cloud has no gradient in color → the ensemble is not tuned in that session; choose a different session

### 1.5 Position tuning (S27)
- [ ] AUTOMATED: x-axis spans actual position range. Print `np.percentile(position_all, [0.5, 99.5])`. Expected: −169 to 270 cm (supervisor confirmed). Flag if range is less than 300 cm total
- [ ] AUTOMATED: Confirm position values are in cm, not in some normalized unit. If max position ≈ 1.0 → data is normalized; check preprocessing and use unnormalized position
- [ ] AUTOMATED: For each (session, ensemble) pair plotted: confirm the ensemble has R² ≥ 0.01 (is a valid pair)
- [ ] VISUAL: At least one of the plotted tuning curves should show a non-flat profile (a hump or dip). If all curves are flat → either the model cannot find position tuning (the point of the slide) or the data is wrong. Distinguish these cases: compute η² for position separately and report it

### 1.6 Frequency decomposition (S28)
- [ ] AUTOMATED: Slow component = 500 ms moving average of the prediction (12-13 bins at 40 ms). Confirm window size: `window = int(0.5 / 0.040) = 12 bins`. Fast = original − slow
- [ ] AUTOMATED: Slow + fast component variances should approximately sum to total signal variance. Check `Var(slow) + Var(fast) ≈ Var(signal)` within 10% for each pair
- [ ] AUTOMATED: For MLP predictions: slow R² should be in range 0.4–0.8 (consistent with S39). Fast R² should be near 0. Flag any pair where MLP fast R² > 0.1
- [ ] AUTOMATED: Same decomposition applied identically to MLP and TempConv predictions — confirm same window size, same session, same ensemble index
- [ ] AUTOMATED: TempConv and MLP use the SAME test-set predictions for the same random seed. Confirm seed used for comparison is consistent (use seed 42 as reference)
- [ ] VISUAL: The bar chart should show MLP slow bars clearly taller than MLP fast bars. If they are equal → decomposition is wrong

---

## Level 2 — Figure Correctness

Run after Level 1 passes. Check each generated PNG before it enters the PPTX.

### 2.1 Automated figure checks (run via a validation script on every output PNG)
Write `eval/validate_figures.py` that loads each PNG and checks:
- [ ] AUTOMATED: Native size matches manifest figsize ± 0.1 inch (at 200 dpi: ± 20 pixels). Use PIL: `img.size[0] / 200 ≈ figsize[0]`
- [ ] AUTOMATED: Image is not blank (mean pixel value < 250 out of 255 — i.e., not all white)
- [ ] AUTOMATED: Image dimensions are landscape (width > height) for all full-width figures

### 2.2 Visual figure checklist (inspect each PNG at 100% zoom before inserting)
For every figure:
- [ ] VISUAL: No matplotlib `fig.suptitle()` or `ax.set_title()` visible in the image
- [ ] VISUAL: All axis labels use canonical names from `FEATURE_NAMES` / `AXIS_LABELS` — no raw column names (`frame_raw_500msMedian` etc.)
- [ ] VISUAL: Tick labels are readable at the figure's native size. Rule of thumb: open the PNG at 100% — if you need to squint at a tick label, the font is too small
- [ ] VISUAL: Legend entries use canonical model names (`MLP`, `TempConv-Cont`, `TempConv-Pred`) — no code identifiers
- [ ] VISUAL: No annotation text boxes with arrows inside the axes. Statistics (ρ, p, n) appear either in the technical footnote (outside axes, 11pt) or as plain text in one corner of the axes (no box border, no arrow)
- [ ] VISUAL: Colorbar label is present and uses canonical `AXIS_LABELS` string

### 2.3 Per-figure specific checks

| Figure | Specific check |
|---|---|
| S06 behavioral trace | y-axis labels show units in parentheses; position panel y-axis shows ticks at 0 and 270 |
| S10 linear bar | bars are visually near-zero; y-axis ceiling ≤ 0.15 so the emptiness is obvious |
| S12 MLP bar | E08 and E18 labelled at top; mako_r colormap runs light→dark left→right |
| S19 GPV heatmap | head_angle row is visually the darkest; no raw feature names on y-axis |
| S20 Cond-PV scatter | head_angle point is clearly the furthest from origin AND far from diagonal; diagonal line is dashed |
| S23 tuning curves | 6 panels, 3×2 grid; each panel has its own y-axis scale; x-axis label "Head Angle (cm)" appears only on bottom row |
| S25 2D scatter | colorbar present; axes not square; color gradient is visible (not all one color) |
| S26 stability | overlaid curves are semi-transparent; one bold mean curve; right panel bars show small spread |
| S27 position tuning | x-axis range −169 to 270 with ticks at 0 and every 100 cm |
| S28 frequency bars | slow bars clearly taller than fast bars for MLP; TempConv fast bars either higher than MLP or same |
| S29 performance | error bars are ± SD not ± SE (confirm in code); bars are approximately the same height across models |
| S30 attribution scatter | diagonal dashed line "y = x"; both panels have same axis limits; ρ value in upper-left as plain text |

---

## Level 3 — Slide Correctness

Run after all figures pass Level 2. Check the PPTX before presenting.

### 3.1 Automated PPTX checks
Write `eval/validate_pptx.py` that opens the output PPTX and checks:
- [ ] AUTOMATED: Total slide count matches manifest (32 main + N backup)
- [ ] AUTOMATED: For every slide with a figure: shape width and height in EMU matches expected figsize × 914400 EMU/inch ± 1% (confirms no rescaling occurred)
- [ ] AUTOMATED: Every slide has exactly one title placeholder (or text box in title position) with non-empty text
- [ ] AUTOMATED: Citation markers [N] present in slide text where specified in manifest. Check slide numbers: S14→[1], S15→[1], S16→[2]+[3], S17→[2]+[3], S18→[2], S21→[2], S05→[4]
- [ ] AUTOMATED: Backup slides are present and hidden (`slide.shapes` accessible but slide is in the hidden range)

### 3.2 Visual slide checklist (go through every slide at presentation zoom)
- [ ] VISUAL: Slide title is bold
- [ ] VISUAL: Slide title does not repeat text that is already the figure's axis label or subplot title
- [ ] VISUAL: Figure fills its designated area without whitespace gaps at edges
- [ ] VISUAL: Technical footnotes (if present) are visible at 11pt but clearly subordinate to the main figure
- [ ] VISUAL: On slides with both a figure and bullets: bullets are in a text box that does not overlap the figure
- [ ] VISUAL: Animation slides (S02 video, S11 MLP demo, S24 training evolution): animations play on click, not autoplay (unless S02 which should autoplay)
- [ ] VISUAL: No slide has a horizontal scrollbar in normal view (figure too wide)

### 3.3 Projection simulation check
Before the supervisor meeting, run this once:
- [ ] VISUAL: Set PowerPoint to slideshow mode, advance through every slide
- [ ] VISUAL: On each slide, check that tick labels and axis labels are readable from 2m distance (simulate by stepping back from your monitor)
- [ ] VISUAL: Bold slide titles are clearly legible
- [ ] VISUAL: Citation markers [N] are visible but do not dominate

---

## Evaluation Schedule (integrated into execution phases)

| Phase | Eval level | When |
|---|---|---|
| Phase 0c (propagate naming) | Level 2.2 (visual check) | After each script is updated, regenerate its figure and check names |
| Phase 1a (head angle scatter) | Level 1.4 then Level 2.2 + 2.3 | Before committing the new script |
| Phase 1b (stability) | Level 1.3 then Level 2.2 + 2.3 | Before committing |
| Phase 1c (position tuning) | Level 1.5 then Level 2.2 + 2.3 | Before committing |
| Phase 1d (frequency analysis) | Level 1.6 then Level 2.2 + 2.3 | FIRST — result determines S28 message |
| Phase 2 (regenerate all) | Level 2.1 (automated) | Run validate_figures.py on entire outputs directory |
| Phase 3 (PPTX rebuild) | Level 3.1 (automated) then 3.2 (visual) | After PPTX is generated |
| Pre-meeting | Level 3.3 (projection simulation) | Day before meeting |

---

## Red Flags — Stop and Investigate

These indicate a fundamental error, not just a cosmetic issue:

- Position tuning x-axis max < 5 → position is normalized, not in cm
- Head angle range outside [−5, 5] → wrong feature column loaded
- MLP grand mean R² outside [0.03, 0.06] → different validity mask than expected
- TempConv valid pair count less than 200 → `.any()` mask bug has returned
- Any attribution array with values > 1.0 → not a ΔR² value, wrong array loaded
- Cross-seed Pearson r < 0.5 for E18 → consistency analysis is comparing wrong sessions
- Slow R² > fast R² is FALSE for MLP → frequency decomposition is swapped
