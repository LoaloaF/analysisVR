# CLAUDE.md — analysisVR / embedding_development

## Mandatory: update slide manifest with every build

Whenever a new `eval/build_vN.py` is created and run, immediately update
`docs/slide_manifest.md` to reflect:
- The new version number and slide count in the header
- Any slides inserted, removed, or reordered (update all affected S-numbers and idx values)
- The figure→script→slide table
- The orphaned figures list (remove figures that are now in the deck; add figures
  that were in the deck but have been removed)
- Any new known anomalies

The manifest is the single source of truth for the current deck state. It must
stay in sync with the latest build output.

## Build chain convention

Builds are cumulative: each `build_vN.py` opens `ultimate_presentation_v(N-1).pptx`
and saves `ultimate_presentation_vN.pptx`. Always check which file the new build
opens (it may skip versions — e.g. v19 opens v17 directly).

Current latest: `ultimate_presentation_v51.pptx` (94 slides)

## Key paths

| Resource | Path |
|---|---|
| Presentation outputs | outputs/ultimate_presentation_vN.pptx |
| Figure outputs (MLP) | outputs/mlps/ensembles_multiseed/ |
| Figure outputs (CEBRA comparison) | outputs/cebra_comparison/ |
| Figure outputs (ml_vs_naive) | outputs/mlps/ml_vs_naive/ |
| Build scripts | eval/build_vN.py |
| Slide manifest | docs/slide_manifest.md |
| Slide narrative plan | docs/slide_plan_v2.md |
| Conda environment | analysisVR |

## Eval scripts and their figures

See `docs/slide_manifest.md` — FIGURE → SCRIPT COMPLETE MAPPING table.
Key orphaned figures not yet in the deck are listed under ORPHANED FIGURES.

## Feature naming

Always use canonical short names from `utils/figure_style.py` FEATURE_NAMES_SHORT:
  frame_raw_500msMedian → Fwd Speed
  head_angle → Head Angle
  frame_position → Track Pos.
  cue_visible → Cue Visible
  (etc — full list in slide_manifest.md header)

## python-pptx `_next_slide_partname` bug

**Always apply this patch in every build script that calls `add_slide`:**

```python
from pptx.parts.presentation import PresentationPart

@property
def _safe_next_slide_partname(self):
    return self.package.next_partname('/ppt/slides/slide%d.xml')

PresentationPart._next_slide_partname = _safe_next_slide_partname
```

**Why:** The default `_next_slide_partname` uses `len(sldIdLst)+1`, which conflicts with
existing slide XML parts when slides have been removed during the same build session.
The package's `next_partname()` correctly checks `iter_parts()` and returns a truly
available number. Omitting this patch causes "Duplicate name: slide84.xml" warnings
and a potentially corrupt PPTX.

## Consistency metric note

`eval_embedding_linear_map.py` computes the CEBRA-analogous geometric consistency
(ridge regression R² from embedding_A → embedding_B). This is the correct metric
and lives at S48 in v21.

The probe-prediction Pearson r metric (eval_cebra_consistency.py output) was
removed from the deck in build_v20.py — it is NOT CEBRA's consistency metric.
Do not reintroduce it as a consistency figure.
