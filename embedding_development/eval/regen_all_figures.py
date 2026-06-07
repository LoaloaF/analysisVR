#!/usr/bin/env python3
"""
eval/regen_all_figures.py  —  Universal figure regeneration script.

Runs every script that produces a figure currently in the deck (v45, 93 slides),
in dependency order.  Prints a per-script pass/fail summary and exits non-zero
if any script fails.

Usage:
    conda run -n analysisVR python eval/regen_all_figures.py

    # Re-run a single group only:
    conda run -n analysisVR python eval/regen_all_figures.py --group r2
    conda run -n analysisVR python eval/regen_all_figures.py --group attribution
    conda run -n analysisVR python eval/regen_all_figures.py --group casestudies

Groups (run in order within each group; groups themselves are independent):
  r2            R² bar charts and model comparisons   → S14,S15,S16,S18,S21,S22,S25–S29
  frequency     Trend/noise + freq trace example      → S35,S36
  attribution   GPV/CPV/IG heatmaps and comparisons  → S39,S41,S43,S44,S45,S46
  consistency   Embedding and attribution consistency → S49,S50,S51,S52
  head_angle    Head angle case study                 → S52,S54,S55,S56,S57
  position      Position case study                  → S59,S60,S61,S62
  validation    E07/E23 bucket validation            → S64,S65,S66
  e07           E07 cue-zone case study              → S68,S69,S70
"""

import argparse
import os
import subprocess
import sys
import time

BASE = os.path.dirname(os.path.abspath(__file__))

PYTHON = sys.executable  # same conda env that launched this script

# ── Figure → script table (slide number: script path, description) ─────────────
# Each entry: (group, script_relative_to_BASE, deck_slides, output_figures)
SCRIPTS = [
    # ── R² / model comparisons ─────────────────────────────────────────────────
    ('r2', 'generate_r2_figures.py',
     'S14,S21,S25', 'r2_bar_linear, r2_bar_mlp, r2_bar_tempconv_comparison'),

    ('r2', 'generate_ml_vs_naive_figures.py',
     'S15,S18,S22', 'nonlinearity_advantage, mlp_vs_linear_r2, ablation_proof'),

    ('r2', 'gen_new_analysis_figures.py',
     'S16,S27,S29,S51', 'joint_effect_scatter, r2_distribution_cdf, tempconv_delta_r2, cross_ensemble_consistency'),

    ('r2', 'eval_cebra_compare.py',
     'S26,S28', 'r2_comparison_grand_mean, two_thresholds_scatter'),

    # ── Frequency ──────────────────────────────────────────────────────────────
    ('frequency', 'eval_trend_noise.py',
     'S35', 'trend_noise_comparison'),

    ('frequency', 'eval_freq_trace_example.py',
     'S36', 'freq_trace_example'),

    # ── Attribution ────────────────────────────────────────────────────────────
    ('attribution', 'generate_gpv_task_ensembles.py',
     'S39', 'gpv_task_ensembles'),

    ('attribution', 'generate_attribution_figures.py',
     'S41,S43', 'global_vs_cond_pv_scatter, ig_per_ensemble_heatmap'),

    ('attribution', 'generate_group_attribution_comparison.py',
     'S44', 'group_attribution_comparison'),

    ('attribution', 'generate_attribution_agreement_figure.py',
     'S45', 'attribution_agreement_scatter'),

    ('attribution', 'generate_ml_vs_naive_figures.py',
     'S46', 'ml_vs_naive_scatter'),  # also covers S15,S18,S22 — idempotent

    ('attribution', 'eval_evolution_lineplots.py',
     'S47', 'evolution_lineplot_ig, evolution_lineplot_gpv'),

    # ── Consistency ────────────────────────────────────────────────────────────
    ('consistency', 'gen_attribution_consistency.py',
     'S48', 'attribution_consistency_merged'),

    ('consistency', 'eval_embedding_linear_map.py',
     'S49', 'embedding_linear_map'),

    ('consistency', 'eval_cross_ensemble_prediction.py',
     'S50', 'cross_ensemble_prediction'),

    # ── Head angle case study ──────────────────────────────────────────────────
    ('head_angle', 'gen_head_angle_attribution_summary.py',
     'S52', 'head_angle_attribution_summary'),

    ('head_angle', 'gen_head_angle_tuning_variety.py',
     'S54', 'head_angle_tuning_variety'),

    ('head_angle', 'eval_head_angle_scatter.py',
     'S55', 'head_angle_scatter_2x2'),

    ('head_angle', 'eval_head_angle_stability.py',
     'S56,S57', 'head_angle_stability, head_angle_tuning_6panel'),

    # ── Position case study ────────────────────────────────────────────────────
    ('position', 'eval_position_tuning.py',
     'S59', 'position_tuning'),

    ('position', 'eval_position_proxy.py',
     'S60', 'position_collinearity'),

    ('position', 'eval_position_ablation.py',
     'S61,S62', 'position_ablation'),

    # ── E07/E23 validation ─────────────────────────────────────────────────────
    ('validation', 'e07_e23_bucket_analysis.py',
     'S64,S65,S66', 'e07_e23_bucket_signal, e07_e23_bucket_gpv, e07_e23_bucket_covariation'),

    # ── E07 cue-zone case study ────────────────────────────────────────────────
    ('e07', 'eval_e07_cue_zone_tuning.py',
     'S68', 'e07_cue_zone_tuning'),

    ('e07', 'eval_e07_cue_zone_joint_gpv.py',
     'S69', 'e07_cue_zone_joint_gpv'),

    ('e07', 'eval_e07_cue_zone_competition.py',
     'S70', 'e07_cue_zone_competition'),
]


def run_script(script_path):
    """Run a single eval script; return (returncode, elapsed_seconds, stderr_tail)."""
    full = os.path.join(BASE, script_path)
    if not os.path.exists(full):
        return None, 0, f'FILE NOT FOUND: {full}'
    t0 = time.time()
    result = subprocess.run(
        [PYTHON, full],
        capture_output=True,
        text=True,
        cwd=os.path.dirname(BASE),  # repo root
    )
    elapsed = time.time() - t0
    stderr_tail = '\n'.join(result.stderr.strip().splitlines()[-6:]) if result.stderr.strip() else ''
    return result.returncode, elapsed, stderr_tail


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--group', default='all',
                        help='Which group to run (default: all)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print scripts that would run without executing')
    args = parser.parse_args()

    target_group = args.group.lower()
    rows = SCRIPTS if target_group == 'all' else [r for r in SCRIPTS if r[0] == target_group]

    if not rows:
        print(f'No scripts found for group "{target_group}".')
        print(f'Available groups: {sorted({r[0] for r in SCRIPTS})}')
        sys.exit(1)

    print(f'{"DRY RUN — " if args.dry_run else ""}Regenerating {len(rows)} script(s)'
          + (f' in group "{target_group}"' if target_group != 'all' else '') + '\n')

    results = []
    for group, script, slides, figures in rows:
        label = f'[{group}] {script}'
        print(f'  Running {label}  (→ {slides}) ... ', end='', flush=True)
        if args.dry_run:
            print('(skipped)')
            results.append((script, 'dry-run', 0, ''))
            continue
        rc, elapsed, stderr = run_script(script)
        if rc is None:
            status = 'MISSING'
            print(f'MISSING')
        elif rc == 0:
            status = 'OK'
            print(f'OK  ({elapsed:.1f}s)')
        else:
            status = 'FAIL'
            print(f'FAIL  ({elapsed:.1f}s)')
            if stderr:
                for line in stderr.splitlines():
                    print(f'    | {line}')
        results.append((script, status, elapsed, stderr))

    # Summary
    print()
    ok    = sum(1 for _, s, _, _ in results if s == 'OK')
    fail  = sum(1 for _, s, _, _ in results if s == 'FAIL')
    miss  = sum(1 for _, s, _, _ in results if s == 'MISSING')
    total = len(results)
    print(f'Result: {ok}/{total} OK,  {fail} failed,  {miss} missing')
    sys.exit(1 if (fail + miss) > 0 else 0)


if __name__ == '__main__':
    main()
