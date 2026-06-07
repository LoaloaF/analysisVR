#!/usr/bin/env python3
"""
eval/check_overlaps.py

Standalone overlap audit for all manifest figures.

Runs each figure-generation script in a subprocess and collects the [OVERLAP]
warnings printed by savefig_manifest's built-in renderer-based checker.
Each check covers five failure modes:
  1. Adjacent x-tick labels whose bboxes intersect
  2. Y-axis label that extends past the left canvas edge (clipped label)
  3. X-axis label that extends past the bottom canvas edge
  4. Legend bbox that intersects an annotation text bbox
  5. Two axes bboxes that overlap >10 % of the smaller axis area

Usage:
    python eval/check_overlaps.py            # full audit (≈ 2-4 min, loads data)
    python eval/check_overlaps.py --quick    # skip data-heavy scripts

Exit code: 0 if no overlaps, 1 if any found.
"""
import argparse
import os
import subprocess
import sys
import time

# ── Figure scripts that generate manifest PNGs ─────────────────────────────────
# Each entry: (script_path, [png_filenames_it_produces], is_data_heavy)
SCRIPTS = [
    # fast — load pre-computed .npy arrays only
    ('eval/generate_attribution_figures.py',
     ['gpv_group_ensemble_heatmap.png', 'ig_per_ensemble_heatmap.png',
      'global_vs_cond_pv_scatter.png'],
     False),

    ('eval/generate_case_studies.py',
     ['case_studies_e07_e23.png'],
     False),

    ('eval/generate_attribution_agreement_figure.py',
     ['attribution_agreement_scatter.png'],
     False),

    ('eval/generate_group_attribution_comparison.py',
     ['group_attribution_comparison.png'],
     False),

    ('eval/generate_r2_figures.py',
     ['r2_bar_linear.png', 'r2_bar_mlp.png',
      'r2_grand_mean_bars.png', 'two_thresholds_scatter.png'],
     False),

    ('eval/generate_consistency_figure.py',
     ['r2_consistency_bar.png'],
     False),

    # data-heavy — load session_dataset_ensembles.pkl (≥ 10 s each)
    ('eval/eval_head_angle_scatter.py',
     ['head_angle_scatter_2x2.png'],
     True),

    ('eval/eval_head_angle_stability.py',
     ['head_angle_stability.png', 'head_angle_tuning_6panel.png'],
     True),

    ('eval/eval_position_tuning.py',
     ['position_tuning.png'],
     True),

    ('eval/generate_behavioral_trace.py',
     ['behavioral_trace.png'],
     True),

    ('eval/eval_trend_noise.py',
     ['trend_noise_comparison.png'],
     True),

    ('eval/generate_ml_vs_naive_figures.py',
     ['ml_vs_naive_scatter.png', 'nonlinearity_advantage.png',
      'mlp_vs_linear_r2.png', 'ablation_proof.png'],
     True),
]

CONDA_ENV = 'analysisVR'

# ── Helpers ────────────────────────────────────────────────────────────────────

def run_script(script_path, root):
    """Run one generation script via conda and return (stdout, stderr, elapsed)."""
    t0 = time.time()
    result = subprocess.run(
        ['conda', 'run', '--no-capture-output', '-n', CONDA_ENV,
         'python', script_path],
        capture_output=True, text=True,
        cwd=root,
        timeout=300,
    )
    return result.stdout, result.stderr, time.time() - t0


def parse_overlaps(stdout):
    """Extract [OVERLAP] lines from script stdout."""
    return [ln.strip() for ln in stdout.splitlines() if '[OVERLAP]' in ln]


def _bar(n, width=50):
    filled = int(width * n / max(n, 1))
    return '█' * filled + '░' * (width - filled)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--quick', action='store_true',
                        help='Skip data-heavy scripts (no session_dataset loading)')
    args = parser.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    scripts = [(s, pngs, heavy) for s, pngs, heavy in SCRIPTS
               if not (args.quick and heavy)]

    print(f"\n{'='*65}")
    print(f"Overlap audit — {len(scripts)} scripts"
          + (" (quick mode: data-heavy scripts skipped)" if args.quick else ""))
    print(f"{'='*65}\n")

    all_overlaps = []   # list of (script, [overlap_msg])
    errors = []

    for script, pngs, heavy in scripts:
        label = os.path.basename(script)
        print(f"  Running {label} ...", end='', flush=True)

        try:
            stdout, stderr, elapsed = run_script(script, root)
        except subprocess.TimeoutExpired:
            print(f"  [TIMEOUT]")
            errors.append(f"{label}: timed out after 300 s")
            continue
        except Exception as e:
            print(f"  [ERROR] {e}")
            errors.append(f"{label}: {e}")
            continue

        overlaps = parse_overlaps(stdout)

        status = 'FAIL' if overlaps else 'ok'
        print(f"  {status}  ({elapsed:.1f}s)")

        if overlaps:
            all_overlaps.append((label, overlaps))
            for msg in overlaps:
                # Indent the raw message for readability
                body = msg.replace('[OVERLAP]', '').strip()
                print(f"    [OVERLAP] {body}")

        # Surface script errors (not [OVERLAP] lines)
        if stderr.strip():
            # Only show genuine errors, not harmless matplotlib deprecation warnings
            err_lines = [ln for ln in stderr.splitlines()
                         if 'Error' in ln or 'Traceback' in ln]
            if err_lines:
                print(f"    [STDERR] {err_lines[0][:120]}")
                errors.append(f"{label}: {err_lines[0][:80]}")

    # ── Summary ────────────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    n_fail = len(all_overlaps)
    n_pass = len(scripts) - n_fail - len(errors)
    print(f"Results: {n_pass} passed, {n_fail} overlap failures, {len(errors)} errors")

    if all_overlaps:
        print("\nOverlap failures:")
        for script_name, msgs in all_overlaps:
            print(f"  {script_name}:")
            for m in msgs:
                body = m.replace('[OVERLAP]', '').strip()
                print(f"    • {body}")

    if errors:
        print("\nScript errors:")
        for e in errors:
            print(f"  • {e}")

    if not all_overlaps and not errors:
        print("All figures passed overlap checks.")
    print(f"{'='*65}\n")

    return 1 if (all_overlaps or errors) else 0


if __name__ == '__main__':
    sys.exit(main())
