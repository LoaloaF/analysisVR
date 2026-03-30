#!/usr/bin/env python3
"""Train session-level interval decoders and save compact result tables."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Sequence


def _default_excluded_sessions() -> List[str]:
    # Keep the same default session exclusions used by the current workflow.
    return [
        "2024-11-29_17-21_rYL006_P1100_LinearTrackStop_28min",
        "2025-01-22_17-51_rYL006_P1100_LinearTrackStop_5min",
        "2025-01-21_18-49_rYL006_P1100_LinearTrackStop_30min",
    ]


def _extract_session_date(session_name: str) -> Optional[datetime]:
    # Parse the session date token directly from the session name.
    for token in str(session_name).split("_"):
        try:
            return datetime.strptime(token, "%Y-%m-%d")
        except ValueError:
            continue
    return None


def _filter_sessions_by_min_date(
    session_names: Sequence[str],
    min_session_date: Optional[str],
) -> tuple[List[str], List[str]]:
    # Apply the same session-name date filtering used before loading analytics.
    if not min_session_date:
        return list(session_names), []

    cutoff = datetime.strptime(str(min_session_date), "%Y-%m-%d")
    kept, dropped = [], []
    for session_name in session_names:
        session_date = _extract_session_date(session_name)
        if session_date is None or session_date < cutoff:
            dropped.append(session_name)
        else:
            kept.append(session_name)
    return kept, dropped


def _assembly_sort_key(name: str) -> tuple[int, object]:
    # Sort assembly names numerically when decoding all assemblies.
    match = re.fullmatch(r"Assembly(\d+)", str(name))
    if match:
        return (0, int(match.group(1)))
    return (1, str(name))


def _resolve_assembly_cols(df, assembly_arg: str) -> List[str]:
    # Expand `all` into the available assembly columns once per run.
    assembly_arg = str(assembly_arg).strip()
    if assembly_arg.lower() not in {"all", "*"}:
        return [assembly_arg]

    assemblies = sorted(
        [str(col) for col in df.columns if re.fullmatch(r"Assembly\d{3}", str(col))],
        key=_assembly_sort_key,
    )
    if not assemblies:
        raise RuntimeError("No assembly columns matching 'Assembly###' were found in EnsembleT0Projection.")
    return assemblies


def _save_outputs(run_dir: Path, results_df, cfg_dict: dict[str, object]) -> str:
    # Save only the compact results table and the config used to produce it.
    run_dir.mkdir(parents=True, exist_ok=True)
    results_csv = run_dir / "results.csv"
    results_df.to_csv(results_csv, index=False)
    (run_dir / "config.json").write_text(json.dumps(cfg_dict, indent=2, default=str))
    return str(results_csv)


def _build_arg_parser() -> argparse.ArgumentParser:
    # Keep the CLI focused on the knobs still used by the compact decoder.
    parser = argparse.ArgumentParser(description="Train session-level interval decoding models.")
    parser.add_argument("--animal-id", type=int, default=6)
    parser.add_argument("--paradigm-id", type=int, default=1100)
    parser.add_argument("--assembly-col", type=str, default="Assembly012")
    parser.add_argument("--min-trials-per-session", type=int, default=20)
    parser.add_argument("--min-class-count", type=int, default=4)
    parser.add_argument("--max-cv-splits", type=int, default=5)
    parser.add_argument("--shuffle-n", type=int, default=1000)
    parser.add_argument(
        "--output-root",
        type=str,
        default="scripted_plotting/animal_6_analysis/interval_session_decoding_runs",
    )
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument("--exclude-session", action="append", default=[])
    parser.add_argument("--min-session-date", type=str, default="2024-11-27")
    parser.add_argument("--disable-session-date-filter", action="store_true")
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()

    # Set up output paths and local imports from the current checkout.
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name.strip() or f"run_{ts}"
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    script_path = Path(__file__).resolve()
    analysisvr_root = script_path.parents[2]
    workspace_root = analysisvr_root.parent
    for path in (analysisvr_root, workspace_root):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)

    from baseVR.base_functionality import init_import_paths

    init_import_paths()

    from CustomLogger import CustomLogger as Logger
    from analytics_processing import analytics
    from analytics_processing.sessions_from_nas_parsing import (
        fullfnames2snames,
        sessionlist_fullfnames_from_args,
    )
    from interval_session_decoding_pipeline import SessionDecodeConfig, run_session_interval_decoding

    Logger().init_logger(None, None, logging_level="DEBUG")

    # Collect the sessions requested for this run and apply the date filter.
    excluded = _default_excluded_sessions() + list(args.exclude_session)
    session_dirs = sessionlist_fullfnames_from_args(
        [args.paradigm_id],
        [args.animal_id],
        None,
        excl_session_names=excluded,
    )[0]
    session_names = fullfnames2snames(session_dirs)

    min_session_date = None if args.disable_session_date_filter else args.min_session_date
    filtered_session_names, dropped_session_names = _filter_sessions_by_min_date(session_names, min_session_date)

    if min_session_date:
        print(
            f"[load] session_date_filter>={min_session_date} "
            f"kept={len(filtered_session_names)} dropped={len(dropped_session_names)}",
            flush=True,
        )
    if not filtered_session_names:
        raise RuntimeError("No sessions left after date filtering. Adjust --min-session-date.")

    # Load the ensemble analytic once and reuse it across assemblies.
    print(f"[load] sessions={len(filtered_session_names)} feature={args.assembly_col}", flush=True)
    t0_ens = analytics.get_analytics("EnsembleT0Projection", session_names=filtered_session_names)
    print(f"[load] t0_ens shape={t0_ens.shape}", flush=True)

    assembly_cols = _resolve_assembly_cols(t0_ens, args.assembly_col)
    print(f"[load] assemblies={assembly_cols}", flush=True)

    completed = 0

    # Train and save one compact result table per requested assembly.
    for assembly_col in assembly_cols:
        cfg = SessionDecodeConfig(
            assembly_col=assembly_col,
            min_trials_per_session=args.min_trials_per_session,
            min_class_count=args.min_class_count,
            max_cv_splits=args.max_cv_splits,
            random_state=42,
            shuffle_n=max(0, int(args.shuffle_n)),
            min_session_date=min_session_date,
            drop_unparseable_session_dates=True,
        )

        run_dir_name = run_name if len(assembly_cols) == 1 else f"{run_name}__{assembly_col}"
        run_dir = output_root / run_dir_name

        print(f"[train] assembly={assembly_col} output_dir={run_dir}", flush=True)
        print(f"[train] shuffle_n={int(cfg.shuffle_n)}", flush=True)

        results_df = run_session_interval_decoding(
            t0_ens,
            cfg=cfg,
            progress=True,
            log_every=25,
        )

        results_csv = _save_outputs(run_dir, results_df, asdict(cfg))

        print(f"[done] assembly={assembly_col}", flush=True)
        print(f"  rows(results)={len(results_df)}", flush=True)
        print(f"  results_csv={results_csv}", flush=True)
        completed += 1

    # Print a short summary after the full assembly loop finishes.
    print(
        f"[summary] completed={completed} requested={len(assembly_cols)} "
        f"sessions_requested={len(session_names)} sessions_after_date_filter={len(filtered_session_names)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
