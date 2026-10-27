#!/usr/bin/env python3
"""Train interval decoders and save compact result tables."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence


def _default_excluded_sessions() -> List[str]:
    return [
        "2024-11-29_17-21_rYL006_P1100_LinearTrackStop_28min",
        "2025-01-22_17-51_rYL006_P1100_LinearTrackStop_5min",
        "2025-01-21_18-49_rYL006_P1100_LinearTrackStop_30min",
    ]


def _extract_session_date(session_name: str) -> Optional[datetime]:
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


def _save_outputs(
    run_dir: Path,
    results_df,
    diagnostics_df,
    meta: Dict[str, object],
    cfg_dict: Dict[str, object],
) -> Dict[str, object]:
    run_dir.mkdir(parents=True, exist_ok=True)
    results_csv = run_dir / "results.csv"
    diagnostics_csv = run_dir / "diagnostics.csv"
    results_df.to_csv(results_csv, index=False)
    diagnostics_df.to_csv(diagnostics_csv, index=False)

    parquet_status = "ok"
    try:
        results_df.to_parquet(run_dir / "results.parquet", index=False)
        diagnostics_df.to_parquet(run_dir / "diagnostics.parquet", index=False)
    except Exception as err:
        parquet_status = f"failed: {type(err).__name__}: {err}"

    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
    (run_dir / "config.json").write_text(json.dumps(cfg_dict, indent=2, default=str))
    return {
        "results_csv": str(results_csv),
        "diagnostics_csv": str(diagnostics_csv),
        "parquet_status": parquet_status,
    }


def _assembly_sort_key(name: str) -> tuple[int, object]:
    match = re.fullmatch(r"Assembly(\d+)", str(name))
    if match:
        return (0, int(match.group(1)))
    return (1, str(name))


def _resolve_assembly_cols(df, assembly_arg: str) -> List[str]:
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


def _run_parallel_children(args, run_name: str, assembly_cols: Sequence[str]) -> None:
    script_path = Path(__file__).resolve()
    pending = list(assembly_cols)
    running = []
    failures = []

    while pending or running:
        while pending and len(running) < args.n_jobs:
            assembly_col = pending.pop(0)
            child_cmd = [
                sys.executable,
                str(script_path),
                "--animal-id",
                str(args.animal_id),
                "--paradigm-id",
                str(args.paradigm_id),
                "--assembly-col",
                str(assembly_col),
                "--window-half-width",
                str(args.window_half_width),
                "--min-samples-per-time",
                str(args.min_samples_per_time),
                "--min-class-count",
                str(args.min_class_count),
                "--max-cv-splits",
                str(args.max_cv_splits),
                "--shuffle-n",
                str(args.shuffle_n),
                "--bootstrap-n",
                str(args.bootstrap_n),
                "--bootstrap-ci",
                str(args.bootstrap_ci),
                "--output-root",
                str(args.output_root),
                "--run-name",
                f"{run_name}__{assembly_col}",
                "--n-jobs",
                "1",
                "--min-session-date",
                str(args.min_session_date),
            ]
            for session_name in args.exclude_session:
                child_cmd.extend(["--exclude-session", str(session_name)])
            if args.disable_session_date_filter:
                child_cmd.append("--disable-session-date-filter")
            if args.skip_existing:
                child_cmd.append("--skip-existing")

            print(f"[spawn] assembly={assembly_col}", flush=True)
            running.append((assembly_col, subprocess.Popen(child_cmd)))

        for idx, (assembly_col, proc) in enumerate(running):
            returncode = proc.poll()
            if returncode is None:
                continue
            running.pop(idx)
            if returncode:
                failures.append((assembly_col, returncode))
            break
        else:
            time.sleep(0.1)

    if failures:
        raise RuntimeError(f"Failed assemblies: {failures}")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train interval decoding models.")
    parser.add_argument("--animal-id", type=int, default=6)
    parser.add_argument("--paradigm-id", type=int, default=1100)
    parser.add_argument("--assembly-col", type=str, default="Assembly012")
    parser.add_argument("--window-half-width", type=int, default=1)
    parser.add_argument("--min-samples-per-time", type=int, default=20)
    parser.add_argument("--min-class-count", type=int, default=4)
    parser.add_argument("--max-cv-splits", type=int, default=5)
    parser.add_argument("--shuffle-n", type=int, default=0)
    parser.add_argument("--bootstrap-n", type=int, default=0)
    parser.add_argument("--bootstrap-ci", type=float, default=0.95)
    parser.add_argument(
        "--output-root",
        type=str,
        default="scripted_plotting/animal_6_analysis/interval_decoding_runs",
    )
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument("--exclude-session", action="append", default=[])
    parser.add_argument("--min-session-date", type=str, default="2024-11-27")
    parser.add_argument("--disable-session-date-filter", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--n-jobs", type=int, default=1)
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()

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
    from interval_decoding_pipeline import DecodeConfig, run_interval_decoding

    Logger().init_logger(None, None, logging_level="DEBUG")

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

    print(f"[load] sessions={len(filtered_session_names)} feature={args.assembly_col}", flush=True)
    t0_ens = analytics.get_analytics("EnsembleT0Projection", session_names=filtered_session_names)
    print(f"[load] t0_ens shape={t0_ens.shape}", flush=True)

    assembly_cols = _resolve_assembly_cols(t0_ens, args.assembly_col)
    print(f"[load] assemblies={assembly_cols}", flush=True)

    if args.n_jobs > 1 and len(assembly_cols) > 1:
        _run_parallel_children(args, run_name, assembly_cols)
        print(f"[summary] completed={len(assembly_cols)} requested={len(assembly_cols)}", flush=True)
        return

    completed = 0
    for assembly_col in assembly_cols:
        cfg = DecodeConfig(
            assembly_col=assembly_col,
            window_half_width=args.window_half_width,
            min_samples_per_time=args.min_samples_per_time,
            min_class_count=args.min_class_count,
            max_cv_splits=args.max_cv_splits,
            random_state=42,
            shuffle_n=max(0, int(args.shuffle_n)),
            bootstrap_n=max(0, int(args.bootstrap_n)),
            bootstrap_ci=float(args.bootstrap_ci),
            min_session_date=min_session_date,
            drop_unparseable_session_dates=True,
        )

        run_dir_name = run_name if len(assembly_cols) == 1 else f"{run_name}__{assembly_col}"
        run_dir = output_root / run_dir_name
        if args.skip_existing and (run_dir / "results.csv").exists():
            print(f"[skip] assembly={assembly_col} output_dir={run_dir}", flush=True)
            continue

        run_dir.mkdir(parents=True, exist_ok=True)

        print(f"[train] assembly={assembly_col} output_dir={run_dir}", flush=True)
        print(
            f"[train] shuffle_n={int(cfg.shuffle_n)} bootstrap_n={int(cfg.bootstrap_n)} "
            f"bootstrap_ci={float(cfg.bootstrap_ci):.3f}",
            flush=True,
        )
        results_df, diagnostics_df, meta = run_interval_decoding(
            t0_ens,
            cfg=cfg,
            progress=True,
            log_every=25,
        )

        meta["assembly"] = assembly_col
        meta["session_date_filter"] = min_session_date
        meta["sessions_requested"] = int(len(session_names))
        meta["sessions_after_date_filter"] = int(len(filtered_session_names))
        meta["sessions_dropped_by_date_filter"] = int(len(dropped_session_names))

        save_info = _save_outputs(run_dir, results_df, diagnostics_df, meta, asdict(cfg))

        print(f"[done] assembly={assembly_col}", flush=True)
        print(f"  rows(results)={len(results_df)} rows(diagnostics)={len(diagnostics_df)}", flush=True)
        print(f"  results_csv={save_info['results_csv']}", flush=True)
        print(f"  diagnostics_csv={save_info['diagnostics_csv']}", flush=True)
        print(f"  parquet={save_info['parquet_status']}", flush=True)
        completed += 1

    print(f"[summary] completed={completed} requested={len(assembly_cols)}", flush=True)


if __name__ == "__main__":
    main()
