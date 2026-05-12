"""Parallel helpers for the choice-encoding negative-binomial GLM notebook.

The original notebook intentionally keeps the statistical model simple and
transparent, but the screen is embarrassingly parallel across bins and units.
This module keeps the model equations equivalent while making the expensive
screen stage usable on large CPU servers.
"""

from __future__ import annotations

import math
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from statsmodels.tools.sm_exceptions import PerfectSeparationError


FIT_ERRORS = (ValueError, np.linalg.LinAlgError, PerfectSeparationError)
_SCREEN_STATE: dict[str, Any] = {}
_THREADPOOL_LIMITER: Any | None = None


def configure_single_thread_blas(limits: int = 1) -> None:
    """Limit BLAS/OpenMP threads inside the current process.

    The GLM screen parallelizes across processes. Letting each process also use
    many BLAS threads causes oversubscription on large servers, so this function
    forcefully caps both environment variables and already-loaded thread pools.
    """

    limits = max(1, int(limits))
    for var in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[var] = str(limits)
    try:
        from threadpoolctl import threadpool_limits

        global _THREADPOOL_LIMITER
        _THREADPOOL_LIMITER = threadpool_limits(limits=limits)
    except Exception:
        pass


def estimate_alpha(y: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    mu = float(np.mean(y))
    var = float(np.var(y, ddof=1)) if y.size > 1 else 0.0
    return float(max((var - mu) / (mu**2), 1e-8)) if mu > 0 and var > mu else 1e-8


def nb_deviance(y: np.ndarray, mu: np.ndarray, alpha: float) -> float:
    family = sm.families.NegativeBinomial(alpha=float(max(alpha, 1e-8)))
    return float(family.deviance(np.asarray(y, dtype=float), np.clip(np.asarray(mu, dtype=float), 1e-9, None)))


def active_columns(df: pd.DataFrame, columns: list[str]) -> list[str]:
    active = []
    for col in columns:
        if col not in df.columns:
            continue
        values = pd.to_numeric(df[col], errors="coerce")
        if values.notna().sum() >= 2 and values.nunique(dropna=True) > 1:
            active.append(col)
    return active


def make_design(
    train_df: pd.DataFrame,
    ref_df: pd.DataFrame,
    predictors: list[str],
    continuous_cols: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_x = pd.DataFrame(index=train_df.index)
    ref_x = pd.DataFrame(index=ref_df.index)
    for col in predictors:
        train_col = pd.to_numeric(train_df[col], errors="coerce")
        ref_col = pd.to_numeric(ref_df[col], errors="coerce")
        fill = float(train_col.mean())
        train_col = train_col.fillna(fill)
        ref_col = ref_col.fillna(fill)
        if col in continuous_cols:
            scale = float(train_col.std(ddof=0))
            scale = scale if np.isfinite(scale) and scale > 0 else 1.0
            train_col = (train_col - fill) / scale
            ref_col = (ref_col - fill) / scale
        train_x[col] = train_col
        ref_x[col] = ref_col
    if train_x.shape[1] == 0:
        train_x = pd.DataFrame({"const": np.ones(len(train_df))}, index=train_df.index)
        ref_x = pd.DataFrame({"const": np.ones(len(ref_df))}, index=ref_df.index)
    else:
        train_x = sm.add_constant(train_x, has_constant="add")
        ref_x = sm.add_constant(ref_x, has_constant="add")
    return train_x, ref_x


def fit_nb(
    train_df: pd.DataFrame,
    predictors: list[str],
    ref_df: pd.DataFrame,
    settings: dict[str, Any],
) -> tuple[Any, pd.DataFrame, float, list[str]]:
    predictors = active_columns(train_df, predictors)
    train_x, ref_x = make_design(train_df, ref_df, predictors, settings["continuous_cols"])
    y = train_df["spike_count"].to_numpy(dtype=float)
    alpha = estimate_alpha(y)
    offset = np.log(np.clip(train_df["exposure"].to_numpy(dtype=float), 1e-9, None))
    result = sm.GLM(y, train_x, family=sm.families.NegativeBinomial(alpha=alpha), offset=offset).fit(
        maxiter=200,
        disp=0,
    )
    return result, ref_x, alpha, predictors


def trial_cv_splits(df: pd.DataFrame, settings: dict[str, Any]) -> list[tuple[np.ndarray, np.ndarray]]:
    trials = pd.Series(df["trial_id"].dropna().unique())
    if len(trials) < settings["min_trials"]:
        return []
    trials = trials.sample(frac=1, random_state=settings["random_state"]).to_numpy()
    n_splits = min(settings["n_cv_folds"], len(trials))
    folds = np.array_split(trials, n_splits)
    splits = []
    for test_trials in folds:
        test_mask = df["trial_id"].isin(test_trials).to_numpy()
        train_mask = ~test_mask
        if train_mask.sum() >= settings["min_rows"] and test_mask.sum() > 0:
            splits.append((train_mask, test_mask))
    return splits


def lr_pvalue(full_res: Any, reduced_res: Any) -> float:
    lr_stat = max(float(2.0 * (full_res.llf - reduced_res.llf)), 0.0)
    df_diff = float(full_res.df_model - reduced_res.df_model)
    return float(stats.chi2.sf(lr_stat, df_diff)) if df_diff > 0 else np.nan


def fit_feature_groups(
    df: pd.DataFrame,
    feature_groups: dict[str, list[str]],
    baseline_cols: list[str],
    settings: dict[str, Any],
) -> list[dict[str, Any]]:
    df = df.dropna(subset=["trial_id", "spike_count", "exposure"]).copy()
    df = df[df["exposure"].gt(0) & df["spike_count"].ge(0)]
    n_trials = int(df["trial_id"].nunique())
    mean_count = float(df["spike_count"].mean())
    if (
        len(df) < settings["min_rows"]
        or n_trials < settings["min_trials"]
        or mean_count < settings["min_mean_count"]
        or df["spike_count"].var() <= 0
    ):
        return []

    baseline = active_columns(df, baseline_cols)
    groups = {name: active_columns(df, cols) for name, cols in feature_groups.items()}
    groups = {name: cols for name, cols in groups.items() if cols}
    full_cols = baseline + sorted({c for cols in groups.values() for c in cols})
    if not groups:
        return []

    splits = trial_cv_splits(df, settings)
    if not splits:
        return []

    dev_full = 0.0
    dev_null = 0.0
    dev_reduced = {name: 0.0 for name in groups}
    valid_reduced = {name: 0 for name in groups}
    fold_alphas = []

    for train_mask, test_mask in splits:
        train_df = df.loc[train_mask]
        test_df = df.loc[test_mask]
        try:
            null_res, null_x, null_alpha, _ = fit_nb(train_df, baseline, test_df, settings)
            full_res, full_x, full_alpha, used_full = fit_nb(train_df, full_cols, test_df, settings)
        except FIT_ERRORS:
            return []

        test_offset = np.log(np.clip(test_df["exposure"].to_numpy(dtype=float), 1e-9, None))
        y_test = test_df["spike_count"].to_numpy(dtype=float)
        mu_null = np.clip(null_res.predict(null_x, offset=test_offset), 1e-9, None)
        mu_full = np.clip(full_res.predict(full_x, offset=test_offset), 1e-9, None)
        dev_null += nb_deviance(y_test, mu_null, null_alpha)
        dev_full += nb_deviance(y_test, mu_full, full_alpha)
        fold_alphas.append(full_alpha)

        for name, cols in groups.items():
            held_out_cols = [c for c in cols if c in used_full]
            if not held_out_cols:
                continue
            reduced_cols = [c for c in used_full if c not in held_out_cols]
            try:
                red_res, red_x, red_alpha, _ = fit_nb(train_df, reduced_cols, test_df, settings)
            except FIT_ERRORS:
                continue
            mu_red = np.clip(red_res.predict(red_x, offset=test_offset), 1e-9, None)
            dev_reduced[name] += nb_deviance(y_test, mu_red, red_alpha)
            valid_reduced[name] += 1

    if not np.isfinite(dev_null) or dev_null <= settings["min_null_deviance"] or not np.isfinite(dev_full):
        return []

    full_fde = 1.0 - dev_full / dev_null
    if not np.isfinite(full_fde) or abs(full_fde) > settings["max_abs_fde"]:
        return []
    try:
        full_res, _, alpha_full, used_full = fit_nb(df, full_cols, df, settings)
    except FIT_ERRORS:
        return []

    records = []
    for name, cols in groups.items():
        used_group = [c for c in cols if c in used_full]
        if valid_reduced[name] != len(splits) or not used_group or not np.isfinite(dev_reduced[name]):
            continue
        reduced_fde = 1.0 - dev_reduced[name] / dev_null
        delta_fde = full_fde - reduced_fde
        if (
            not np.isfinite(reduced_fde)
            or not np.isfinite(delta_fde)
            or abs(reduced_fde) > settings["max_abs_fde"]
            or abs(delta_fde) > settings["max_abs_fde"]
        ):
            continue
        reduced_cols = [c for c in used_full if c not in used_group]
        try:
            reduced_res, _, _, _ = fit_nb(df, reduced_cols, df, settings)
            p_value = lr_pvalue(full_res, reduced_res)
        except FIT_ERRORS:
            p_value = np.nan
        records.append(
            {
                "feature_group": name,
                "family": "negbin",
                "n_rows": int(len(df)),
                "n_trials": n_trials,
                "mean_count": float(df["spike_count"].mean()),
                "var_count": float(df["spike_count"].var(ddof=1)),
                "alpha_nb": float(alpha_full),
                "full_fde_cv": float(full_fde),
                "reduced_fde_cv": float(reduced_fde),
                "delta_fde": float(delta_fde),
                "p_value": p_value,
                "n_predictors_full": int(len(used_full)),
                "n_predictors_group": int(len(used_group)),
            }
        )
    return records


def add_panel_fdr(df: pd.DataFrame, panel_cols: list[str], fdr_alpha: float) -> pd.DataFrame:
    if df.empty:
        df["q_value"] = np.nan
        df["is_significant"] = False
        return df
    out = df.copy()
    out["q_value"] = np.nan
    for _, idx in out.groupby(panel_cols, dropna=False).groups.items():
        mask = out.index.isin(idx) & out["p_value"].notna()
        if mask.any():
            out.loc[mask, "q_value"] = multipletests(out.loc[mask, "p_value"], method="fdr_bh")[1]
    out["is_significant"] = out["q_value"].lt(fdr_alpha) & out["delta_fde"].gt(0)
    return out


def _init_worker_state(state: dict[str, Any]) -> None:
    configure_single_thread_blas()
    global _SCREEN_STATE
    _SCREEN_STATE = state


def _key_record(group_keys: list[str], key: Any) -> dict[str, Any]:
    key_tuple = key if isinstance(key, tuple) else (key,)
    return dict(zip(group_keys, key_tuple))


def _screen_chunk(bounds: tuple[int, int]) -> tuple[int, int, list[dict[str, Any]]]:
    configure_single_thread_blas()
    start, stop = bounds
    state = _SCREEN_STATE
    base_df = state["base_df"]
    unit_columns = state["unit_columns"]
    group_keys = state["group_keys"]
    feature_groups = state["feature_groups"]
    baseline_cols = state["baseline_cols"]
    predictors = state["predictors"]
    panel = state["panel"]
    region_lookup = state["region_lookup"]
    settings = state["settings"]
    max_fits = state["max_fits"]
    keep_cols = state["keep_cols"]
    groups = state["groups"]

    rows: list[dict[str, Any]] = []
    processed = 0
    n_units = len(unit_columns)

    for group_i in range(start, stop):
        key, idx = groups[group_i]
        sub = base_df.iloc[idx]
        key_record = _key_record(group_keys, key)
        common = sub[keep_cols].copy()
        valid_common = common["trial_id"].notna() & common["exposure"].gt(0)
        session_id = key_record.get("session_id")

        for unit_i, unit in enumerate(unit_columns):
            global_fit_i = group_i * n_units + unit_i + 1
            if max_fits is not None and global_fit_i > max_fits:
                return start, processed, rows
            processed += 1

            spike_count = pd.to_numeric(sub[unit], errors="coerce")
            valid = valid_common & spike_count.notna() & spike_count.ge(0)
            n_rows = int(valid.sum())
            if n_rows < settings["min_rows"]:
                continue
            y = spike_count.loc[valid].to_numpy(dtype=float)
            if y.size == 0 or float(np.mean(y)) < settings["min_mean_count"] or float(np.var(y, ddof=1)) <= 0:
                continue

            model_df = common.loc[valid].copy()
            if int(model_df["trial_id"].nunique()) < settings["min_trials"]:
                continue
            model_df["unit"] = unit
            model_df["brain_region"] = [region_lookup.get((s, unit), "Unknown") for s in model_df["session_id"]]
            model_df["spike_count"] = y

            for rec in fit_feature_groups(model_df, feature_groups, baseline_cols, settings):
                rows.append(
                    {
                        "panel": panel,
                        **key_record,
                        "unit": unit,
                        "brain_region": region_lookup.get((session_id, unit), "Unknown"),
                        **rec,
                    }
                )

    return start, processed, rows


def run_screen_parallel(
    base_df: pd.DataFrame,
    unit_columns: list[str],
    group_keys: list[str],
    feature_groups: dict[str, list[str]],
    panel: str,
    settings: dict[str, Any],
    region_lookup: dict[tuple[Any, str], str],
    max_fits: int | None = None,
    baseline_cols: list[str] | None = None,
    n_workers: int | None = None,
    groups_per_task: int = 1,
    start_method: str = "fork",
    progress_every: int = 500,
) -> pd.DataFrame:
    """Run a feature-group screen in parallel across group-key chunks."""

    configure_single_thread_blas()
    baseline_cols = baseline_cols or ["trial_number_z"]
    predictors = sorted({c for cols in feature_groups.values() for c in cols} | set(baseline_cols))
    base_df = base_df.reset_index(drop=True)
    groups = list(base_df.groupby(group_keys, sort=False).indices.items())
    total = len(groups) * len(unit_columns)
    total_to_run = min(total, int(max_fits)) if max_fits is not None else total
    n_groups_to_run = math.ceil(total_to_run / max(len(unit_columns), 1)) if total_to_run else 0
    groups_per_task = max(1, int(groups_per_task))

    result_columns = [
        "panel",
        *group_keys,
        "unit",
        "brain_region",
        "feature_group",
        "family",
        "n_rows",
        "n_trials",
        "mean_count",
        "var_count",
        "alpha_nb",
        "full_fde_cv",
        "reduced_fde_cv",
        "delta_fde",
        "p_value",
        "n_predictors_full",
        "n_predictors_group",
    ]
    if total_to_run == 0:
        return pd.DataFrame(columns=result_columns)

    keep_cols = list(
        dict.fromkeys(
            [
                "session_id",
                "trial_id",
                "exposure",
                "trial_number",
                "trial_number_z",
                *group_keys,
                *predictors,
            ]
        )
    )
    keep_cols = [c for c in keep_cols if c in base_df.columns]

    cpu_count = os.cpu_count() or 1
    n_workers = int(n_workers or max(1, min(cpu_count - 1, 32)))
    n_workers = max(1, min(n_workers, n_groups_to_run))
    chunk_bounds = [
        (start, min(start + groups_per_task, n_groups_to_run))
        for start in range(0, n_groups_to_run, groups_per_task)
    ]

    state = {
        "base_df": base_df,
        "unit_columns": list(unit_columns),
        "group_keys": list(group_keys),
        "feature_groups": {k: list(v) for k, v in feature_groups.items()},
        "baseline_cols": list(baseline_cols),
        "predictors": predictors,
        "panel": panel,
        "region_lookup": region_lookup,
        "settings": {**settings, "continuous_cols": set(settings["continuous_cols"])},
        "max_fits": None if max_fits is None else int(max_fits),
        "keep_cols": keep_cols,
        "groups": groups,
    }

    global _SCREEN_STATE
    _SCREEN_STATE = state
    ctx = mp.get_context(start_method)
    initializer = None if start_method == "fork" else _init_worker_state
    initargs = () if start_method == "fork" else (state,)

    print(
        f"[{panel}] parallel screen: {total_to_run:,}/{total:,} outer fits, "
        f"{len(chunk_bounds):,} tasks, {n_workers:,} workers, BLAS threads=1"
    )

    started = time.time()
    processed = 0
    row_count = 0
    next_progress = min(progress_every, total_to_run)
    chunk_results: list[tuple[int, list[dict[str, Any]]]] = []

    with ProcessPoolExecutor(
        max_workers=n_workers,
        mp_context=ctx,
        initializer=initializer,
        initargs=initargs,
    ) as pool:
        futures = [pool.submit(_screen_chunk, bounds) for bounds in chunk_bounds]
        for fut in as_completed(futures):
            start, n_processed, rows = fut.result()
            processed += n_processed
            row_count += len(rows)
            chunk_results.append((start, rows))
            if processed >= next_progress or processed >= total_to_run:
                elapsed = time.time() - started
                rate = processed / max(elapsed, 1e-9)
                print(
                    f"[{panel}] completed {processed:,}/{total_to_run:,} outer fits "
                    f"| result rows={row_count:,} | {rate:,.1f} fits/s"
                )
                while next_progress <= processed and next_progress < total_to_run:
                    next_progress += progress_every

    ordered_rows: list[dict[str, Any]] = []
    for _, rows in sorted(chunk_results, key=lambda item: item[0]):
        ordered_rows.extend(rows)
    return pd.DataFrame(ordered_rows, columns=result_columns)
