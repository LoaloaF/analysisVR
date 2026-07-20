"""Compact interval decoding pipeline based on grouped CV logistic regression."""

from __future__ import annotations

import hashlib
import json
import math
import time
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold


DEFAULT_TARGETS: Dict[str, str] = {
    "Outcome": "outcome_binary",
    "Cue": "cue",
    "R1 choice": "choice_R1",
    "R2 choice": "choice_R2",
}

DEFAULT_SCORE_METRIC = "balanced_accuracy"


@dataclass
class DecodeConfig:
    assembly_col: str = "Assembly007"
    window_half_width: int = 1
    min_samples_per_time: int = 20
    min_class_count: int = 4
    max_cv_splits: int = 5
    random_state: int = 42
    shuffle_n: int = 0
    bootstrap_n: int = 0
    bootstrap_ci: float = 0.95
    use_group_cv: bool = True
    min_session_date: Optional[str] = "2024-11-27"
    drop_unparseable_session_dates: bool = True
    excluded_intervals: Optional[Sequence[str]] = None
    shuffle_within_group: bool = True
    bootstrap_within_group: bool = True


def _parse_session_date_token(session_name: str) -> Optional[datetime]:
    text = str(session_name)
    for token in text.split("_"):
        try:
            return datetime.strptime(token, "%Y-%m-%d")
        except ValueError:
            continue
    return None


def _ensure_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if isinstance(out.index, pd.MultiIndex) or out.index.name is not None:
        out = out.reset_index()
    return out


def _find_trial_col(df: pd.DataFrame) -> Optional[str]:
    for name in ["trial_id", "behavior_trial_id", "trial", "trial_index", "triald_id"]:
        if name in df.columns:
            return name
    return None


def _infer_bin_width_s(df: pd.DataFrame) -> float:
    if {"from_ephys_timestamp", "to_ephys_timestamp"}.issubset(df.columns):
        width = pd.to_numeric(df["to_ephys_timestamp"], errors="coerce") - pd.to_numeric(
            df["from_ephys_timestamp"], errors="coerce"
        )
        width = width.dropna()
        if not width.empty:
            return float(width.median()) / 1e6
    return 0.04


def prepare_interval_table(
    ens_df: pd.DataFrame,
    cfg: DecodeConfig,
) -> Tuple[pd.DataFrame, str, str, str, List[str], Dict[str, float], float]:
    """Prepare one row per raw interval bin."""

    df = _ensure_dataframe(ens_df)
    session_col = "session_id"
    interval_col = "interval_name"

    required_cols = {session_col, interval_col, cfg.assembly_col, "cue", "choice_R1", "choice_R2"}
    missing = sorted(required_cols - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    trial_col = _find_trial_col(df)
    if trial_col is None:
        trial_col = "trial_id"
        df[trial_col] = (
            df.groupby([session_col, interval_col], sort=False, dropna=False)
            .cumcount()
            .astype(int)
        )

    if cfg.min_session_date:
        cutoff = datetime.strptime(str(cfg.min_session_date), "%Y-%m-%d")
        parsed = df[session_col].map(_parse_session_date_token)
        keep = parsed.map(lambda value: (value is not None) and (value >= cutoff))
        if not cfg.drop_unparseable_session_dates:
            keep = keep | parsed.isna()
        df = df[keep].copy()

    if cfg.excluded_intervals:
        excluded = {str(value) for value in cfg.excluded_intervals}
        df = df[~df[interval_col].astype(str).isin(excluded)].copy()

    feature_cols = [cfg.assembly_col]
    numeric_cols = feature_cols + ["cue", "choice_R1", "choice_R2"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["outcome_binary"] = (
        ((df["cue"] == 1) & (df["choice_R1"] == 1))
        | ((df["cue"] == 2) & (df["choice_R2"].isin([1, 2])))
    ).astype(float)

    sort_cols = [session_col, trial_col, interval_col]
    for col in ["from_ephys_timestamp", "to_ephys_timestamp", "entry_id"]:
        if col in df.columns:
            sort_cols.append(col)
    df = df.sort_values(sort_cols).copy()

    group_cols = [session_col, trial_col, interval_col]
    df["time_bin_idx"] = df.groupby(group_cols, sort=False, dropna=False).cumcount().astype(int)
    df["n_bins_interval"] = (
        df.groupby(group_cols, sort=False, dropna=False)["time_bin_idx"]
        .transform("max")
        .add(1)
        .astype(int)
    )
    df["time_rel"] = np.where(
        df["n_bins_interval"] > 1,
        df["time_bin_idx"] / (df["n_bins_interval"] - 1),
        0.0,
    )
    df["time_rel_bin"] = df["time_bin_idx"].astype(int)

    interval_len_bins = df.groupby(interval_col, dropna=False)["n_bins_interval"].median().to_dict()
    return df, session_col, trial_col, interval_col, feature_cols, interval_len_bins, _infer_bin_width_s(df)


def _target_filter(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    out = df.copy()
    allowed = [1, 2] if target_col == "cue" else [0, 1]
    out = out[out[target_col].isin(allowed)]
    out[target_col] = out[target_col].astype(int)
    return out


def _build_trial_bin_df(
    df: pd.DataFrame,
    session_col: str,
    trial_col: str,
    interval_col: str,
    feature_cols: Sequence[str],
    target_col: str,
) -> pd.DataFrame:
    cols = [
        session_col,
        trial_col,
        interval_col,
        "time_bin_idx",
        "time_rel",
        "time_rel_bin",
        *feature_cols,
        target_col,
    ]
    out = df[cols].copy()
    out = out.dropna(subset=[session_col, trial_col, interval_col, "time_bin_idx", target_col, *feature_cols])
    out[target_col] = pd.to_numeric(out[target_col], errors="coerce")
    out = out.dropna(subset=[target_col])
    out = _target_filter(out, target_col)

    agg_map = {col: "mean" for col in feature_cols}
    agg_map[target_col] = "first"
    agg_map["time_rel"] = "mean"
    agg_map["time_rel_bin"] = "mean"
    return out.groupby([session_col, trial_col, interval_col, "time_bin_idx"], as_index=False).agg(agg_map)


def _window_pool(
    interval_df: pd.DataFrame,
    center_bin: int,
    window_half_width: int,
    session_col: str,
    trial_col: str,
    feature_cols: Sequence[str],
    target_col: str,
) -> pd.DataFrame:
    lo = int(center_bin) - int(window_half_width)
    hi = int(center_bin) + int(window_half_width)
    win = interval_df[interval_df["time_bin_idx"].between(lo, hi)].copy()
    if win.empty:
        return win

    agg_map = {col: "mean" for col in feature_cols}
    agg_map[target_col] = "first"
    agg_map["time_rel"] = "mean"
    agg_map["time_rel_bin"] = "mean"
    agg_map["time_bin_idx"] = "mean"
    return win.groupby([session_col, trial_col], as_index=False).agg(agg_map)


def _label_to_binary(y: np.ndarray) -> Tuple[np.ndarray, Dict[int, int]]:
    values = np.sort(np.unique(y))
    if values.size != 2:
        raise ValueError(f"Expected binary labels, got {values.tolist()}")
    mapping = {int(values[0]): 0, int(values[1]): 1}
    y01 = np.array([mapping[int(value)] for value in y], dtype=np.int64)
    return y01, mapping


def _build_cv_splits(
    y: np.ndarray,
    groups: np.ndarray,
    cfg: DecodeConfig,
) -> Tuple[Optional[List[Tuple[np.ndarray, np.ndarray]]], str, int]:
    class_counts = pd.Series(y).value_counts(dropna=True)
    if len(class_counts) < 2:
        return None, "single_class", 0
    if int(class_counts.min()) < int(cfg.min_class_count):
        return None, "too_few_per_class", 0

    if cfg.use_group_cv:
        group_labels = pd.Series(groups).astype(str)
        n_groups = int(group_labels.nunique())
        n_splits = int(min(int(cfg.max_cv_splits), n_groups))
        if n_splits >= 2:
            try:
                cv = StratifiedGroupKFold(
                    n_splits=n_splits,
                    shuffle=True,
                    random_state=int(cfg.random_state),
                )
                splits = list(cv.split(np.zeros(len(y)), y, groups=group_labels))
                if splits:
                    return splits, "stratified_group", n_splits
            except ValueError:
                pass

    n_splits = int(min(int(cfg.max_cv_splits), int(class_counts.min())))
    if n_splits < 2:
        return None, "cv_not_possible", 0

    try:
        cv = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=int(cfg.random_state),
        )
        splits = list(cv.split(np.zeros(len(y)), y))
    except ValueError:
        return None, "cv_error", 0
    return (splits, "stratified", n_splits) if splits else (None, "cv_not_possible", 0)


def _standardize_train_test(
    x_train: np.ndarray,
    x_test: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        median = np.nanmedian(x_train, axis=0)
    median = np.nan_to_num(median, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)

    xtr = np.asarray(x_train, dtype=np.float32).copy()
    nan_i, nan_j = np.where(np.isnan(xtr))
    if nan_i.size:
        xtr[nan_i, nan_j] = median[nan_j]

    mean = xtr.mean(axis=0).astype(np.float32, copy=False)
    std = xtr.std(axis=0).astype(np.float32, copy=False)
    std = np.where(std > 1e-6, std, 1.0).astype(np.float32, copy=False)
    xtr = ((xtr - mean) / std).astype(np.float32, copy=False)

    if x_test is None:
        return xtr, None

    xte = np.asarray(x_test, dtype=np.float32).copy()
    nan_i, nan_j = np.where(np.isnan(xte))
    if nan_i.size:
        xte[nan_i, nan_j] = median[nan_j]
    xte = ((xte - mean) / std).astype(np.float32, copy=False)
    return xtr, xte


def _fit_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    random_state: int,
) -> np.ndarray:
    xtr, xte = _standardize_train_test(x_train, x_test)
    if xte is None:
        raise RuntimeError("Expected standardized test matrix.")

    model = LogisticRegression(
        solver="liblinear",
        penalty="l2",
        class_weight="balanced",
        max_iter=2000,
        random_state=int(random_state),
    )
    model.fit(xtr, y_train)
    return model.predict(xte).astype(np.int64, copy=False)


def _stable_seed(*parts: object) -> int:
    payload = "||".join(str(part) for part in parts).encode("utf-8")
    return int(hashlib.md5(payload).hexdigest()[:8], 16)


def _metric_score(metric: str, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if metric == "accuracy":
        return float(accuracy_score(y_true, y_pred))
    return float(balanced_accuracy_score(y_true, y_pred))


def _one_sided_above_chance_pvalue(n_correct: int, n_total: int, p0: float = 0.5) -> float:
    if n_total <= 0:
        return float("nan")
    mean = n_total * p0
    variance = n_total * p0 * (1.0 - p0)
    z = (float(n_correct) - mean - 0.5) / math.sqrt(variance)
    return float(0.5 * math.erfc(z / math.sqrt(2.0)))


def _permute_labels(
    y: np.ndarray,
    groups: np.ndarray,
    rng: np.random.Generator,
    within_group: bool,
) -> np.ndarray:
    y_perm = np.array(y, copy=True)
    if not within_group:
        return rng.permutation(y_perm)

    group_labels = pd.Series(groups).astype(str).to_numpy()
    for group in np.unique(group_labels):
        idx = np.where(group_labels == group)[0]
        if idx.size > 1:
            y_perm[idx] = rng.permutation(y_perm[idx])
    return y_perm


def _empty_shuffle_stats(cfg: DecodeConfig) -> Dict[str, object]:
    return {
        "shuffle_n": int(max(0, cfg.shuffle_n)),
        "shuffle_metric": DEFAULT_SCORE_METRIC,
        "shuffle_n_effective": 0,
        "shuffle_fail_count": 0,
        "shuffle_null_mean": np.nan,
        "shuffle_null_std": np.nan,
        "shuffle_effect": np.nan,
        "p_shuffle": np.nan,
        "significant_shuffle_0_05": np.nan,
    }


def _shuffle_significance_empirical(
    x_all: np.ndarray,
    y_true: np.ndarray,
    groups: np.ndarray,
    splits: Sequence[Tuple[np.ndarray, np.ndarray]],
    cfg: DecodeConfig,
    observed_metric: float,
    random_state: int,
) -> Dict[str, object]:
    stats = _empty_shuffle_stats(cfg)
    n_shuffles = int(stats["shuffle_n"])
    if n_shuffles <= 0 or not np.isfinite(observed_metric):
        stats["shuffle_n"] = 0
        return stats

    rng = np.random.default_rng(int(random_state))
    null_scores: List[float] = []
    fail_count = 0

    for shuffle_idx in range(n_shuffles):
        y_perm = _permute_labels(y_true, groups, rng, within_group=bool(cfg.shuffle_within_group))
        fold_scores: List[float] = []
        failed = False

        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            y_tr = y_perm[train_idx]
            y_te = y_perm[test_idx]
            if np.unique(y_tr).size != 2 or np.unique(y_te).size != 2:
                failed = True
                break
            try:
                y_hat = _fit_predict(
                    x_train=x_all[train_idx],
                    y_train=y_tr,
                    x_test=x_all[test_idx],
                    random_state=int(random_state + shuffle_idx * 1000 + fold_idx),
                )
            except Exception:
                failed = True
                break
            fold_scores.append(_metric_score(DEFAULT_SCORE_METRIC, y_te, y_hat))

        if failed or not fold_scores:
            fail_count += 1
            continue
        null_scores.append(float(np.mean(fold_scores)))

    stats["shuffle_fail_count"] = int(fail_count)
    if not null_scores:
        return stats

    null_arr = np.asarray(null_scores, dtype=float)
    null_mean = float(np.mean(null_arr))
    p_shuffle = float((1.0 + np.sum(null_arr >= observed_metric)) / (1.0 + null_arr.size))
    stats.update(
        {
            "shuffle_n_effective": int(null_arr.size),
            "shuffle_null_mean": null_mean,
            "shuffle_null_std": float(np.std(null_arr, ddof=1)) if null_arr.size > 1 else 0.0,
            "shuffle_effect": float(observed_metric - null_mean),
            "p_shuffle": p_shuffle,
            "significant_shuffle_0_05": bool(p_shuffle < 0.05),
        }
    )
    return stats


def _empty_bootstrap_stats(cfg: DecodeConfig) -> Dict[str, object]:
    return {
        "bootstrap_n": int(max(0, cfg.bootstrap_n)),
        "bootstrap_metric": DEFAULT_SCORE_METRIC,
        "bootstrap_ci": float(cfg.bootstrap_ci),
        "bootstrap_n_effective": 0,
        "bootstrap_fail_count": 0,
        "bootstrap_mean": np.nan,
        "bootstrap_std": np.nan,
        "bootstrap_ci_low": np.nan,
        "bootstrap_ci_high": np.nan,
        "bootstrap_effect": np.nan,
        "p_bootstrap_above_chance": np.nan,
        "significant_bootstrap_0_05": np.nan,
    }


def _bootstrap_metric_from_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
    cfg: DecodeConfig,
    observed_metric: float,
    chance_level: float,
    random_state: int,
) -> Dict[str, object]:
    stats = _empty_bootstrap_stats(cfg)
    n_boot = int(stats["bootstrap_n"])
    if n_boot <= 0 or not np.isfinite(observed_metric):
        stats["bootstrap_n"] = 0
        return stats

    ci = float(cfg.bootstrap_ci)
    if not (0.0 < ci < 1.0):
        raise ValueError("bootstrap_ci must be in (0, 1).")

    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    group_labels = pd.Series(groups).astype(str).to_numpy()
    if y_true.size == 0 or y_true.size != y_pred.size or y_true.size != group_labels.size:
        stats["bootstrap_fail_count"] = int(n_boot)
        return stats

    rng = np.random.default_rng(int(random_state))
    scores: List[float] = []
    fail_count = 0

    if cfg.bootstrap_within_group:
        unique_groups = np.unique(group_labels)
        idx_by_group = {group: np.where(group_labels == group)[0] for group in unique_groups}
    else:
        unique_groups = np.array([], dtype=str)
        idx_by_group = {}

    for _ in range(n_boot):
        if cfg.bootstrap_within_group and unique_groups.size > 0:
            chosen = rng.choice(unique_groups, size=unique_groups.size, replace=True)
            sample_idx = np.concatenate([idx_by_group[str(group)] for group in chosen]).astype(int, copy=False)
        else:
            sample_idx = rng.integers(low=0, high=y_true.size, size=y_true.size, endpoint=False)

        yt = y_true[sample_idx]
        yp = y_pred[sample_idx]
        if np.unique(yt).size != 2:
            fail_count += 1
            continue
        scores.append(_metric_score(DEFAULT_SCORE_METRIC, yt, yp))

    stats["bootstrap_fail_count"] = int(fail_count)
    if not scores:
        return stats

    score_arr = np.asarray(scores, dtype=float)
    low_q = (1.0 - ci) / 2.0
    high_q = 1.0 - low_q
    mean_score = float(np.mean(score_arr))
    p_boot = float((1.0 + np.sum(score_arr <= chance_level)) / (1.0 + score_arr.size))
    stats.update(
        {
            "bootstrap_n_effective": int(score_arr.size),
            "bootstrap_mean": mean_score,
            "bootstrap_std": float(np.std(score_arr, ddof=1)) if score_arr.size > 1 else 0.0,
            "bootstrap_ci_low": float(np.quantile(score_arr, low_q)),
            "bootstrap_ci_high": float(np.quantile(score_arr, high_q)),
            "bootstrap_effect": float(observed_metric - mean_score),
            "p_bootstrap_above_chance": p_boot,
            "significant_bootstrap_0_05": bool(p_boot < 0.05),
        }
    )
    return stats


def run_interval_decoding(
    ens_df: pd.DataFrame,
    cfg: Optional[DecodeConfig] = None,
    targets: Optional[Dict[str, str]] = None,
    progress: bool = False,
    log_every: int = 25,
    target_names: Optional[Sequence[str]] = None,
    interval_names: Optional[Sequence[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    """Run grouped-CV interval decoding and compute shuffle/bootstrap statistics."""

    cfg = cfg or DecodeConfig()
    target_map = targets or DEFAULT_TARGETS
    target_filter = {str(value) for value in target_names} if target_names else None
    interval_filter = {str(value) for value in interval_names} if interval_names else None

    (
        df,
        session_col,
        trial_col,
        interval_col,
        feature_cols,
        interval_len_bins,
        bin_width_s,
    ) = prepare_interval_table(ens_df, cfg)

    results: List[Dict[str, object]] = []
    diagnostics: List[Dict[str, object]] = []
    started = time.perf_counter()
    task_count = 0

    if progress:
        print(
            f"[start] feature={feature_cols[0]} targets={list(target_map.keys())} "
            f"shuffle_n={int(cfg.shuffle_n)} bootstrap_n={int(cfg.bootstrap_n)}",
            flush=True,
        )

    for target_name, target_col in target_map.items():
        if target_filter is not None and str(target_name) not in target_filter:
            continue

        trial_bin_df = _build_trial_bin_df(
            df=df,
            session_col=session_col,
            trial_col=trial_col,
            interval_col=interval_col,
            feature_cols=feature_cols,
            target_col=target_col,
        )
        if trial_bin_df.empty:
            continue

        for interval_name, interval_df in trial_bin_df.groupby(interval_col, sort=False):
            if interval_filter is not None and str(interval_name) not in interval_filter:
                continue

            for raw_bin in sorted(interval_df["time_bin_idx"].unique()):
                model_bin = int(round(float(raw_bin)))
                task_count += 1

                pooled = _window_pool(
                    interval_df=interval_df,
                    center_bin=model_bin,
                    window_half_width=cfg.window_half_width,
                    session_col=session_col,
                    trial_col=trial_col,
                    feature_cols=feature_cols,
                    target_col=target_col,
                )
                if pooled.empty:
                    continue

                n_samples = int(len(pooled))
                y_raw = pooled[target_col].to_numpy(dtype=int)
                class_counts = pd.Series(y_raw).value_counts(dropna=True)
                n_classes = int(len(class_counts))
                min_class = int(class_counts.min()) if n_classes else 0
                n_sessions = int(pooled[session_col].astype(str).nunique())
                time_rel = float(pooled["time_rel"].median())
                time_bin_idx = float(pooled["time_bin_idx"].median())
                time_rel_bin_est = int(round(float(pooled["time_rel_bin"].median())))

                diagnostics.append(
                    {
                        "target_name": target_name,
                        "target_col": target_col,
                        "interval_name": str(interval_name),
                        "model_bin": int(model_bin),
                        "time_rel_bin": int(time_rel_bin_est),
                        "time_bin_idx": float(time_bin_idx),
                        "time_rel": time_rel,
                        "n_samples": n_samples,
                        "n_sessions": n_sessions,
                        "n_classes": n_classes,
                        "min_class_count": min_class,
                    }
                )

                if progress and (task_count == 1 or task_count % max(1, int(log_every)) == 0):
                    elapsed = time.perf_counter() - started
                    print(
                        f"[decode] task={task_count} target={target_name} interval={interval_name} "
                        f"bin={model_bin} n={n_samples} elapsed_s={elapsed:.1f}",
                        flush=True,
                    )

                base_row: Dict[str, object] = {
                    "backend": "logreg",
                    "assembly": cfg.assembly_col,
                    "n_features": len(feature_cols),
                    "target_name": target_name,
                    "target_col": target_col,
                    "interval_name": str(interval_name),
                    "model_bin": int(model_bin),
                    "time_rel_bin": int(time_rel_bin_est),
                    "time_bin_idx": float(time_bin_idx),
                    "time_rel": time_rel,
                    "time_s": float(model_bin * bin_width_s),
                    "interval_n_bins_median": float(interval_len_bins.get(interval_name, np.nan)),
                    "window_half_width": int(cfg.window_half_width),
                    "n_samples": n_samples,
                    "n_sessions": n_sessions,
                    "n_classes": n_classes,
                    "min_class_count": min_class,
                    "chance_level": np.nan,
                    "n_eval_samples": np.nan,
                    "n_eval_correct": np.nan,
                    "acc_minus_chance": np.nan,
                    "p_above_chance": np.nan,
                    "significant_0_05": np.nan,
                    "label_mapping": None,
                    "fit_error": None,
                    **_empty_shuffle_stats(cfg),
                    **_empty_bootstrap_stats(cfg),
                }

                if n_samples < int(cfg.min_samples_per_time):
                    results.append(
                        {
                            **base_row,
                            "accuracy": np.nan,
                            "balanced_accuracy": np.nan,
                            "n_splits": 0,
                            "cv_strategy": "too_few_samples",
                            "fit_time_s": np.nan,
                        }
                    )
                    continue

                if n_classes != 2:
                    results.append(
                        {
                            **base_row,
                            "accuracy": np.nan,
                            "balanced_accuracy": np.nan,
                            "n_splits": 0,
                            "cv_strategy": "single_class",
                            "fit_time_s": np.nan,
                        }
                    )
                    continue

                x_all = pooled[list(feature_cols)].to_numpy(dtype=np.float32)
                y01, label_map = _label_to_binary(y_raw)
                groups = pooled[session_col].astype(str).to_numpy()
                splits, cv_strategy, n_splits = _build_cv_splits(y01, groups, cfg)

                if not splits:
                    results.append(
                        {
                            **base_row,
                            "accuracy": np.nan,
                            "balanced_accuracy": np.nan,
                            "n_splits": int(n_splits),
                            "cv_strategy": cv_strategy,
                            "fit_time_s": np.nan,
                            "label_mapping": label_map,
                        }
                    )
                    continue

                fold_acc: List[float] = []
                fold_bacc: List[float] = []
                fit_times: List[float] = []
                n_eval_total = 0
                n_eval_correct = 0
                eval_true_parts: List[np.ndarray] = []
                eval_pred_parts: List[np.ndarray] = []
                eval_group_parts: List[np.ndarray] = []
                fit_error: Optional[str] = None

                for fold_idx, (train_idx, test_idx) in enumerate(splits):
                    start = time.perf_counter()
                    try:
                        y_hat = _fit_predict(
                            x_train=x_all[train_idx],
                            y_train=y01[train_idx],
                            x_test=x_all[test_idx],
                            random_state=int(cfg.random_state + fold_idx),
                        )
                    except Exception as err:
                        fit_error = f"{type(err).__name__}: {err}"
                        break
                    fit_times.append(float(time.perf_counter() - start))

                    y_test = y01[test_idx]
                    fold_acc.append(float(accuracy_score(y_test, y_hat)))
                    fold_bacc.append(float(balanced_accuracy_score(y_test, y_hat)))
                    n_eval_total += int(len(y_test))
                    n_eval_correct += int((y_hat == y_test).sum())
                    eval_true_parts.append(np.asarray(y_test, dtype=np.int64))
                    eval_pred_parts.append(np.asarray(y_hat, dtype=np.int64))
                    eval_group_parts.append(np.asarray(groups[test_idx]))

                if fit_error is not None:
                    results.append(
                        {
                            **base_row,
                            "accuracy": np.nan,
                            "balanced_accuracy": np.nan,
                            "n_splits": int(n_splits),
                            "cv_strategy": "fit_error",
                            "fit_time_s": np.nan,
                            "label_mapping": label_map,
                            "fit_error": fit_error,
                        }
                    )
                    continue

                chance_level = 0.5
                observed_accuracy = float(n_eval_correct) / float(n_eval_total)
                observed_bacc = float(np.mean(fold_bacc))
                p_above = _one_sided_above_chance_pvalue(
                    n_correct=n_eval_correct,
                    n_total=n_eval_total,
                    p0=chance_level,
                )

                slice_seed = _stable_seed(target_name, interval_name, model_bin)
                shuffle_stats = _shuffle_significance_empirical(
                    x_all=x_all,
                    y_true=y01,
                    groups=groups,
                    splits=splits,
                    cfg=cfg,
                    observed_metric=observed_bacc,
                    random_state=int(cfg.random_state + slice_seed),
                )
                bootstrap_stats = _bootstrap_metric_from_predictions(
                    y_true=np.concatenate(eval_true_parts, axis=0),
                    y_pred=np.concatenate(eval_pred_parts, axis=0),
                    groups=np.concatenate(eval_group_parts, axis=0),
                    cfg=cfg,
                    observed_metric=observed_bacc,
                    chance_level=float(chance_level),
                    random_state=int(cfg.random_state + slice_seed + 17),
                )

                results.append(
                    {
                        **base_row,
                        "accuracy": float(np.mean(fold_acc)),
                        "balanced_accuracy": observed_bacc,
                        "n_splits": int(n_splits),
                        "cv_strategy": cv_strategy,
                        "fit_time_s": float(np.mean(fit_times)) if fit_times else np.nan,
                        "n_eval_samples": int(n_eval_total),
                        "n_eval_correct": int(n_eval_correct),
                        "chance_level": float(chance_level),
                        "acc_minus_chance": float(observed_accuracy - chance_level),
                        "p_above_chance": p_above,
                        "significant_0_05": bool(p_above < 0.05) if pd.notna(p_above) else np.nan,
                        "label_mapping": label_map,
                        **shuffle_stats,
                        **bootstrap_stats,
                    }
                )

    elapsed_s = float(time.perf_counter() - started)
    if progress:
        print(f"[done] tasks={task_count} rows={len(results)} elapsed_s={elapsed_s:.1f}", flush=True)

    results_df = pd.DataFrame(results)
    diagnostics_df = pd.DataFrame(diagnostics)
    meta: Dict[str, object] = {
        "feature_cols": list(feature_cols),
        "n_features": len(feature_cols),
        "session_col": session_col,
        "trial_col": trial_col,
        "interval_col": interval_col,
        "bin_width_s": float(bin_width_s),
        "elapsed_s": elapsed_s,
        "task_count": int(task_count),
        "target_filter": sorted(target_filter) if target_filter is not None else None,
        "interval_filter": sorted(interval_filter) if interval_filter is not None else None,
        "shuffle_n": int(max(0, cfg.shuffle_n)),
        "shuffle_metric": DEFAULT_SCORE_METRIC,
        "shuffle_within_group": bool(cfg.shuffle_within_group),
        "bootstrap_n": int(max(0, cfg.bootstrap_n)),
        "bootstrap_metric": DEFAULT_SCORE_METRIC,
        "bootstrap_ci": float(cfg.bootstrap_ci),
        "bootstrap_within_group": bool(cfg.bootstrap_within_group),
        "min_session_date": cfg.min_session_date,
        "drop_unparseable_session_dates": bool(cfg.drop_unparseable_session_dates),
    }
    return results_df, diagnostics_df, meta


def load_decode_config(config_path: Path | str) -> DecodeConfig:
    """Load DecodeConfig from a saved config JSON."""

    payload = json.loads(Path(config_path).read_text())
    valid_fields = set(DecodeConfig.__dataclass_fields__.keys())
    cfg_kwargs = {key: value for key, value in payload.items() if key in valid_fields}
    return DecodeConfig(**cfg_kwargs)


__all__ = [
    "DEFAULT_TARGETS",
    "DecodeConfig",
    "load_decode_config",
    "prepare_interval_table",
    "run_interval_decoding",
]
