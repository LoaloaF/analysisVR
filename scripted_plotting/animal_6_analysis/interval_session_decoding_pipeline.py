"""Session-level interval decoding with balanced accuracy and shuffle control."""

from __future__ import annotations

import hashlib
import json
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

from interval_decoding_pipeline import DEFAULT_TARGETS, prepare_interval_table

DEFAULT_SCORE_METRIC = "balanced_accuracy"

RESULT_COLUMNS = [
    "assembly",
    "session_id",
    "target_name",
    "target_col",
    "interval_name",
    "n_trials",
    "min_class_count",
    "n_features",
    "n_bins_used",
    "n_splits",
    "balanced_accuracy",
    "shuffle_n",
    "shuffle_n_effective",
    "shuffle_fail_count",
    "shuffle_null_mean",
    "shuffle_null_std",
    "shuffle_effect",
    "p_shuffle",
    "significant_shuffle_0_05",
]


@dataclass
class SessionDecodeConfig:
    assembly_col: str = "Assembly007"
    min_trials_per_session: int = 20
    min_class_count: int = 4
    max_cv_splits: int = 5
    random_state: int = 42
    shuffle_n: int = 1000
    min_session_date: Optional[str] = "2024-11-27"
    drop_unparseable_session_dates: bool = True
    excluded_intervals: Optional[Sequence[str]] = None


def _target_filter(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    # Keep only the binary labels used by the original decoder.
    out = df.copy()
    allowed = [1, 2] if target_col == "cue" else [0, 1]
    out = out[out[target_col].isin(allowed)]
    out[target_col] = out[target_col].astype(int)
    return out


def _build_session_trial_matrix(
    df: pd.DataFrame,
    session_col: str,
    trial_col: str,
    interval_col: str,
    feature_col: str,
    target_col: str,
) -> pd.DataFrame:
    # Keep only the columns needed to build one trial-by-bin matrix.
    cols = [session_col, trial_col, interval_col, "time_bin_idx", feature_col, target_col]
    out = df[cols].copy()

    # Drop invalid IDs, coerce numeric values, and keep valid decoder labels.
    out = out.dropna(subset=[session_col, trial_col, interval_col, "time_bin_idx", target_col])
    out[target_col] = pd.to_numeric(out[target_col], errors="coerce")
    out[feature_col] = pd.to_numeric(out[feature_col], errors="coerce")
    out = out.dropna(subset=[target_col])
    out = _target_filter(out, target_col)
    if out.empty:
        return out

    # Stack interval bins into one feature vector per trial.
    index_cols = [session_col, trial_col, interval_col]
    label_df = out.groupby(index_cols, as_index=False)[target_col].first()
    wide_df = (
        out.pivot_table(
            index=index_cols,
            columns="time_bin_idx",
            values=feature_col,
            aggfunc="mean",
        )
        .sort_index(axis=1)
        .rename(columns=lambda bin_idx: f"{feature_col}_bin_{int(bin_idx):03d}")
        .reset_index()
    )
    return label_df.merge(wide_df, on=index_cols, how="left")


def _build_cv_splits(y: np.ndarray, cfg: SessionDecodeConfig) -> tuple[List[tuple[np.ndarray, np.ndarray]], int]:
    # Match the original fold count based on the smallest class.
    n_splits = int(min(int(cfg.max_cv_splits), int(pd.Series(y).value_counts(dropna=True).min())))
    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=int(cfg.random_state),
    )
    return list(cv.split(np.zeros(len(y)), y)), n_splits


def _standardize_train_test(x_train: np.ndarray, x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Impute missing values from the training fold before scaling.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        median = np.nanmedian(x_train, axis=0)
    median = np.nan_to_num(median, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)

    xtr = np.asarray(x_train, dtype=np.float32).copy()
    nan_i, nan_j = np.where(np.isnan(xtr))
    if nan_i.size:
        xtr[nan_i, nan_j] = median[nan_j]

    # Standardize both folds using the training-fold statistics.
    mean = xtr.mean(axis=0).astype(np.float32, copy=False)
    std = xtr.std(axis=0).astype(np.float32, copy=False)
    std = np.where(std > 1e-6, std, 1.0).astype(np.float32, copy=False)
    xtr = ((xtr - mean) / std).astype(np.float32, copy=False)

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
    # Fit the same balanced logistic regression used by the current pipeline.
    xtr, xte = _standardize_train_test(x_train, x_test)
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
    # Derive deterministic shuffle seeds for each decoded slice.
    payload = "||".join(str(part) for part in parts).encode("utf-8")
    return int(hashlib.md5(payload).hexdigest()[:8], 16)


def _empty_shuffle_stats(cfg: SessionDecodeConfig) -> Dict[str, object]:
    # Fill the compact shuffle fields when no null distribution is available.
    return {
        "shuffle_n": int(max(0, cfg.shuffle_n)),
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
    splits: Sequence[tuple[np.ndarray, np.ndarray]],
    cfg: SessionDecodeConfig,
    observed_metric: float,
    random_state: int,
) -> Dict[str, object]:
    # Reuse the observed folds while shuffling labels to build the null.
    stats = _empty_shuffle_stats(cfg)
    n_shuffles = int(stats["shuffle_n"])
    if n_shuffles <= 0 or not np.isfinite(observed_metric):
        stats["shuffle_n"] = 0
        return stats

    rng = np.random.default_rng(int(random_state))
    null_scores: List[float] = []
    fail_count = 0

    # Skip shuffled splits that collapse to a single class in either fold.
    for shuffle_idx in range(n_shuffles):
        y_perm = rng.permutation(y_true)
        fold_scores: List[float] = []
        failed = False

        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            y_tr = y_perm[train_idx]
            y_te = y_perm[test_idx]
            if np.unique(y_tr).size != 2 or np.unique(y_te).size != 2:
                failed = True
                break

            y_hat = _fit_predict(
                x_train=x_all[train_idx],
                y_train=y_tr,
                x_test=x_all[test_idx],
                random_state=int(random_state + shuffle_idx * 1000 + fold_idx),
            )
            fold_scores.append(float(balanced_accuracy_score(y_te, y_hat)))

        if failed or not fold_scores:
            fail_count += 1
            continue

        null_scores.append(float(np.mean(fold_scores)))

    stats["shuffle_fail_count"] = int(fail_count)
    if not null_scores:
        return stats

    # Summarize the empirical null and compute the same right-tail p-value.
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


def run_session_interval_decoding(
    ens_df: pd.DataFrame,
    cfg: Optional[SessionDecodeConfig] = None,
    targets: Optional[Dict[str, str]] = None,
    progress: bool = False,
    log_every: int = 25,
    target_names: Optional[Sequence[str]] = None,
    interval_names: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Run session-level interval decoding and return compact result rows."""

    cfg = cfg or SessionDecodeConfig()
    target_map = targets or DEFAULT_TARGETS
    target_filter = {str(value) for value in target_names} if target_names else None
    interval_filter = {str(value) for value in interval_names} if interval_names else None

    # Prepare one row per interval bin using the shared interval table helper.
    (
        df,
        session_col,
        trial_col,
        interval_col,
        feature_cols,
        _interval_len_bins,
        _,
    ) = prepare_interval_table(ens_df, cfg)
    feature_col = feature_cols[0]

    results: List[Dict[str, object]] = []
    skipped = {"too_few_trials": 0, "single_class": 0, "too_few_per_class": 0}
    started = time.perf_counter()
    task_count = 0

    if progress:
        print(
            f"[start] session-level feature={feature_col} targets={list(target_map.keys())} "
            f"shuffle_n={int(cfg.shuffle_n)}",
            flush=True,
        )

    # Decode each requested target independently.
    for target_name, target_col in target_map.items():
        if target_filter is not None and str(target_name) not in target_filter:
            continue

        trial_matrix = _build_session_trial_matrix(
            df=df,
            session_col=session_col,
            trial_col=trial_col,
            interval_col=interval_col,
            feature_col=feature_col,
            target_col=target_col,
        )
        if trial_matrix.empty:
            continue

        feature_matrix_cols = [
            col
            for col in trial_matrix.columns
            if col not in {session_col, trial_col, interval_col, target_col}
        ]
        if not feature_matrix_cols:
            continue

        # Keep the original interval-wide bin stack for each interval.
        for interval_name, interval_df in trial_matrix.groupby(interval_col, sort=False):
            if interval_filter is not None and str(interval_name) not in interval_filter:
                continue

            interval_feature_cols = [col for col in feature_matrix_cols if interval_df[col].notna().any()]
            if not interval_feature_cols:
                continue

            # Fit one session-specific decoder per interval.
            for session_id, session_df in interval_df.groupby(session_col, sort=False):
                task_count += 1
                session_df = session_df.sort_values(trial_col).copy()
                n_trials = int(len(session_df))
                y_raw = session_df[target_col].to_numpy(dtype=int)
                class_counts = pd.Series(y_raw).value_counts(dropna=True)
                n_classes = int(len(class_counts))
                min_class = int(class_counts.min()) if n_classes else 0

                if progress and (task_count == 1 or task_count % max(1, int(log_every)) == 0):
                    elapsed = time.perf_counter() - started
                    print(
                        f"[decode] task={task_count} session={session_id} target={target_name} "
                        f"interval={interval_name} n={n_trials} elapsed_s={elapsed:.1f}",
                        flush=True,
                    )

                # Skip slices that cannot support the original stratified decoder.
                if n_trials < int(cfg.min_trials_per_session):
                    skipped["too_few_trials"] += 1
                    continue
                if n_classes != 2:
                    skipped["single_class"] += 1
                    continue
                if min_class < int(cfg.min_class_count):
                    skipped["too_few_per_class"] += 1
                    continue

                # Keep the same label remapping and interval-wide feature set.
                label_values = np.sort(np.unique(y_raw))
                y01 = np.where(y_raw == label_values[0], 0, 1).astype(np.int64, copy=False)
                x_all = session_df[interval_feature_cols].to_numpy(dtype=np.float32)
                n_bins_used = int(sum(session_df[interval_feature_cols].notna().any(axis=0)))
                splits, n_splits = _build_cv_splits(y01, cfg)

                # Evaluate balanced accuracy with the original fold-specific seeds.
                fold_bacc: List[float] = []
                for fold_idx, (train_idx, test_idx) in enumerate(splits):
                    y_hat = _fit_predict(
                        x_train=x_all[train_idx],
                        y_train=y01[train_idx],
                        x_test=x_all[test_idx],
                        random_state=int(cfg.random_state + fold_idx),
                    )
                    fold_bacc.append(float(balanced_accuracy_score(y01[test_idx], y_hat)))

                observed_bacc = float(np.mean(fold_bacc))

                # Estimate significance with the same empirical shuffle procedure.
                slice_seed = _stable_seed(session_id, target_name, interval_name)
                shuffle_stats = _shuffle_significance_empirical(
                    x_all=x_all,
                    y_true=y01,
                    splits=splits,
                    cfg=cfg,
                    observed_metric=observed_bacc,
                    random_state=int(cfg.random_state + slice_seed),
                )

                results.append(
                    {
                        "assembly": feature_col,
                        "session_id": str(session_id),
                        "target_name": target_name,
                        "target_col": target_col,
                        "interval_name": str(interval_name),
                        "n_trials": n_trials,
                        "min_class_count": min_class,
                        "n_features": int(len(interval_feature_cols)),
                        "n_bins_used": n_bins_used,
                        "n_splits": int(n_splits),
                        "balanced_accuracy": observed_bacc,
                        **shuffle_stats,
                    }
                )

    if progress:
        elapsed_s = time.perf_counter() - started
        print(
            "[done] "
            f"session-level tasks={task_count} rows={len(results)} "
            f"skipped={skipped} elapsed_s={elapsed_s:.1f}",
            flush=True,
        )

    return pd.DataFrame(results, columns=RESULT_COLUMNS)


def load_session_decode_config(config_path: Path | str) -> SessionDecodeConfig:
    # Load only the config fields used by the compact session decoder.
    payload = json.loads(Path(config_path).read_text())
    valid_fields = set(SessionDecodeConfig.__dataclass_fields__.keys())
    cfg_kwargs = {key: value for key, value in payload.items() if key in valid_fields}
    return SessionDecodeConfig(**cfg_kwargs)


__all__ = [
    "DEFAULT_SCORE_METRIC",
    "DEFAULT_TARGETS",
    "RESULT_COLUMNS",
    "SessionDecodeConfig",
    "load_session_decode_config",
    "run_session_interval_decoding",
]
