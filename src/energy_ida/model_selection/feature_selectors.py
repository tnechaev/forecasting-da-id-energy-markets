from __future__ import annotations

import json
import os
import re
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits

from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


VALID_WINDOW_MODES = {"expanding", "rolling", "rolling_gap_aware", "rolling_contiguous"}


@dataclass
class FeatureSelectionResult:
    model_type: str
    target_name: str
    target_column: str
    selected_features: list[str]
    feature_summary: pd.DataFrame
    fold_metrics: pd.DataFrame
    config: dict[str, Any]

    def save(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)

        self.feature_summary.to_csv(output_dir / "feature_summary.csv", index=False)
        self.fold_metrics.to_csv(output_dir / "fold_metrics.csv", index=False)

        with open(output_dir / "selected_features.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "model_type": self.model_type,
                    "target_name": self.target_name,
                    "target_column": self.target_column,
                    "selected_features": self.selected_features,
                    "config": self.config,
                },
                f,
                indent=2,
            )


def rmse(y_true, y_pred) -> float:
    tmp = (
        pd.DataFrame({"y": y_true, "pred": y_pred})
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    if tmp.empty:
        return float("nan")
    return float(mean_squared_error(tmp["y"], tmp["pred"]) ** 0.5)


def mae(y_true, y_pred) -> float:
    tmp = (
        pd.DataFrame({"y": y_true, "pred": y_pred})
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    if tmp.empty:
        return float("nan")
    return float(mean_absolute_error(tmp["y"], tmp["pred"]))


def dedupe(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))


def make_safe_imputer() -> SimpleImputer:
    return SimpleImputer(strategy="median", keep_empty_features=True)


def auto_walk_forward_params(active_target_days: int) -> dict[str, int | str]:
    if active_target_days < 20:
        train_window_days = max(5, int(round(active_target_days * 0.50)))
        val_days = max(3, int(round(active_target_days * 0.20)))
    elif active_target_days < 730:
        train_window_days = max(20, int(round(active_target_days * 0.50)))
        val_days = max(5, int(round(active_target_days * 0.15)))
    else:
        train_window_days = 365
        val_days = 30

    gap_days = 1

    if train_window_days + gap_days + val_days > active_target_days:
        gap_days = 0

    if train_window_days + gap_days + val_days > active_target_days:
        val_days = max(3, active_target_days - train_window_days - gap_days)

    return {
        "active_target_days": int(active_target_days),
        "train_window_days": int(train_window_days),
        "gap_days": int(gap_days),
        "val_days": int(val_days),
        "step_days": int(val_days),
        "window_mode": "expanding",
    }



def make_walk_forward_folds(
    df: pd.DataFrame,
    time_col: str = "timestamp_utc",
    train_window_days: int = 365,
    val_days: int = 30,
    gap_days: int = 1,
    step_days: int = 30,
    window_mode: str = "expanding",
    max_folds: int | None = None,
    min_train_day_coverage: float = 0.80,
    min_val_day_coverage: float = 0.80,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Walk-forward folds over UTC dates.

    Modes:
      expanding:
        Uses all available observed UTC dates before validation, excluding
        gap_days. This keeps fragmented older history.

      rolling_gap_aware:
        Uses the previous N observed UTC dates before validation, excluding
        gap_days. This is the main research mode when history has gaps: it
        preserves valid old data, but fold metrics report gap diagnostics so the
        gap is visible.

      rolling:
        Alias for rolling_gap_aware.

      rolling_contiguous:
        Uses the previous N calendar UTC days before validation, excluding
        gap_days. This is a stricter production-style robustness mode and may
        skip old disconnected data blocks.

    All modes use timestamp_utc as the date basis. Local-time columns may still
    be used as features.
    """
    if window_mode == "rolling":
        window_mode = "rolling_gap_aware"

    if window_mode not in VALID_WINDOW_MODES:
        raise ValueError(f"window_mode must be one of {sorted(VALID_WINDOW_MODES)}")

    if train_window_days <= 0:
        raise ValueError("train_window_days must be positive.")
    if val_days <= 0:
        raise ValueError("val_days must be positive.")
    if gap_days < 0:
        raise ValueError("gap_days must be >= 0.")
    if step_days <= 0:
        raise ValueError("step_days must be positive.")
    if not (0 < min_train_day_coverage <= 1):
        raise ValueError("min_train_day_coverage must be in (0, 1].")
    if not (0 < min_val_day_coverage <= 1):
        raise ValueError("min_val_day_coverage must be in (0, 1].")
    if df.empty:
        return []

    tmp = df[[time_col]].copy()
    tmp[time_col] = pd.to_datetime(tmp[time_col], utc=True, errors="coerce")
    tmp = tmp.dropna(subset=[time_col])
    tmp["date"] = tmp[time_col].dt.date

    if tmp.empty:
        return []

    observed_dates = np.array(sorted(tmp["date"].dropna().unique()))
    observed_date_set = set(observed_dates)

    if len(observed_dates) == 0:
        return []

    # ------------------------------------------------------------------
    # Observed-date modes: keep fragmented but valid history.
    # ------------------------------------------------------------------
    if window_mode in {"rolling_gap_aware", "expanding"}:
        n_observed = len(observed_dates)

        if n_observed < train_window_days + gap_days + val_days:
            return []

        folds = []
        train_end_pos = train_window_days - 1

        while True:
            val_start_pos = train_end_pos + gap_days + 1
            val_end_pos = val_start_pos + val_days - 1

            if val_end_pos >= n_observed:
                break

            val_dates = set(observed_dates[val_start_pos:val_end_pos + 1])

            if window_mode == "expanding":
                train_dates = set(observed_dates[:train_end_pos + 1])
            else:
                train_start_pos = train_end_pos - train_window_days + 1
                if train_start_pos < 0:
                    break
                train_dates = set(observed_dates[train_start_pos:train_end_pos + 1])

            train_idx = tmp.index[tmp["date"].isin(train_dates)].to_numpy()
            val_idx = tmp.index[tmp["date"].isin(val_dates)].to_numpy()

            if len(train_idx) > 0 and len(val_idx) > 0:
                folds.append((train_idx, val_idx))

            if max_folds is not None and len(folds) >= max_folds:
                break

            train_end_pos += step_days

        return folds

    # ------------------------------------------------------------------
    # Strict contiguous-calendar mode.
    # ------------------------------------------------------------------
    calendar_dates = np.array(
        pd.date_range(
            start=pd.Timestamp(observed_dates.min()),
            end=pd.Timestamp(observed_dates.max()),
            freq="D",
        ).date
    )

    n_calendar = len(calendar_dates)

    if n_calendar < train_window_days + gap_days + val_days:
        return []

    min_train_observed_days = max(1, int(np.ceil(train_window_days * min_train_day_coverage)))
    min_val_observed_days = max(1, int(np.ceil(val_days * min_val_day_coverage)))

    folds = []
    train_end_pos = train_window_days - 1

    while True:
        val_start_pos = train_end_pos + gap_days + 1
        val_end_pos = val_start_pos + val_days - 1

        if val_end_pos >= n_calendar:
            break

        train_start_pos = train_end_pos - train_window_days + 1
        if train_start_pos < 0:
            break

        train_window_dates = list(calendar_dates[train_start_pos:train_end_pos + 1])
        val_window_dates = list(calendar_dates[val_start_pos:val_end_pos + 1])

        train_observed_days = sum(d in observed_date_set for d in train_window_dates)
        val_observed_days = sum(d in observed_date_set for d in val_window_dates)

        if (
            train_observed_days >= min_train_observed_days
            and val_observed_days >= min_val_observed_days
        ):
            train_dates = set(train_window_dates)
            val_dates = set(val_window_dates)

            train_idx = tmp.index[tmp["date"].isin(train_dates)].to_numpy()
            val_idx = tmp.index[tmp["date"].isin(val_dates)].to_numpy()

            if len(train_idx) > 0 and len(val_idx) > 0:
                folds.append((train_idx, val_idx))

        if max_folds is not None and len(folds) >= max_folds:
            break

        train_end_pos += step_days

    return folds


def fold_gap_diagnostics(df: pd.DataFrame, idx: np.ndarray) -> dict[str, Any]:
    s = pd.to_datetime(df.iloc[idx]["timestamp_utc"], utc=True, errors="coerce").dropna()
    dates = pd.Series(sorted(s.dt.date.unique()))

    if len(dates) <= 1:
        return {
            "observed_utc_days": int(len(dates)),
            "max_gap_days": 0,
            "mean_gap_days": 0.0,
        }

    deltas = pd.to_datetime(dates).diff().dt.days.dropna()

    return {
        "observed_utc_days": int(len(dates)),
        "max_gap_days": int(deltas.max()),
        "mean_gap_days": float(deltas.mean()),
    }


def fold_bounds(df: pd.DataFrame, idx: np.ndarray) -> tuple[str, str]:
    s = pd.to_datetime(df.iloc[idx]["timestamp_utc"], utc=True)
    return str(s.min()), str(s.max())


def no_folds_message(
    work: pd.DataFrame,
    train_window_days: int,
    val_days: int,
    gap_days: int,
    step_days: int,
    window_mode: str,
) -> str:
    active_days = pd.to_datetime(work["timestamp_utc"], utc=True).dt.date.nunique()
    suggestion = auto_walk_forward_params(active_days)

    return (
        "\nNo valid walk-forward folds.\n\n"
        f"Available active UTC target days: {active_days}\n"
        "Requested fold setup:\n"
        f"  fold_date_basis=timestamp_utc\n"
        f"  window_mode={window_mode}\n"
        f"  train_window_days={train_window_days}\n"
        f"  gap_days={gap_days}\n"
        f"  val_days={val_days}\n"
        f"  step_days={step_days}\n\n"
        f"Suggested auto setup: {suggestion}\n"
    )


def is_lagged_or_rolling_feature(feature: str) -> bool:
    return "_lag_" in feature or "_rolling_" in feature


def is_leakage_risky_feature(feature: str) -> bool:
    if is_lagged_or_rolling_feature(feature):
        return False

    if feature.startswith("actual_"):
        return True

    if "forecast_error" in feature:
        return True

    if feature.endswith("_actual_load"):
        return True

    if "_actual_" in feature:
        return True

    return False


def feature_family(feature: str) -> str:
    f = feature

    f = re.sub(r"_lag_\d+d_same_local_qh$", "", f)
    f = re.sub(r"_rolling_(mean|median|std)_\d+d_same_local_qh$", "", f)

    if "regime_" in f:
        if "vol_" in f:
            return "regime_volatility"
        if "mean_shift" in f:
            return "regime_mean_shift"
        if "abs_median" in f:
            return "regime_level"
        return "regime"

    if f in {
        "hour_sin",
        "hour_cos",
        "quarter_sin",
        "quarter_cos",
        "weekday_sin",
        "weekday_cos",
        "month_sin",
        "month_cos",
        "is_weekend",
    }:
        return "calendar"

    if f.startswith("ida1_"):
        return "ida1_history"
    if f.startswith("ida2_"):
        return "ida2_history"
    if f.startswith("ida3_"):
        return "ida3_history"
    if f.startswith("continuous_15min_"):
        return "continuous_history"

    if f.startswith("da_price"):
        return "da_price"

    if "forecast_revision" in f or "revision_id_minus_da" in f:
        return "forecast_revision"

    if "forecast_da" in f:
        return "da_forecast"

    if "forecast_id" in f:
        return "id_forecast"

    if "forecast_error" in f:
        return "lagged_forecast_error"

    if f.startswith("wind"):
        return "wind"

    if f.startswith("solar"):
        return "solar"

    if f.startswith("load") or "residual_load" in f:
        return "load_residual"

    return f


def feature_base_signal(feature: str) -> str:
    f = feature

    f = re.sub(r"_lag_\d+d_same_local_qh$", "", f)
    f = re.sub(r"_rolling_(mean|median|std)_\d+d_same_local_qh$", "", f)
    f = re.sub(r"_regime_vol_28d$", "", f)
    f = re.sub(r"_regime_vol_7d_over_28d$", "", f)
    f = re.sub(r"_regime_mean_shift_7d_vs_28d_z$", "", f)
    f = re.sub(r"_regime_abs_median_28d_over_vol$", "", f)

    return f


def feature_pool(feature: str) -> str:
    fam = feature_family(feature)

    if fam == "calendar":
        return "calendar"

    if fam in {
        "ida1_history",
        "ida2_history",
        "ida3_history",
        "continuous_history",
        "da_price",
    }:
        return "market_history"

    if fam in {
        "regime_volatility",
        "regime_mean_shift",
        "regime_level",
        "regime",
    }:
        return "regime"

    if fam in {
        "da_forecast",
        "id_forecast",
        "wind",
        "solar",
        "load_residual",
    }:
        return "forecast_level"

    if fam == "forecast_revision":
        return "forecast_revision"

    if fam == "lagged_forecast_error":
        return "lagged_forecast_error"

    return "other"


def prefilter_feature_columns(
    df: pd.DataFrame,
    feature_columns: list[str],
    min_non_missing_fraction: float = 0.10,
    min_unique_values: int = 2,
) -> tuple[list[str], pd.DataFrame]:
    df = df.loc[:, ~df.columns.duplicated()].copy()
    feature_columns = dedupe(feature_columns)

    rows = []
    kept = []

    n = len(df)
    min_non_missing_count = max(10, int(n * min_non_missing_fraction))

    for c in feature_columns:
        if c not in df.columns:
            rows.append(
                {
                    "feature": c,
                    "prefilter_keep": False,
                    "prefilter_reason": "missing_column",
                    "non_missing_count": 0,
                    "non_missing_fraction": 0.0,
                    "unique_values": 0,
                    "family": feature_family(c),
                    "pool": feature_pool(c),
                    "base_signal": feature_base_signal(c),
                }
            )
            continue

        s = df[c]
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0]

        if is_leakage_risky_feature(c):
            rows.append(
                {
                    "feature": c,
                    "prefilter_keep": False,
                    "prefilter_reason": "blocked_possible_lookahead",
                    "non_missing_count": int(s.notna().sum()),
                    "non_missing_fraction": float(s.notna().mean()),
                    "unique_values": int(s.nunique(dropna=True)),
                    "family": feature_family(c),
                    "pool": feature_pool(c),
                    "base_signal": feature_base_signal(c),
                }
            )
            continue

        if not pd.api.types.is_numeric_dtype(s):
            rows.append(
                {
                    "feature": c,
                    "prefilter_keep": False,
                    "prefilter_reason": "non_numeric",
                    "non_missing_count": int(s.notna().sum()),
                    "non_missing_fraction": float(s.notna().mean()),
                    "unique_values": int(s.nunique(dropna=True)),
                    "family": feature_family(c),
                    "pool": feature_pool(c),
                    "base_signal": feature_base_signal(c),
                }
            )
            continue

        non_missing_count = int(s.notna().sum())
        non_missing_fraction = float(s.notna().mean())
        unique_values = int(s.nunique(dropna=True))

        if non_missing_count < min_non_missing_count:
            rows.append(
                {
                    "feature": c,
                    "prefilter_keep": False,
                    "prefilter_reason": "too_sparse",
                    "non_missing_count": non_missing_count,
                    "non_missing_fraction": non_missing_fraction,
                    "unique_values": unique_values,
                    "family": feature_family(c),
                    "pool": feature_pool(c),
                    "base_signal": feature_base_signal(c),
                }
            )
            continue

        if unique_values < min_unique_values:
            rows.append(
                {
                    "feature": c,
                    "prefilter_keep": False,
                    "prefilter_reason": "constant_or_all_missing",
                    "non_missing_count": non_missing_count,
                    "non_missing_fraction": non_missing_fraction,
                    "unique_values": unique_values,
                    "family": feature_family(c),
                    "pool": feature_pool(c),
                    "base_signal": feature_base_signal(c),
                }
            )
            continue

        kept.append(c)
        rows.append(
            {
                "feature": c,
                "prefilter_keep": True,
                "prefilter_reason": "kept",
                "non_missing_count": non_missing_count,
                "non_missing_fraction": non_missing_fraction,
                "unique_values": unique_values,
                "family": feature_family(c),
                "pool": feature_pool(c),
                "base_signal": feature_base_signal(c),
            }
        )

    return kept, pd.DataFrame(rows)


def sample_indices(idx: np.ndarray, max_rows: int | None, random_state: int) -> np.ndarray:
    if max_rows is None or max_rows <= 0 or len(idx) <= max_rows:
        return idx

    rng = np.random.default_rng(random_state)
    out = rng.choice(idx, size=max_rows, replace=False)
    return np.sort(out)


def compute_univariate_scores(
    df: pd.DataFrame,
    features: list[str],
    target_column: str,
    max_rows: int = 60_000,
    random_state: int = 42,
) -> pd.DataFrame:
    features = dedupe(features)

    if not features:
        return pd.DataFrame(
            columns=["feature", "univariate_abs_spearman", "family", "pool", "base_signal"]
        )

    idx = np.arange(len(df))
    idx = sample_indices(idx, max_rows=max_rows, random_state=random_state)

    sample = df.iloc[idx][features + [target_column]].copy()

    rows = []
    y = sample[target_column].astype(float)

    for f in features:
        x = sample[f].astype(float)
        tmp = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()

        if len(tmp) < 100 or tmp["x"].nunique() < 2 or tmp["y"].nunique() < 2:
            score = 0.0
        else:
            score = abs(float(tmp["x"].corr(tmp["y"], method="spearman")))

        rows.append(
            {
                "feature": f,
                "univariate_abs_spearman": score if pd.notna(score) else 0.0,
                "family": feature_family(f),
                "pool": feature_pool(f),
                "base_signal": feature_base_signal(f),
            }
        )

    return pd.DataFrame(rows)


def prune_correlated_features(
    df: pd.DataFrame,
    features: list[str],
    target_column: str,
    corr_threshold: float = 0.92,
    max_rows: int = 60_000,
    random_state: int = 42,
) -> tuple[list[str], pd.DataFrame]:
    features = dedupe(features)

    if len(features) <= 1:
        return features, pd.DataFrame(
            {
                "feature": features,
                "corr_prune_keep": True,
                "corr_prune_reason": "only_feature",
                "correlated_with": "",
                "max_abs_corr_with_kept": 0.0,
                "univariate_abs_spearman": 0.0,
                "family": [feature_family(f) for f in features],
                "pool": [feature_pool(f) for f in features],
                "base_signal": [feature_base_signal(f) for f in features],
            }
        )

    scores = compute_univariate_scores(
        df=df,
        features=features,
        target_column=target_column,
        max_rows=max_rows,
        random_state=random_state,
    )

    ordered = scores.sort_values(
        ["univariate_abs_spearman", "feature"],
        ascending=[False, True],
    )["feature"].tolist()

    idx = np.arange(len(df))
    idx = sample_indices(idx, max_rows=max_rows, random_state=random_state + 7)
    sample = df.iloc[idx][ordered].copy()
    sample = sample.replace([np.inf, -np.inf], np.nan)

    kept = []
    rows = []

    for f in ordered:
        if not kept:
            kept.append(f)
            rows.append(
                {
                    "feature": f,
                    "corr_prune_keep": True,
                    "corr_prune_reason": "first_ranked_feature",
                    "correlated_with": "",
                    "max_abs_corr_with_kept": 0.0,
                }
            )
            continue

        max_corr = 0.0
        corr_with = ""

        for k in kept:
            tmp = sample[[f, k]].dropna()
            if len(tmp) < 100 or tmp[f].nunique() < 2 or tmp[k].nunique() < 2:
                corr = 0.0
            else:
                corr = abs(float(tmp[f].corr(tmp[k], method="spearman")))

            if pd.notna(corr) and corr > max_corr:
                max_corr = corr
                corr_with = k

        if max_corr >= corr_threshold:
            rows.append(
                {
                    "feature": f,
                    "corr_prune_keep": False,
                    "corr_prune_reason": "too_correlated_with_kept_feature",
                    "correlated_with": corr_with,
                    "max_abs_corr_with_kept": max_corr,
                }
            )
        else:
            kept.append(f)
            rows.append(
                {
                    "feature": f,
                    "corr_prune_keep": True,
                    "corr_prune_reason": "kept",
                    "correlated_with": corr_with,
                    "max_abs_corr_with_kept": max_corr,
                }
            )

    out = pd.DataFrame(rows).merge(scores, on="feature", how="left")
    return kept, out


def prune_correlated_features_by_pool(
    df: pd.DataFrame,
    features: list[str],
    target_column: str,
    corr_threshold: float = 0.92,
    max_rows: int = 60_000,
    random_state: int = 42,
) -> tuple[list[str], pd.DataFrame]:
    features = dedupe(features)

    if not features:
        return [], pd.DataFrame()

    by_pool: dict[str, list[str]] = {}
    for f in features:
        by_pool.setdefault(feature_pool(f), []).append(f)

    kept_all = []
    summaries = []

    for pool_name, pool_features in sorted(by_pool.items()):
        kept, summary = prune_correlated_features(
            df=df,
            features=pool_features,
            target_column=target_column,
            corr_threshold=corr_threshold,
            max_rows=max_rows,
            random_state=random_state,
        )

        summary["pool"] = pool_name
        kept_all.extend(kept)
        summaries.append(summary)

    if summaries:
        out = pd.concat(summaries, ignore_index=True)
    else:
        out = pd.DataFrame()

    return dedupe(kept_all), out


def apply_conservative_final_selection(
    summary: pd.DataFrame,
    score_col: str,
    max_total_features: int,
    max_features_per_family: int,
    max_features_per_base_signal: int = 4,
    min_secondary_score_quantile: float = 0.75,
) -> list[str]:
    """
    Final feature selection.

    This intentionally does NOT force feature-pool quotas. The previous
    diversified selector hurt IDA1 because it pushed out strong own-spread
    persistence features to make room for weak regime/calendar/forecast features.

    Logic:
      1. Use strict candidates first. These are features that passed the
         model-specific stability gate, prefiltering and correlation pruning.
      2. If too few strict candidates exist, add secondary candidates using a
         data-driven score threshold within the current feature universe.
      3. Control redundancy softly by capping repeated transformations of the
         same base signal, but allow several own-target history horizons. For
         electricity spreads/prices, 3d mean, 7d median, 28d median and 7d lag
         can all be useful even if correlated.
      4. Do not require regime features. Regime features are selected only when
         they pass the same evidence rules as everything else.
    """
    work = summary.copy()

    if work.empty:
        return []

    if "feature" not in work.columns:
        return []

    if "selected_pre_cap" not in work.columns:
        work["selected_pre_cap"] = False
    work["selected_pre_cap"] = work["selected_pre_cap"].fillna(False).astype(bool)

    if score_col not in work.columns:
        work[score_col] = 0.0

    numeric_defaults = {
        score_col: 0.0,
        "clean_selection_frequency": 0.0,
        "selection_frequency": 0.0,
        "univariate_abs_spearman": 0.0,
        "best_rank": np.inf,
        "mean_rank": np.inf,
        "non_missing_fraction": 0.0,
    }

    for col, default in numeric_defaults.items():
        if col not in work.columns:
            work[col] = default
        work[col] = pd.to_numeric(work[col], errors="coerce").fillna(default)

    for col in ["prefilter_keep", "corr_prune_keep"]:
        if col not in work.columns:
            work[col] = True
        work[col] = work[col].fillna(False).astype(bool)

    work["family"] = work["feature"].map(feature_family)
    work["pool"] = work["feature"].map(feature_pool)
    work["base_signal"] = work["feature"].map(feature_base_signal)

    usable = work[
        work["prefilter_keep"]
        & work["corr_prune_keep"]
        & (work[score_col] > 0)
        & (work["non_missing_fraction"] >= 0.10)
    ].copy()

    if usable.empty:
        return []

    positive_scores = usable.loc[usable[score_col] > 0, score_col]
    if positive_scores.empty:
        secondary_score_cutoff = 0.0
    else:
        secondary_score_cutoff = float(positive_scores.quantile(min_secondary_score_quantile))

    usable["strict_candidate"] = usable["selected_pre_cap"]

    usable["secondary_candidate"] = (
        usable["strict_candidate"]
        | (
            (usable[score_col] >= secondary_score_cutoff)
            & (
                (usable["clean_selection_frequency"] > 0)
                | (usable["selection_frequency"] > 0)
                | (usable["best_rank"] <= 25)
                | (usable["univariate_abs_spearman"] >= usable["univariate_abs_spearman"].quantile(0.75))
            )
        )
    )

    candidates = usable[usable["secondary_candidate"]].copy()
    if candidates.empty:
        candidates = usable.copy()

    candidates = candidates.sort_values(
        [
            "strict_candidate",
            "clean_selection_frequency",
            "selection_frequency",
            score_col,
            "univariate_abs_spearman",
            "best_rank",
            "feature",
        ],
        ascending=[False, False, False, False, False, True, True],
    )

    selected: list[str] = []
    family_counts: dict[str, int] = {}
    base_counts: dict[str, int] = {}

    # Data-driven fallback size: if the model found many stable candidates, keep
    # more; if it found only a few, keep fewer. The upper bound prevents large,
    # noisy feature sets.
    n_strict = int(candidates["strict_candidate"].sum())
    dynamic_max_features = min(
        max_total_features,
        max(6, n_strict + int(np.ceil(np.sqrt(max(n_strict, 1))))),
    )

    def can_add(row: pd.Series) -> bool:
        f = row["feature"]
        fam = row["family"]
        base = row["base_signal"]

        if f in selected:
            return False

        if len(selected) >= dynamic_max_features:
            return False

        if family_counts.get(fam, 0) >= max_features_per_family:
            return False

        if base_counts.get(base, 0) >= max_features_per_base_signal:
            return False

        return True

    def add(row: pd.Series) -> None:
        f = row["feature"]
        fam = row["family"]
        base = row["base_signal"]

        selected.append(f)
        family_counts[fam] = family_counts.get(fam, 0) + 1
        base_counts[base] = base_counts.get(base, 0) + 1

    # Pass 1: strict stable candidates.
    for _, row in candidates[candidates["strict_candidate"]].iterrows():
        if can_add(row):
            add(row)

    # Pass 2: secondary candidates only if there is still room.
    for _, row in candidates[~candidates["strict_candidate"]].iterrows():
        if can_add(row):
            add(row)

    return selected

def fallback_univariate_final_selection(
    summary: pd.DataFrame,
    max_total_features: int,
    max_features_per_family: int,
    min_non_missing_fraction: float = 0.20,
) -> list[str]:
    work = summary.copy()

    if "univariate_abs_spearman" not in work.columns:
        return []

    work["univariate_abs_spearman"] = pd.to_numeric(
        work["univariate_abs_spearman"],
        errors="coerce",
    ).fillna(0.0)

    work["non_missing_fraction"] = pd.to_numeric(
        work.get("non_missing_fraction", 0.0),
        errors="coerce",
    ).fillna(0.0)

    work["prefilter_keep"] = work.get("prefilter_keep", False).fillna(False).astype(bool)
    work["corr_prune_keep"] = work.get("corr_prune_keep", False).fillna(False).astype(bool)
    work["family"] = work["feature"].map(feature_family)
    work["pool"] = work["feature"].map(feature_pool)
    work["base_signal"] = work["feature"].map(feature_base_signal)

    work = work[
        work["prefilter_keep"]
        & work["corr_prune_keep"]
        & (work["non_missing_fraction"] >= min_non_missing_fraction)
        & (work["univariate_abs_spearman"] > 0)
    ].copy()

    if work.empty:
        return []

    work["selected_pre_cap"] = True

    return apply_conservative_final_selection(
        summary=work,
        score_col="univariate_abs_spearman",
        max_total_features=max_total_features,
        max_features_per_family=max_features_per_family,
        max_features_per_base_signal=3,
    )


class SeasonalNaiveFeatureSelector:
    def __init__(
        self,
        target_name: str,
        target_column: str,
        feature_columns: list[str],
    ):
        self.target_name = target_name
        self.target_column = target_column
        self.feature_columns = dedupe(feature_columns)

    def select(self, df: pd.DataFrame) -> FeatureSelectionResult:
        patterns = [
            f"{self.target_column}_lag_1d_same_local_qh",
            f"{self.target_column}_lag_7d_same_local_qh",
            f"{self.target_column}_rolling_mean_7d_same_local_qh",
            f"{self.target_column}_rolling_median_7d_same_local_qh",
            f"{self.target_column}_rolling_mean_14d_same_local_qh",
            f"{self.target_column}_rolling_median_14d_same_local_qh",
        ]

        selected = [c for c in patterns if c in self.feature_columns]

        summary = pd.DataFrame(
            [
                {
                    "feature": c,
                    "selected": c in selected,
                    "selection_reason": "seasonal_naive_fixed_rule" if c in selected else "not_baseline_feature",
                    "family": feature_family(c),
                    "pool": feature_pool(c),
                    "base_signal": feature_base_signal(c),
                }
                for c in self.feature_columns
            ]
        )

        return FeatureSelectionResult(
            model_type="seasonal_naive",
            target_name=self.target_name,
            target_column=self.target_column,
            selected_features=selected,
            feature_summary=summary,
            fold_metrics=pd.DataFrame(),
            config={"selector": "fixed seasonal naive lag rule"},
        )


class ElasticNetStabilitySelector:
    def __init__(
        self,
        target_name: str,
        target_column: str,
        feature_columns: list[str],
        alpha_grid: list[float] | None = None,
        l1_ratio_grid: list[float] | None = None,
        train_window_days: int = 365,
        val_days: int = 30,
        gap_days: int = 1,
        step_days: int = 30,
        window_mode: str = "expanding",
        max_folds: int | None = None,
        min_selection_frequency: float = 0.65,
        coefficient_epsilon: float = 1e-8,
        min_non_missing_fraction: float = 0.10,
        max_train_rows_per_fold: int | None = 30_000,
        fast_mode: bool = False,
        random_state: int = 42,
        n_jobs: int = 2,
        corr_threshold: float = 0.92,
        max_total_features: int = 25,
        max_features_per_family: int = 4,
        max_allowed_overfit_ratio: float = 1.50,
    ):
        self.target_name = target_name
        self.target_column = target_column
        self.feature_columns = dedupe(feature_columns)

        if fast_mode:
            self.alpha_grid = alpha_grid or [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
            self.l1_ratio_grid = l1_ratio_grid or [0.5, 0.7, 0.9, 1.0]
        else:
            self.alpha_grid = alpha_grid or [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]
            self.l1_ratio_grid = l1_ratio_grid or [0.5, 0.7, 0.9, 1.0]

        self.train_window_days = train_window_days
        self.val_days = val_days
        self.gap_days = gap_days
        self.step_days = step_days
        self.window_mode = window_mode
        self.max_folds = max_folds

        self.min_selection_frequency = min_selection_frequency
        self.coefficient_epsilon = coefficient_epsilon
        self.min_non_missing_fraction = min_non_missing_fraction
        self.max_train_rows_per_fold = max_train_rows_per_fold
        self.fast_mode = fast_mode
        self.random_state = random_state
        self.n_jobs = n_jobs

        self.corr_threshold = corr_threshold
        self.max_total_features = max_total_features
        self.max_features_per_family = max_features_per_family
        self.max_allowed_overfit_ratio = max_allowed_overfit_ratio

    def make_model(self, alpha: float, l1_ratio: float) -> Pipeline:
        return Pipeline(
            steps=[
                ("imputer", make_safe_imputer()),
                ("scaler", StandardScaler()),
                (
                    "model",
                    ElasticNet(
                        alpha=alpha,
                        l1_ratio=l1_ratio,
                        max_iter=50_000 if self.fast_mode else 100_000,
                        tol=2e-3 if self.fast_mode else 1e-3,
                        selection="random",
                        random_state=self.random_state,
                    ),
                ),
            ]
        )

    def fit_one_fold(
        self,
        fold_id: int,
        n_folds: int,
        train_idx: np.ndarray,
        val_idx: np.ndarray,
        work: pd.DataFrame,
        kept_features: list[str],
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        start = time.perf_counter()

        sampled_train_idx = sample_indices(
            train_idx,
            max_rows=self.max_train_rows_per_fold,
            random_state=self.random_state + fold_id,
        )

        X = work[kept_features]
        y = work[self.target_column].astype(float)

        X_train = X.iloc[sampled_train_idx]
        y_train = y.iloc[sampled_train_idx]
        X_val = X.iloc[val_idx]
        y_val = y.iloc[val_idx]

        fold_all_missing_features = [
            c for c in kept_features if X_train[c].notna().sum() == 0
        ]

        n_grid = len(self.alpha_grid) * len(self.l1_ratio_grid)

        print(
            f"[ElasticNet fold {fold_id}/{n_folds}] "
            f"train_rows={len(train_idx):,}, used={len(sampled_train_idx):,}, "
            f"val_rows={len(val_idx):,}, features={len(kept_features)}, grid={n_grid}",
            flush=True,
        )

        best = None

        with threadpool_limits(limits=1):
            for alpha in self.alpha_grid:
                for l1_ratio in self.l1_ratio_grid:
                    model = self.make_model(alpha=alpha, l1_ratio=l1_ratio)

                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always", ConvergenceWarning)
                        warnings.simplefilter("ignore", UserWarning)
                        model.fit(X_train, y_train)

                    convergence_warning_count = sum(
                        isinstance(w.message, ConvergenceWarning)
                        for w in caught
                    )

                    pred_val = model.predict(X_val)
                    score = rmse(y_val, pred_val)

                    if best is None or score < best["val_rmse"]:
                        best = {
                            "model": model,
                            "alpha": alpha,
                            "l1_ratio": l1_ratio,
                            "val_rmse": score,
                            "convergence_warning_count": convergence_warning_count,
                        }

        model = best["model"]

        pred_train = model.predict(X_train)
        pred_val = model.predict(X_val)

        train_rmse = rmse(y_train, pred_train)
        val_rmse = rmse(y_val, pred_val)
        overfit_ratio = float(val_rmse / train_rmse) if train_rmse > 0 else np.nan
        clean_fold = bool(pd.notna(overfit_ratio) and overfit_ratio <= self.max_allowed_overfit_ratio)

        train_start, train_end = fold_bounds(work, train_idx)
        val_start, val_end = fold_bounds(work, val_idx)

        train_gap_diag = fold_gap_diagnostics(work, train_idx)
        val_gap_diag = fold_gap_diagnostics(work, val_idx)

        elapsed = time.perf_counter() - start

        print(
            f"[ElasticNet fold {fold_id}/{n_folds}] done in {elapsed:,.1f}s | "
            f"alpha={best['alpha']}, l1={best['l1_ratio']}, "
            f"train_rmse={train_rmse:,.3f}, val_rmse={val_rmse:,.3f}, "
            f"overfit={overfit_ratio:,.2f}, clean={clean_fold}",
            flush=True,
        )

        coefs = np.asarray(model.named_steps["model"].coef_, dtype=float)

        if len(coefs) != len(kept_features):
            raise RuntimeError(
                f"Coefficient length mismatch: len(coefs)={len(coefs)}, "
                f"len(kept_features)={len(kept_features)}."
            )

        coef_df = pd.DataFrame(
            {
                "fold": fold_id,
                "feature": kept_features,
                "coefficient": coefs,
                "abs_coefficient": np.abs(coefs),
                "selected_in_fold": np.abs(coefs) > self.coefficient_epsilon,
                "clean_fold": clean_fold,
                "selected_in_clean_fold": (np.abs(coefs) > self.coefficient_epsilon) & clean_fold,
                "alpha": best["alpha"],
                "l1_ratio": best["l1_ratio"],
                "fold_all_missing": [c in fold_all_missing_features for c in kept_features],
            }
        )

        metrics = {
            "fold": fold_id,
            "train_start": train_start,
            "train_end": train_end,
            "val_start": val_start,
            "val_end": val_end,
            "train_observed_utc_days": train_gap_diag["observed_utc_days"],
            "train_max_gap_days": train_gap_diag["max_gap_days"],
            "train_mean_gap_days": train_gap_diag["mean_gap_days"],
            "val_observed_utc_days": val_gap_diag["observed_utc_days"],
            "val_max_gap_days": val_gap_diag["max_gap_days"],
            "val_mean_gap_days": val_gap_diag["mean_gap_days"],
            "n_train": len(train_idx),
            "n_train_used": len(sampled_train_idx),
            "n_val": len(val_idx),
            "train_rmse": train_rmse,
            "val_rmse": val_rmse,
            "train_mae": mae(y_train, pred_train),
            "val_mae": mae(y_val, pred_val),
            "overfit_ratio_rmse": overfit_ratio,
            "clean_fold_for_selection": clean_fold,
            "alpha": best["alpha"],
            "l1_ratio": best["l1_ratio"],
            "best_model_convergence_warning_count": best["convergence_warning_count"],
            "fold_all_missing_feature_count": len(fold_all_missing_features),
            "fold_all_missing_features": "|".join(fold_all_missing_features[:50]),
            "elapsed_seconds": elapsed,
        }

        return coef_df, metrics

    def select(self, df: pd.DataFrame) -> FeatureSelectionResult:
        run_start = time.perf_counter()
        df = df.loc[:, ~df.columns.duplicated()].copy()

        work = df.dropna(subset=[self.target_column]).copy()
        work = work.sort_values("timestamp_utc").reset_index(drop=True)

        folds = make_walk_forward_folds(
            work,
            train_window_days=self.train_window_days,
            val_days=self.val_days,
            gap_days=self.gap_days,
            step_days=self.step_days,
            window_mode=self.window_mode,
            max_folds=self.max_folds,
        )

        if not folds:
            raise ValueError(
                no_folds_message(
                    work=work,
                    train_window_days=self.train_window_days,
                    val_days=self.val_days,
                    gap_days=self.gap_days,
                    step_days=self.step_days,
                    window_mode=self.window_mode,
                )
            )

        kept_features, prefilter_summary = prefilter_feature_columns(
            work,
            self.feature_columns,
            min_non_missing_fraction=self.min_non_missing_fraction,
        )

        kept_features, corr_summary = prune_correlated_features_by_pool(
            df=work,
            features=kept_features,
            target_column=self.target_column,
            corr_threshold=self.corr_threshold,
            random_state=self.random_state,
        )

        if not kept_features:
            raise ValueError("No usable features after prefiltering and pool-wise collinearity pruning.")

        n_grid = len(self.alpha_grid) * len(self.l1_ratio_grid)

        print(f"ElasticNet prefilter + pool corr prune: kept {len(kept_features)} / {len(self.feature_columns)} features.")
        print("ElasticNet backend: threads")
        print(f"ElasticNet CPU count visible to Python: {os.cpu_count()}")
        print(f"ElasticNet n_jobs requested: {self.n_jobs}")
        print(f"ElasticNet folds: {len(folds)}")
        print(f"ElasticNet grid size per fold: {n_grid}")
        print(f"ElasticNet total fits: {len(folds) * n_grid}")
        print(f"ElasticNet max_train_rows_per_fold: {self.max_train_rows_per_fold}")
        print(f"ElasticNet max_allowed_overfit_ratio: {self.max_allowed_overfit_ratio}")

        results = Parallel(
            n_jobs=self.n_jobs,
            prefer="threads",
            verbose=10,
        )(
            delayed(self.fit_one_fold)(
                fold_id,
                len(folds),
                train_idx,
                val_idx,
                work,
                kept_features,
            )
            for fold_id, (train_idx, val_idx) in enumerate(folds, start=1)
        )

        coef_df = pd.concat([r[0] for r in results], ignore_index=True)
        metrics_df = pd.DataFrame([r[1] for r in results]).sort_values("fold")

        clean_folds = int(metrics_df["clean_fold_for_selection"].sum())
        if clean_folds == 0:
            print("Warning: no clean folds passed overfit check. Falling back to all folds for selection.")
            coef_df["selected_in_clean_fold"] = coef_df["selected_in_fold"]
            clean_folds = len(metrics_df)

        summary = (
            coef_df.groupby("feature")
            .agg(
                selection_frequency=("selected_in_fold", "mean"),
                clean_selection_frequency=("selected_in_clean_fold", "mean"),
                selected_folds=("selected_in_fold", "sum"),
                selected_clean_folds=("selected_in_clean_fold", "sum"),
                mean_abs_coefficient=("abs_coefficient", "mean"),
                median_abs_coefficient=("abs_coefficient", "median"),
                max_abs_coefficient=("abs_coefficient", "max"),
                fold_all_missing_frequency=("fold_all_missing", "mean"),
                coefficient_sign_stability=(
                    "coefficient",
                    lambda x: np.nan if (x == 0).all() else abs(np.sign(x[x != 0]).mean()),
                ),
            )
            .reset_index()
        )

        summary["selected_pre_cap"] = (
            (summary["clean_selection_frequency"] >= self.min_selection_frequency)
            & (summary["median_abs_coefficient"] > self.coefficient_epsilon)
            & (summary["fold_all_missing_frequency"] < 0.50)
        )
        summary["selected_pre_cap"] = summary["selected_pre_cap"].fillna(False).astype(bool)

        summary = summary.merge(prefilter_summary, on="feature", how="outer")
        summary = summary.merge(corr_summary, on="feature", how="left")

        summary["family"] = summary["feature"].map(feature_family)
        summary["pool"] = summary["feature"].map(feature_pool)
        summary["base_signal"] = summary["feature"].map(feature_base_signal)
        summary["selected_pre_cap"] = summary["selected_pre_cap"].fillna(False).astype(bool)

        selected_features = apply_conservative_final_selection(
            summary=summary,
            score_col="median_abs_coefficient",
            max_total_features=self.max_total_features,
            max_features_per_family=self.max_features_per_family,
            max_features_per_base_signal=4,
        )

        selection_method = "elasticnet_coefficients"

        if not selected_features:
            print(
                "ElasticNet selected zero coefficient-based features. "
                "Using conservative univariate fallback.",
                flush=True,
            )
            selected_features = fallback_univariate_final_selection(
                summary=summary,
                max_total_features=min(self.max_total_features, 12),
                max_features_per_family=min(self.max_features_per_family, 3),
                min_non_missing_fraction=max(0.20, self.min_non_missing_fraction),
            )
            selection_method = "univariate_fallback_after_zero_elasticnet_coefficients"

        summary["selected"] = summary["feature"].isin(selected_features)
        summary["selection_method"] = selection_method

        summary = summary.sort_values(
            ["selected", "clean_selection_frequency", "median_abs_coefficient"],
            ascending=[False, False, False],
        ).reset_index(drop=True)

        total_elapsed = time.perf_counter() - run_start

        print(f"ElasticNet selector finished in {total_elapsed / 60:,.2f} minutes.")
        print(f"ElasticNet clean folds used for selection: {clean_folds}/{len(metrics_df)}")
        print(f"ElasticNet selected features after caps: {len(selected_features)}")

        return FeatureSelectionResult(
            model_type="elasticnet",
            target_name=self.target_name,
            target_column=self.target_column,
            selected_features=selected_features,
            feature_summary=summary,
            fold_metrics=metrics_df,
            config={
                "selection_method": selection_method,
                "alpha_grid": self.alpha_grid,
                "l1_ratio_grid": self.l1_ratio_grid,
                "grid_size_per_fold": n_grid,
                "total_fits": len(folds) * n_grid,
                "train_window_days": self.train_window_days,
                "gap_days": self.gap_days,
                "val_days": self.val_days,
                "step_days": self.step_days,
                "window_mode": self.window_mode,
                "fold_date_basis": "timestamp_utc",
                "rolling_alias": "rolling_gap_aware",
                "max_folds": self.max_folds,
                "min_selection_frequency": self.min_selection_frequency,
                "coefficient_epsilon": self.coefficient_epsilon,
                "min_non_missing_fraction": self.min_non_missing_fraction,
                "max_train_rows_per_fold": self.max_train_rows_per_fold,
                "fast_mode": self.fast_mode,
                "n_jobs": self.n_jobs,
                "parallel_backend": "threads",
                "corr_threshold": self.corr_threshold,
                "corr_pruning": "within_feature_pool_no_forced_pool_quotas",
                "max_total_features": self.max_total_features,
                "max_features_per_family": self.max_features_per_family,
                "max_allowed_overfit_ratio": self.max_allowed_overfit_ratio,
                "clean_folds_used_for_selection": clean_folds,
                "n_features_before_prefilter": len(self.feature_columns),
                "n_features_after_prefilter_and_corr_prune": len(kept_features),
                "elapsed_seconds": total_elapsed,
            },
        )


class XGBoostStabilitySelector:
    def __init__(
        self,
        target_name: str,
        target_column: str,
        feature_columns: list[str],
        train_window_days: int = 365,
        val_days: int = 30,
        gap_days: int = 1,
        step_days: int = 30,
        window_mode: str = "expanding",
        max_folds: int | None = None,
        min_selection_frequency: float = 0.65,
        cumulative_importance_cutoff: float = 0.75,
        max_features_per_fold: int = 20,
        min_non_missing_fraction: float = 0.10,
        random_state: int = 42,
        xgb_params: dict[str, Any] | None = None,
        n_jobs: int = 2,
        corr_threshold: float = 0.92,
        max_total_features: int = 25,
        max_features_per_family: int = 4,
        max_allowed_overfit_ratio: float = 1.50,
    ):
        self.target_name = target_name
        self.target_column = target_column
        self.feature_columns = dedupe(feature_columns)

        self.train_window_days = train_window_days
        self.val_days = val_days
        self.gap_days = gap_days
        self.step_days = step_days
        self.window_mode = window_mode
        self.max_folds = max_folds

        self.min_selection_frequency = min_selection_frequency
        self.cumulative_importance_cutoff = cumulative_importance_cutoff
        self.max_features_per_fold = max_features_per_fold
        self.min_non_missing_fraction = min_non_missing_fraction
        self.random_state = random_state
        self.n_jobs = n_jobs

        self.corr_threshold = corr_threshold
        self.max_total_features = max_total_features
        self.max_features_per_family = max_features_per_family
        self.max_allowed_overfit_ratio = max_allowed_overfit_ratio

        self.xgb_params = xgb_params or {
            "n_estimators": 250,
            "max_depth": 2,
            "learning_rate": 0.025,
            "subsample": 0.70,
            "colsample_bytree": 0.55,
            "min_child_weight": 50,
            "reg_lambda": 80.0,
            "reg_alpha": 10.0,
            "gamma": 1.0,
            "objective": "reg:squarederror",
            "tree_method": "hist",
            "random_state": random_state,
            "n_jobs": 1,
        }

    def make_model(self):
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:
            raise ImportError("xgboost is required. Install requirements-ml.txt.") from exc

        return XGBRegressor(**self.xgb_params)

    def fit_one_fold(
        self,
        fold_id: int,
        n_folds: int,
        train_idx: np.ndarray,
        val_idx: np.ndarray,
        work: pd.DataFrame,
        kept_features: list[str],
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        start = time.perf_counter()

        X = work[kept_features]
        y = work[self.target_column].astype(float)

        X_train = X.iloc[train_idx]
        X_val = X.iloc[val_idx]
        y_train = y.iloc[train_idx]
        y_val = y.iloc[val_idx]

        print(
            f"[XGBoost fold {fold_id}/{n_folds}] "
            f"train_rows={len(train_idx):,}, val_rows={len(val_idx):,}, "
            f"features={len(kept_features)}",
            flush=True,
        )

        model = self.make_model()
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        pred_train = model.predict(X_train)
        pred_val = model.predict(X_val)

        train_rmse = rmse(y_train, pred_train)
        val_rmse = rmse(y_val, pred_val)
        overfit_ratio = float(val_rmse / train_rmse) if train_rmse > 0 else np.nan
        clean_fold = bool(pd.notna(overfit_ratio) and overfit_ratio <= self.max_allowed_overfit_ratio)

        train_start, train_end = fold_bounds(work, train_idx)
        val_start, val_end = fold_bounds(work, val_idx)

        train_gap_diag = fold_gap_diagnostics(work, train_idx)
        val_gap_diag = fold_gap_diagnostics(work, val_idx)

        elapsed = time.perf_counter() - start

        print(
            f"[XGBoost fold {fold_id}/{n_folds}] done in {elapsed:,.1f}s | "
            f"train_rmse={train_rmse:,.3f}, val_rmse={val_rmse:,.3f}, "
            f"overfit={overfit_ratio:,.2f}, clean={clean_fold}",
            flush=True,
        )

        importances = np.asarray(model.feature_importances_, dtype=float)
        total = importances.sum()
        normalized = importances / total if total > 0 else np.zeros_like(importances)

        imp_df = pd.DataFrame(
            {
                "fold": fold_id,
                "feature": kept_features,
                "importance": importances,
                "normalized_importance": normalized,
            }
        ).sort_values("normalized_importance", ascending=False)

        imp_df["cumulative_importance"] = imp_df["normalized_importance"].cumsum()
        imp_df["rank"] = np.arange(1, len(imp_df) + 1)

        imp_df["selected_in_fold"] = (
            (
                (imp_df["cumulative_importance"] <= self.cumulative_importance_cutoff)
                | (imp_df["rank"] == 1)
            )
            & (imp_df["rank"] <= self.max_features_per_fold)
            & (imp_df["normalized_importance"] > 0)
        )

        imp_df["clean_fold"] = clean_fold
        imp_df["selected_in_clean_fold"] = imp_df["selected_in_fold"] & clean_fold

        metrics = {
            "fold": fold_id,
            "train_start": train_start,
            "train_end": train_end,
            "val_start": val_start,
            "val_end": val_end,
            "train_observed_utc_days": train_gap_diag["observed_utc_days"],
            "train_max_gap_days": train_gap_diag["max_gap_days"],
            "train_mean_gap_days": train_gap_diag["mean_gap_days"],
            "val_observed_utc_days": val_gap_diag["observed_utc_days"],
            "val_max_gap_days": val_gap_diag["max_gap_days"],
            "val_mean_gap_days": val_gap_diag["mean_gap_days"],
            "n_train": len(train_idx),
            "n_val": len(val_idx),
            "train_rmse": train_rmse,
            "val_rmse": val_rmse,
            "train_mae": mae(y_train, pred_train),
            "val_mae": mae(y_val, pred_val),
            "overfit_ratio_rmse": overfit_ratio,
            "clean_fold_for_selection": clean_fold,
            "elapsed_seconds": elapsed,
        }

        return imp_df, metrics

    def select(self, df: pd.DataFrame) -> FeatureSelectionResult:
        run_start = time.perf_counter()
        df = df.loc[:, ~df.columns.duplicated()].copy()

        work = df.dropna(subset=[self.target_column]).copy()
        work = work.sort_values("timestamp_utc").reset_index(drop=True)

        folds = make_walk_forward_folds(
            work,
            train_window_days=self.train_window_days,
            val_days=self.val_days,
            gap_days=self.gap_days,
            step_days=self.step_days,
            window_mode=self.window_mode,
            max_folds=self.max_folds,
        )

        if not folds:
            raise ValueError(
                no_folds_message(
                    work=work,
                    train_window_days=self.train_window_days,
                    val_days=self.val_days,
                    gap_days=self.gap_days,
                    step_days=self.step_days,
                    window_mode=self.window_mode,
                )
            )

        kept_features, prefilter_summary = prefilter_feature_columns(
            work,
            self.feature_columns,
            min_non_missing_fraction=self.min_non_missing_fraction,
        )

        kept_features, corr_summary = prune_correlated_features_by_pool(
            df=work,
            features=kept_features,
            target_column=self.target_column,
            corr_threshold=self.corr_threshold,
            random_state=self.random_state,
        )

        if not kept_features:
            raise ValueError("No usable features after prefiltering and pool-wise collinearity pruning.")

        print(f"XGBoost prefilter + pool corr prune: kept {len(kept_features)} / {len(self.feature_columns)} features.")
        print("XGBoost backend: threads")
        print(f"XGBoost CPU count visible to Python: {os.cpu_count()}")
        print(f"XGBoost n_jobs requested: {self.n_jobs}")
        print(f"XGBoost folds: {len(folds)}")
        print(f"XGBoost internal model n_jobs: {self.xgb_params.get('n_jobs')}")
        print(f"XGBoost max_allowed_overfit_ratio: {self.max_allowed_overfit_ratio}")

        results = Parallel(n_jobs=self.n_jobs, prefer="threads", verbose=10)(
            delayed(self.fit_one_fold)(
                fold_id,
                len(folds),
                train_idx,
                val_idx,
                work,
                kept_features,
            )
            for fold_id, (train_idx, val_idx) in enumerate(folds, start=1)
        )

        imp_df = pd.concat([r[0] for r in results], ignore_index=True)
        metrics_df = pd.DataFrame([r[1] for r in results]).sort_values("fold")

        clean_folds = int(metrics_df["clean_fold_for_selection"].sum())
        if clean_folds == 0:
            print("Warning: no clean folds passed overfit check. Falling back to all folds for selection.")
            imp_df["selected_in_clean_fold"] = imp_df["selected_in_fold"]
            clean_folds = len(metrics_df)

        summary = (
            imp_df.groupby("feature")
            .agg(
                selection_frequency=("selected_in_fold", "mean"),
                clean_selection_frequency=("selected_in_clean_fold", "mean"),
                selected_folds=("selected_in_fold", "sum"),
                selected_clean_folds=("selected_in_clean_fold", "sum"),
                mean_importance=("normalized_importance", "mean"),
                median_importance=("normalized_importance", "median"),
                max_importance=("normalized_importance", "max"),
                mean_rank=("rank", "mean"),
                best_rank=("rank", "min"),
            )
            .reset_index()
        )

        summary["selected_pre_cap"] = (
            (summary["clean_selection_frequency"] >= self.min_selection_frequency)
            & (summary["mean_importance"] > 0)
        )
        summary["selected_pre_cap"] = summary["selected_pre_cap"].fillna(False).astype(bool)

        summary = summary.merge(prefilter_summary, on="feature", how="outer")
        summary = summary.merge(corr_summary, on="feature", how="left")

        summary["family"] = summary["feature"].map(feature_family)
        summary["pool"] = summary["feature"].map(feature_pool)
        summary["base_signal"] = summary["feature"].map(feature_base_signal)
        summary["selected_pre_cap"] = summary["selected_pre_cap"].fillna(False).astype(bool)

        selected_features = apply_conservative_final_selection(
            summary=summary,
            score_col="mean_importance",
            max_total_features=self.max_total_features,
            max_features_per_family=self.max_features_per_family,
            max_features_per_base_signal=4,
        )

        summary["selected"] = summary["feature"].isin(selected_features)

        summary = summary.sort_values(
            ["selected", "clean_selection_frequency", "mean_importance", "best_rank"],
            ascending=[False, False, False, True],
        ).reset_index(drop=True)

        total_elapsed = time.perf_counter() - run_start

        print(f"XGBoost selector finished in {total_elapsed / 60:,.2f} minutes.")
        print(f"XGBoost clean folds used for selection: {clean_folds}/{len(metrics_df)}")
        print(f"XGBoost selected features after caps: {len(selected_features)}")

        return FeatureSelectionResult(
            model_type="xgboost",
            target_name=self.target_name,
            target_column=self.target_column,
            selected_features=selected_features,
            feature_summary=summary,
            fold_metrics=metrics_df,
            config={
                "xgb_params": self.xgb_params,
                "train_window_days": self.train_window_days,
                "gap_days": self.gap_days,
                "val_days": self.val_days,
                "step_days": self.step_days,
                "window_mode": self.window_mode,
                "fold_date_basis": "timestamp_utc",
                "rolling_alias": "rolling_gap_aware",
                "max_folds": self.max_folds,
                "min_selection_frequency": self.min_selection_frequency,
                "cumulative_importance_cutoff": self.cumulative_importance_cutoff,
                "max_features_per_fold": self.max_features_per_fold,
                "min_non_missing_fraction": self.min_non_missing_fraction,
                "n_jobs": self.n_jobs,
                "parallel_backend": "threads",
                "corr_threshold": self.corr_threshold,
                "corr_pruning": "within_feature_pool_no_forced_pool_quotas",
                "max_total_features": self.max_total_features,
                "max_features_per_family": self.max_features_per_family,
                "max_allowed_overfit_ratio": self.max_allowed_overfit_ratio,
                "clean_folds_used_for_selection": clean_folds,
                "n_features_before_prefilter": len(self.feature_columns),
                "n_features_after_prefilter_and_corr_prune": len(kept_features),
                "elapsed_seconds": total_elapsed,
            },
        )
