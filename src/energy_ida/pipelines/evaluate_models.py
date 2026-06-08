from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits

from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from energy_ida.config import MASTER_FILE
from energy_ida.features.build_features import build_feature_dataset, list_available_targets
from energy_ida.features.target_config import get_target_spec
from energy_ida.model_selection.feature_selectors import (
    auto_walk_forward_params,
    make_walk_forward_folds,
)


def _valid_y_pred(y_true, y_pred) -> pd.DataFrame:
    tmp = pd.DataFrame(
        {
            "y": pd.Series(y_true).astype(float).to_numpy(),
            "pred": pd.Series(y_pred).astype(float).to_numpy(),
        }
    )
    return tmp.replace([np.inf, -np.inf], np.nan).dropna()


def rmse(y_true, y_pred) -> float:
    tmp = _valid_y_pred(y_true, y_pred)
    if tmp.empty:
        return float("nan")
    return float(mean_squared_error(tmp["y"], tmp["pred"]) ** 0.5)


def mae(y_true, y_pred) -> float:
    tmp = _valid_y_pred(y_true, y_pred)
    if tmp.empty:
        return float("nan")
    return float(mean_absolute_error(tmp["y"], tmp["pred"]))


def n_valid_pairs(y_true, y_pred) -> int:
    return int(len(_valid_y_pred(y_true, y_pred)))


def safe_corr(y_true, y_pred, method: str = "pearson") -> float:
    tmp = _valid_y_pred(y_true, y_pred)

    if len(tmp) < 3:
        return float("nan")

    if tmp["y"].nunique() < 2 or tmp["pred"].nunique() < 2:
        return float("nan")

    return float(tmp["y"].corr(tmp["pred"], method=method))


def directional_accuracy(y_true, y_pred) -> float:
    tmp = _valid_y_pred(y_true, y_pred)
    if tmp.empty:
        return float("nan")
    return float((np.sign(tmp["y"]) == np.sign(tmp["pred"])).mean())


def signal_hit_rate(y_true, y_pred, threshold: float) -> float:
    tmp = _valid_y_pred(y_true, y_pred)
    tmp = tmp[tmp["pred"].abs() > threshold]

    if tmp.empty:
        return float("nan")

    return float((np.sign(tmp["y"]) == np.sign(tmp["pred"])).mean())


def simple_threshold_pnl(
    realized_spread,
    predicted_signal,
    threshold: float,
    transaction_cost_eur_mwh: float = 0.0,
) -> dict[str, float]:
    tmp = _valid_y_pred(realized_spread, predicted_signal)

    if tmp.empty:
        return {
            "pnl_total": float("nan"),
            "pnl_mean_per_slot": float("nan"),
            "trade_count": 0,
            "trade_fraction": float("nan"),
            "hit_rate_traded": float("nan"),
        }

    pos = np.where(
        tmp["pred"] > threshold,
        1.0,
        np.where(tmp["pred"] < -threshold, -1.0, 0.0),
    )

    pnl = pos * tmp["y"].to_numpy() - transaction_cost_eur_mwh * np.abs(pos)
    traded = np.abs(pos) > 0

    if traded.sum() == 0:
        hit_rate = float("nan")
    else:
        hit_rate = float(
            (np.sign(tmp["y"].to_numpy()[traded]) == np.sign(pos[traded])).mean()
        )

    return {
        "pnl_total": float(np.nansum(pnl)),
        "pnl_mean_per_slot": float(np.nanmean(pnl)),
        "trade_count": int(traded.sum()),
        "trade_fraction": float(traded.mean()),
        "hit_rate_traded": hit_rate,
    }


def choose_threshold_from_train_predictions(
    pred_train_signal,
    threshold_mode: str,
    manual_threshold: float,
    threshold_quantile: float,
) -> float:
    pred = pd.Series(pred_train_signal).replace([np.inf, -np.inf], np.nan).dropna().abs()

    if threshold_mode == "manual":
        return float(manual_threshold)

    if threshold_mode == "train_quantile":
        if pred.empty:
            return float(manual_threshold)
        return float(pred.quantile(threshold_quantile))

    raise ValueError("threshold_mode must be 'manual' or 'train_quantile'.")


def load_selected_features(path: Path) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    features = obj.get("selected_features", [])

    if not isinstance(features, list):
        raise ValueError(f"Invalid selected_features in {path}")

    return list(dict.fromkeys(features))


def load_xgb_params(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None

    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    params = obj.get("best_params", obj)

    if not isinstance(params, dict):
        raise ValueError(f"Invalid XGBoost params JSON: {path}")

    params = dict(params)
    params["objective"] = "reg:squarederror"
    params["tree_method"] = params.get("tree_method", "hist")
    params["random_state"] = int(params.get("random_state", 42))
    params["n_jobs"] = int(params.get("n_jobs", 1))

    return params


def get_seasonal_naive_candidates(target_column: str, feature_columns: list[str]) -> list[str]:
    candidates = [
        f"{target_column}_lag_1d_same_local_qh",
        f"{target_column}_lag_7d_same_local_qh",
        f"{target_column}_rolling_median_7d_same_local_qh",
        f"{target_column}_rolling_mean_7d_same_local_qh",
        f"{target_column}_rolling_median_14d_same_local_qh",
        f"{target_column}_rolling_mean_14d_same_local_qh",
        f"{target_column}_rolling_median_28d_same_local_qh",
        f"{target_column}_rolling_mean_28d_same_local_qh",
    ]
    return [c for c in candidates if c in feature_columns]


def make_safe_imputer() -> SimpleImputer:
    return SimpleImputer(strategy="median", keep_empty_features=True)


def make_elasticnet_model(alpha: float, l1_ratio: float) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", make_safe_imputer()),
            ("scaler", StandardScaler()),
            (
                "model",
                ElasticNet(
                    alpha=alpha,
                    l1_ratio=l1_ratio,
                    max_iter=100_000,
                    tol=1e-3,
                    selection="random",
                    random_state=42,
                ),
            ),
        ]
    )


def choose_elasticnet_params_inner_train(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    alpha_grid: list[float],
    l1_ratio_grid: list[float],
) -> dict[str, float]:
    n = len(X_train)

    if n < 200:
        return {"alpha": 10.0, "l1_ratio": 1.0}

    split = int(n * 0.80)

    X_inner_train = X_train.iloc[:split]
    y_inner_train = y_train.iloc[:split]
    X_inner_val = X_train.iloc[split:]
    y_inner_val = y_train.iloc[split:]

    best = None

    with threadpool_limits(limits=1):
        for alpha in alpha_grid:
            for l1_ratio in l1_ratio_grid:
                model = make_elasticnet_model(alpha=alpha, l1_ratio=l1_ratio)
                model.fit(X_inner_train, y_inner_train)

                pred = model.predict(X_inner_val)
                score = rmse(y_inner_val, pred)

                if best is None or score < best["rmse"]:
                    best = {
                        "alpha": alpha,
                        "l1_ratio": l1_ratio,
                        "rmse": score,
                    }

    return {"alpha": float(best["alpha"]), "l1_ratio": float(best["l1_ratio"])}


def default_xgboost_params() -> dict[str, Any]:
    return {
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
        "random_state": 42,
        "n_jobs": 1,
    }


def make_xgboost_model(params: dict[str, Any] | None = None):
    try:
        from xgboost import XGBRegressor
    except ImportError as exc:
        raise ImportError("xgboost is required. Install requirements-ml.txt.") from exc

    final_params = default_xgboost_params()
    if params:
        final_params.update(params)

    final_params["objective"] = "reg:squarederror"
    final_params["tree_method"] = final_params.get("tree_method", "hist")
    final_params["random_state"] = int(final_params.get("random_state", 42))
    final_params["n_jobs"] = int(final_params.get("n_jobs", 1))

    return XGBRegressor(**final_params)


def compute_signal_predictions(
    *,
    spec,
    df: pd.DataFrame,
    raw_prediction,
) -> pd.Series:
    raw_prediction = pd.Series(raw_prediction, index=df.index).astype(float)

    if spec.target_kind == "spread":
        return raw_prediction

    if spec.target_kind == "price":
        if "da_price_eur_mwh" not in df.columns:
            raise ValueError("Price target trading evaluation requires da_price_eur_mwh.")
        return raw_prediction - df["da_price_eur_mwh"].astype(float)

    raise ValueError(f"Unknown target_kind={spec.target_kind}")


def get_realized_spread(spec, df: pd.DataFrame) -> pd.Series:
    if spec.spread_column in df.columns:
        return df[spec.spread_column].astype(float)

    if spec.price_column in df.columns and "da_price_eur_mwh" in df.columns:
        return df[spec.price_column].astype(float) - df["da_price_eur_mwh"].astype(float)

    raise ValueError(
        f"Cannot compute realized spread for {spec.name}. "
        f"Missing {spec.spread_column} or price/DA columns."
    )


def fold_metrics_row(
    *,
    model_name: str,
    fold_id: int,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    y_train,
    pred_train_raw,
    y_val,
    pred_val_raw,
    val_realized_spread,
    pred_train_signal,
    pred_val_signal,
    threshold: float,
    threshold_mode: str,
    threshold_quantile: float,
    transaction_cost: float,
    elapsed_seconds: float,
    target_kind: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    train_rmse = rmse(y_train, pred_train_raw)
    val_rmse = rmse(y_val, pred_val_raw)

    train_mae = mae(y_train, pred_train_raw)
    val_mae = mae(y_val, pred_val_raw)

    pnl = simple_threshold_pnl(
        realized_spread=val_realized_spread,
        predicted_signal=pred_val_signal,
        threshold=threshold,
        transaction_cost_eur_mwh=transaction_cost,
    )

    row = {
        "model": model_name,
        "fold": fold_id,
        "target_kind": target_kind,
        "train_start": str(train_df["timestamp_utc"].min()),
        "train_end": str(train_df["timestamp_utc"].max()),
        "val_start": str(val_df["timestamp_utc"].min()),
        "val_end": str(val_df["timestamp_utc"].max()),
        "n_train": len(train_df),
        "n_val": len(val_df),
        "n_train_scored": n_valid_pairs(y_train, pred_train_raw),
        "n_val_scored": n_valid_pairs(y_val, pred_val_raw),
        "train_rmse": train_rmse,
        "val_rmse": val_rmse,
        "train_mae": train_mae,
        "val_mae": val_mae,
        "overfit_ratio_rmse": float(val_rmse / train_rmse)
        if pd.notna(train_rmse) and train_rmse > 0 else np.nan,
        "val_pearson": safe_corr(y_val, pred_val_raw, method="pearson"),
        "val_spearman": safe_corr(y_val, pred_val_raw, method="spearman"),
        "val_directional_accuracy": directional_accuracy(val_realized_spread, pred_val_signal),
        "val_signal_pearson": safe_corr(val_realized_spread, pred_val_signal, method="pearson"),
        "val_signal_spearman": safe_corr(val_realized_spread, pred_val_signal, method="spearman"),
        "val_hit_rate_abs_signal_gt_threshold": signal_hit_rate(
            val_realized_spread,
            pred_val_signal,
            threshold=threshold,
        ),
        "threshold": threshold,
        "threshold_mode": threshold_mode,
        "threshold_quantile": threshold_quantile,
        "transaction_cost_eur_mwh": transaction_cost,
        "elapsed_seconds": elapsed_seconds,
        **pnl,
    }

    if extra:
        row.update(extra)

    return row


def evaluate_one_fold(
    fold_id: int,
    n_folds: int,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    model_name: str,
    data: pd.DataFrame,
    target_column: str,
    spec,
    used_features: list[str],
    threshold_mode: str,
    manual_threshold: float,
    threshold_quantile: float,
    transaction_cost: float,
    xgb_params: dict[str, Any] | None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    start = time.perf_counter()

    train_df = data.iloc[train_idx].copy()
    val_df = data.iloc[val_idx].copy()

    y_train = train_df[target_column].astype(float)
    y_val = val_df[target_column].astype(float)

    print(
        f"[Evaluation {model_name} fold {fold_id}/{n_folds}] "
        f"train_rows={len(train_df):,}, val_rows={len(val_df):,}, "
        f"features={len(used_features)}",
        flush=True,
    )

    if model_name == "seasonal_naive":
        best_feature = None
        best_train_rmse = None

        for c in used_features:
            train_candidate = train_df[c].astype(float)
            valid_n = n_valid_pairs(y_train, train_candidate)

            if valid_n < 100:
                continue

            score = rmse(y_train, train_candidate)

            if best_train_rmse is None or score < best_train_rmse:
                best_train_rmse = score
                best_feature = c

        if best_feature is None:
            raise ValueError(f"No valid seasonal naive feature in fold {fold_id}.")

        pred_train_raw = train_df[best_feature].astype(float)
        pred_val_raw = val_df[best_feature].astype(float)

        extra = {
            "seasonal_naive_feature": best_feature,
            "seasonal_naive_candidates": "|".join(used_features),
        }

    elif model_name == "elasticnet":
        X_train = train_df[used_features]
        X_val = val_df[used_features]

        params = choose_elasticnet_params_inner_train(
            X_train=X_train,
            y_train=y_train,
            alpha_grid=[0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0],
            l1_ratio_grid=[0.5, 0.7, 0.9, 1.0],
        )

        model = make_elasticnet_model(
            alpha=params["alpha"],
            l1_ratio=params["l1_ratio"],
        )

        with threadpool_limits(limits=1):
            model.fit(X_train, y_train)

        pred_train_raw = pd.Series(model.predict(X_train), index=train_df.index)
        pred_val_raw = pd.Series(model.predict(X_val), index=val_df.index)

        extra = params | {"n_features": len(used_features)}

    elif model_name == "xgboost":
        X_train = train_df[used_features]
        X_val = val_df[used_features]

        model = make_xgboost_model(xgb_params)
        model.fit(X_train, y_train, verbose=False)

        pred_train_raw = pd.Series(model.predict(X_train), index=train_df.index)
        pred_val_raw = pd.Series(model.predict(X_val), index=val_df.index)

        extra = {
            "n_features": len(used_features),
            "xgb_params_source": "custom" if xgb_params else "default",
        }

    else:
        raise ValueError(f"Unknown model: {model_name}")

    train_realized_spread = get_realized_spread(spec, train_df)
    val_realized_spread = get_realized_spread(spec, val_df)

    pred_train_signal = compute_signal_predictions(
        spec=spec,
        df=train_df,
        raw_prediction=pred_train_raw,
    )
    pred_val_signal = compute_signal_predictions(
        spec=spec,
        df=val_df,
        raw_prediction=pred_val_raw,
    )

    threshold = choose_threshold_from_train_predictions(
        pred_train_signal=pred_train_signal,
        threshold_mode=threshold_mode,
        manual_threshold=manual_threshold,
        threshold_quantile=threshold_quantile,
    )

    elapsed = time.perf_counter() - start

    row = fold_metrics_row(
        model_name=model_name,
        fold_id=fold_id,
        train_df=train_df,
        val_df=val_df,
        y_train=y_train,
        pred_train_raw=pred_train_raw,
        y_val=y_val,
        pred_val_raw=pred_val_raw,
        val_realized_spread=val_realized_spread,
        pred_train_signal=pred_train_signal,
        pred_val_signal=pred_val_signal,
        threshold=threshold,
        threshold_mode=threshold_mode,
        threshold_quantile=threshold_quantile,
        transaction_cost=transaction_cost,
        elapsed_seconds=elapsed,
        target_kind=spec.target_kind,
        extra=extra,
    )

    print(
        f"[Evaluation {model_name} fold {fold_id}/{n_folds}] done in {elapsed:,.1f}s | "
        f"train_rmse={row['train_rmse']:,.3f}, val_rmse={row['val_rmse']:,.3f}, "
        f"overfit={row['overfit_ratio_rmse']:,.2f}, pnl={row['pnl_total']:,.2f}",
        flush=True,
    )

    pred_part_cols = [
        "timestamp_utc",
        "timestamp_local",
        "date_local",
        "hour_local",
        "quarter_hour_local",
        target_column,
        spec.price_column,
        spec.spread_column,
        "da_price_eur_mwh",
    ]
    
    # Important: for spread targets, target_column == spec.spread_column.
    # Dedupe before selecting from val_df, otherwise pred_part[target_column]
    # can return a DataFrame with duplicate labels instead of a Series.
    pred_part_cols = list(dict.fromkeys([c for c in pred_part_cols if c in val_df.columns]))
    
    pred_part = val_df[pred_part_cols].copy()
    pred_part = pred_part.loc[:, ~pred_part.columns.duplicated()].copy()
    
    target_actual = val_df[target_column].astype(float)
    raw_pred_series = pd.Series(pred_val_raw, index=val_df.index).astype(float)
    signal_pred_series = pd.Series(pred_val_signal, index=val_df.index).astype(float)
    realized_spread_series = pd.Series(val_realized_spread, index=val_df.index).astype(float)
    
    pred_part["model"] = model_name
    pred_part["fold"] = fold_id
    pred_part["target_kind"] = spec.target_kind
    pred_part["raw_prediction"] = raw_pred_series.to_numpy()
    pred_part["signal_prediction"] = signal_pred_series.to_numpy()
    pred_part["realized_spread"] = realized_spread_series.to_numpy()
    pred_part["threshold"] = threshold
    pred_part["target_error"] = raw_pred_series.to_numpy() - target_actual.to_numpy()
    pred_part["abs_target_error"] = pred_part["target_error"].abs()
    pred_part["signal_error"] = pred_part["signal_prediction"] - pred_part["realized_spread"]
    pred_part["abs_signal_error"] = pred_part["signal_error"].abs()
    pred_part["is_scored"] = pred_part[target_column].notna() & pred_part["raw_prediction"].notna()
    pred_part["position"] = np.where(
        pred_part["signal_prediction"] > threshold,
        1,
        np.where(pred_part["signal_prediction"] < -threshold, -1, 0),
    )
    pred_part["pnl"] = pred_part["position"] * pred_part["realized_spread"]
    pred_part["direction_correct"] = np.where(
        pred_part["position"] != 0,
        np.sign(pred_part["realized_spread"]) == np.sign(pred_part["position"]),
        pd.NA,
    )

    return row, pred_part


def evaluate_model(
    model_name: str,
    data: pd.DataFrame,
    target_column: str,
    spec,
    feature_columns: list[str],
    folds: list[tuple[np.ndarray, np.ndarray]],
    threshold_mode: str,
    manual_threshold: float,
    threshold_quantile: float,
    transaction_cost: float,
    selected_features_path: Path | None = None,
    xgb_params_path: Path | None = None,
    n_jobs: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = data.copy()
    data = data.loc[:, ~data.columns.duplicated()].copy()
    feature_columns = list(dict.fromkeys(feature_columns))

    if model_name == "seasonal_naive":
        used_features = get_seasonal_naive_candidates(target_column, feature_columns)

        if not used_features:
            raise ValueError("No seasonal naive candidate lag/rolling features found.")

    else:
        if selected_features_path is None:
            raise ValueError(f"{model_name} requires --selected-features-json")

        used_features = load_selected_features(selected_features_path)
        used_features = [c for c in used_features if c in data.columns]

        if not used_features:
            raise ValueError(f"No selected features found in dataframe for {model_name}.")

    xgb_params = load_xgb_params(xgb_params_path) if model_name == "xgboost" else None

    print("=" * 120)
    print("PARALLEL EVALUATION SETUP")
    print("=" * 120)
    print("Model:", model_name)
    print("Target kind:", spec.target_kind)
    print("Folds:", len(folds))
    print("Features used:", len(used_features))
    print("n_jobs:", n_jobs)
    print("Backend: threads")
    print("CPU count visible to Python:", os.cpu_count())
    print("XGBoost params:", xgb_params_path if xgb_params_path else "default")

    results = Parallel(n_jobs=n_jobs, prefer="threads", verbose=10)(
        delayed(evaluate_one_fold)(
            fold_id,
            len(folds),
            train_idx,
            val_idx,
            model_name,
            data,
            target_column,
            spec,
            used_features,
            threshold_mode,
            manual_threshold,
            threshold_quantile,
            transaction_cost,
            xgb_params,
        )
        for fold_id, (train_idx, val_idx) in enumerate(folds, start=1)
    )

    metrics = pd.DataFrame([r[0] for r in results]).sort_values("fold")
    predictions = pd.concat([r[1] for r in results], ignore_index=True)

    return metrics, predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate price/spread forecasting models with walk-forward backtest."
    )

    parser.add_argument("--target", choices=list_available_targets(), required=True)
    parser.add_argument("--model", choices=["seasonal_naive", "elasticnet", "xgboost"], required=True)

    parser.add_argument("--master-file", type=Path, default=MASTER_FILE)
    parser.add_argument("--selected-features-json", type=Path, default=None)
    parser.add_argument("--xgb-params-json", type=Path, default=None)

    parser.add_argument("--fold-mode", choices=["manual", "auto"], default="manual")
    parser.add_argument("--window-mode", choices=["expanding", "rolling", "rolling_gap_aware", "rolling_contiguous"], default="expanding")
    parser.add_argument("--train-window-days", type=int, default=365)
    parser.add_argument("--gap-days", type=int, default=1)
    parser.add_argument("--val-days", type=int, default=30)
    parser.add_argument("--step-days", type=int, default=30)
    parser.add_argument("--max-folds", type=int, default=None)

    parser.add_argument("--threshold-mode", choices=["manual", "train_quantile"], default="train_quantile")
    parser.add_argument("--threshold", type=float, default=5.0)
    parser.add_argument("--threshold-quantile", type=float, default=0.60)
    parser.add_argument("--transaction-cost", type=float, default=0.0)

    parser.add_argument("--n-jobs", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, default=Path("reports_local/model_evaluation"))

    return parser.parse_args()


def main() -> None:
    run_start = time.perf_counter()
    args = parse_args()

    spec = get_target_spec(args.target)

    build = build_feature_dataset(
        target_name=args.target,
        master_file=args.master_file,
    )

    data = build.data.copy()
    data = data.loc[:, ~data.columns.duplicated()].copy()

    active_days = pd.to_datetime(data["timestamp_utc"], utc=True).dt.date.nunique()
    active_local_days = pd.to_datetime(data["date_local"]).dt.date.nunique()

    if args.fold_mode == "auto":
        params = auto_walk_forward_params(active_days)
        train_window_days = int(params["train_window_days"])
        gap_days = int(params["gap_days"])
        val_days = int(params["val_days"])
        step_days = int(params["step_days"])
        window_mode = str(params["window_mode"])
        print("Auto fold parameters derived from active UTC target days:")
        print(params)
    else:
        train_window_days = args.train_window_days
        gap_days = args.gap_days
        val_days = args.val_days
        step_days = args.step_days
        window_mode = args.window_mode

    folds = make_walk_forward_folds(
        data,
        time_col="timestamp_utc",
        train_window_days=train_window_days,
        val_days=val_days,
        gap_days=gap_days,
        step_days=step_days,
        window_mode=window_mode,
        max_folds=args.max_folds,
    )

    if not folds:
        raise SystemExit("No valid folds. Use smaller train/validation windows.")

    metrics, predictions = evaluate_model(
        model_name=args.model,
        data=data,
        target_column=build.target_column_name,
        spec=spec,
        feature_columns=build.feature_columns,
        folds=folds,
        threshold_mode=args.threshold_mode,
        manual_threshold=args.threshold,
        threshold_quantile=args.threshold_quantile,
        transaction_cost=args.transaction_cost,
        selected_features_path=args.selected_features_json,
        xgb_params_path=args.xgb_params_json,
        n_jobs=args.n_jobs,
    )

    out_dir = args.output_dir / args.target / args.model
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = out_dir / "fold_metrics.csv"
    predictions_path = out_dir / "predictions.csv"
    summary_path = out_dir / "summary.csv"

    metrics.to_csv(metrics_path, index=False)
    predictions.to_csv(predictions_path, index=False)

    summary = (
        metrics.groupby("model")
        .agg(
            folds=("fold", "count"),
            mean_val_rmse=("val_rmse", "mean"),
            median_val_rmse=("val_rmse", "median"),
            mean_val_mae=("val_mae", "mean"),
            median_val_mae=("val_mae", "median"),
            mean_directional_accuracy=("val_directional_accuracy", "mean"),
            mean_pearson=("val_pearson", "mean"),
            mean_spearman=("val_spearman", "mean"),
            mean_signal_pearson=("val_signal_pearson", "mean"),
            mean_signal_spearman=("val_signal_spearman", "mean"),
            mean_overfit_ratio_rmse=("overfit_ratio_rmse", "mean"),
            total_pnl=("pnl_total", "sum"),
            mean_trade_fraction=("trade_fraction", "mean"),
            mean_hit_rate_traded=("hit_rate_traded", "mean"),
            mean_n_val_scored=("n_val_scored", "mean"),
            mean_threshold=("threshold", "mean"),
            total_elapsed_seconds=("elapsed_seconds", "sum"),
        )
        .reset_index()
    )

    summary.to_csv(summary_path, index=False)

    total_elapsed = time.perf_counter() - run_start

    print("=" * 120)
    print("MODEL EVALUATION")
    print("=" * 120)
    print("Target:", args.target)
    print("Target kind:", spec.target_kind)
    print("Model:", args.model)
    print("Rows:", len(data))
    print("Active UTC target days:", active_days)
    print("Active local target days:", active_local_days)
    print("Fold date basis: timestamp_utc")
    print("Features available:", len(build.feature_columns))
    print(
        "Fold setup:",
        {
            "window_mode": window_mode,
            "train_window_days": train_window_days,
            "gap_days": gap_days,
            "val_days": val_days,
            "step_days": step_days,
            "max_folds": args.max_folds,
            "fold_date_basis": "timestamp_utc",
        },
    )
    print(
        "Threshold setup:",
        {
            "threshold_mode": args.threshold_mode,
            "manual_threshold": args.threshold,
            "threshold_quantile": args.threshold_quantile,
        },
    )
    print("n_jobs:", args.n_jobs)
    print("Folds:", len(folds))
    print("Output:", out_dir)
    print(f"Total elapsed: {total_elapsed / 60:,.2f} minutes")

    print("\nFold metrics:")
    print(metrics.to_string(index=False))

    print("\nSummary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
