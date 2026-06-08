from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from energy_ida.config import MASTER_FILE
from energy_ida.features.build_features import build_feature_dataset, list_available_targets
from energy_ida.features.target_config import get_target_spec
from energy_ida.model_selection.feature_selectors import (
    auto_walk_forward_params,
    make_walk_forward_folds,
)


def rmse(y_true, y_pred) -> float:
    tmp = pd.DataFrame({"y": y_true, "pred": y_pred}).replace([np.inf, -np.inf], np.nan).dropna()
    if tmp.empty:
        return float("nan")
    return float(mean_squared_error(tmp["y"], tmp["pred"]) ** 0.5)


def mae(y_true, y_pred) -> float:
    tmp = pd.DataFrame({"y": y_true, "pred": y_pred}).replace([np.inf, -np.inf], np.nan).dropna()
    if tmp.empty:
        return float("nan")
    return float(mean_absolute_error(tmp["y"], tmp["pred"]))


def safe_corr(y_true, y_pred, method: str = "spearman") -> float:
    tmp = pd.DataFrame({"y": y_true, "pred": y_pred}).replace([np.inf, -np.inf], np.nan).dropna()

    if len(tmp) < 3:
        return float("nan")

    if tmp["y"].nunique() < 2 or tmp["pred"].nunique() < 2:
        return float("nan")

    return float(tmp["y"].corr(tmp["pred"], method=method))


def directional_accuracy(y_true, y_pred) -> float:
    tmp = pd.DataFrame({"y": y_true, "pred": y_pred}).replace([np.inf, -np.inf], np.nan).dropna()

    if tmp.empty:
        return float("nan")

    return float((np.sign(tmp["y"]) == np.sign(tmp["pred"])).mean())


def simple_threshold_pnl(realized_spread, signal_prediction, threshold: float) -> float:
    tmp = pd.DataFrame(
        {
            "spread": realized_spread,
            "signal": signal_prediction,
        }
    ).replace([np.inf, -np.inf], np.nan).dropna()

    if tmp.empty:
        return 0.0

    pos = np.where(
        tmp["signal"] > threshold,
        1.0,
        np.where(tmp["signal"] < -threshold, -1.0, 0.0),
    )

    return float(np.sum(pos * tmp["spread"].to_numpy()))


def load_selected_features(path: Path | None, data: pd.DataFrame, all_features: list[str]) -> list[str]:
    if path is None:
        return [c for c in all_features if c in data.columns]

    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    features = obj.get("selected_features", [])

    if not isinstance(features, list):
        raise ValueError(f"Invalid selected_features in {path}")

    features = list(dict.fromkeys(features))
    features = [c for c in features if c in data.columns]

    if not features:
        raise ValueError(f"No selected features from {path} are present in feature dataframe.")

    return features


def make_xgb_params(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 700, step=50),
        "max_depth": trial.suggest_int("max_depth", 1, 4),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.08, log=True),
        "subsample": trial.suggest_float("subsample", 0.50, 0.95),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.30, 0.90),
        "min_child_weight": trial.suggest_float("min_child_weight", 5.0, 250.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 500.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.001, 200.0, log=True),
        "gamma": trial.suggest_float("gamma", 0.0, 20.0),
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "random_state": 42,
        "n_jobs": 1,
    }


def make_signal(spec, df: pd.DataFrame, raw_prediction) -> pd.Series:
    raw_prediction = pd.Series(raw_prediction, index=df.index).astype(float)

    if spec.target_kind == "spread":
        return raw_prediction

    if spec.target_kind == "price":
        if "da_price_eur_mwh" not in df.columns:
            raise ValueError("Price target optimization requires da_price_eur_mwh.")
        return raw_prediction - df["da_price_eur_mwh"].astype(float)

    raise ValueError(f"Unknown target_kind={spec.target_kind}")


def realized_spread(spec, df: pd.DataFrame) -> pd.Series:
    if spec.spread_column in df.columns:
        return df[spec.spread_column].astype(float)

    if spec.price_column in df.columns and "da_price_eur_mwh" in df.columns:
        return df[spec.price_column].astype(float) - df["da_price_eur_mwh"].astype(float)

    raise ValueError(f"Cannot compute realized spread for {spec.name}")


def fit_predict_xgb(params: dict[str, Any], X_train, y_train, X_val):
    try:
        from xgboost import XGBRegressor
    except ImportError as exc:
        raise ImportError("xgboost is required. Install requirements-ml.txt.") from exc

    model = XGBRegressor(**params)
    model.fit(X_train, y_train, verbose=False)

    pred_train = pd.Series(model.predict(X_train), index=X_train.index)
    pred_val = pd.Series(model.predict(X_val), index=X_val.index)

    return pred_train, pred_val


def evaluate_trial(
    params: dict[str, Any],
    data: pd.DataFrame,
    spec,
    target_column: str,
    feature_columns: list[str],
    folds: list[tuple[np.ndarray, np.ndarray]],
    objective_metric: str,
    max_allowed_overfit_ratio: float,
    overfit_penalty_weight: float,
    threshold_quantile: float,
) -> dict[str, float]:
    rows = []

    for fold_id, (train_idx, val_idx) in enumerate(folds, start=1):
        train_df = data.iloc[train_idx]
        val_df = data.iloc[val_idx]

        X_train = train_df[feature_columns]
        y_train = train_df[target_column].astype(float)

        X_val = val_df[feature_columns]
        y_val = val_df[target_column].astype(float)

        pred_train_raw, pred_val_raw = fit_predict_xgb(
            params=params,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
        )

        pred_train_signal = make_signal(spec, train_df, pred_train_raw)
        pred_val_signal = make_signal(spec, val_df, pred_val_raw)

        val_spread = realized_spread(spec, val_df)

        train_rmse = rmse(y_train, pred_train_raw)
        val_rmse = rmse(y_val, pred_val_raw)
        overfit_ratio = float(val_rmse / train_rmse) if train_rmse > 0 else np.nan
        overfit_excess = max(0.0, overfit_ratio - max_allowed_overfit_ratio) if pd.notna(overfit_ratio) else 0.0

        clean_train_signal = pred_train_signal.abs().replace([np.inf, -np.inf], np.nan).dropna()
        threshold = float(clean_train_signal.quantile(threshold_quantile)) if len(clean_train_signal) else 0.0

        pnl = simple_threshold_pnl(
            realized_spread=val_spread,
            signal_prediction=pred_val_signal,
            threshold=threshold,
        )

        rows.append(
            {
                "fold": fold_id,
                "train_rmse": train_rmse,
                "val_rmse": val_rmse,
                "val_mae": mae(y_val, pred_val_raw),
                "val_spearman": safe_corr(y_val, pred_val_raw, method="spearman"),
                "signal_spearman": safe_corr(val_spread, pred_val_signal, method="spearman"),
                "directional_accuracy": directional_accuracy(val_spread, pred_val_signal),
                "overfit_ratio_rmse": overfit_ratio,
                "overfit_excess": overfit_excess,
                "pnl": pnl,
            }
        )

    metrics = pd.DataFrame(rows)

    mean_val_rmse = float(metrics["val_rmse"].mean())
    mean_val_mae = float(metrics["val_mae"].mean())
    mean_overfit_excess = float(metrics["overfit_excess"].mean())
    mean_spearman = float(metrics["val_spearman"].fillna(0.0).mean())
    mean_signal_spearman = float(metrics["signal_spearman"].fillna(0.0).mean())
    mean_directional_accuracy = float(metrics["directional_accuracy"].fillna(0.0).mean())
    total_pnl = float(metrics["pnl"].sum())

    if objective_metric == "rmse":
        score = mean_val_rmse

    elif objective_metric == "rmse_overfit_penalized":
        score = mean_val_rmse * (1.0 + overfit_penalty_weight * mean_overfit_excess)

    elif objective_metric == "mae":
        score = mean_val_mae

    elif objective_metric == "directional_rmse":
        direction_penalty = max(0.0, 0.55 - mean_directional_accuracy)
        score = mean_val_rmse * (1.0 + 2.0 * direction_penalty)

    elif objective_metric == "spearman_rmse":
        corr_bonus = max(0.0, mean_signal_spearman)
        score = mean_val_rmse * (1.0 - 0.20 * corr_bonus)

    elif objective_metric == "pnl_penalized":
        score = mean_val_rmse - 0.0005 * total_pnl
        score *= 1.0 + overfit_penalty_weight * mean_overfit_excess

    else:
        raise ValueError(f"Unknown objective_metric={objective_metric}")

    return {
        "score": float(score),
        "mean_val_rmse": mean_val_rmse,
        "mean_val_mae": mean_val_mae,
        "mean_overfit_ratio_rmse": float(metrics["overfit_ratio_rmse"].mean()),
        "mean_overfit_excess": mean_overfit_excess,
        "mean_spearman": mean_spearman,
        "mean_signal_spearman": mean_signal_spearman,
        "mean_directional_accuracy": mean_directional_accuracy,
        "total_pnl": total_pnl,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune XGBoost hyperparameters with Optuna.")

    parser.add_argument("--target", choices=list_available_targets(), required=True)
    parser.add_argument("--master-file", type=Path, default=MASTER_FILE)
    parser.add_argument("--selected-features-json", type=Path, default=None)

    parser.add_argument("--fold-mode", choices=["manual", "auto"], default="manual")
    parser.add_argument("--window-mode", choices=["expanding", "rolling", "rolling_gap_aware", "rolling_contiguous"], default="rolling")
    parser.add_argument("--train-window-days", type=int, default=365)
    parser.add_argument("--gap-days", type=int, default=1)
    parser.add_argument("--val-days", type=int, default=30)
    parser.add_argument("--step-days", type=int, default=30)
    parser.add_argument("--max-folds", type=int, default=None)

    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--n-jobs", type=int, default=4)

    parser.add_argument(
        "--objective-metric",
        choices=[
            "rmse",
            "rmse_overfit_penalized",
            "mae",
            "directional_rmse",
            "spearman_rmse",
            "pnl_penalized",
        ],
        default="rmse_overfit_penalized",
    )

    parser.add_argument("--max-allowed-overfit-ratio", type=float, default=1.50)
    parser.add_argument("--overfit-penalty-weight", type=float, default=1.0)
    parser.add_argument("--threshold-quantile", type=float, default=0.60)

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("model_artifacts/hyperparameters/xgboost"),
    )

    parser.add_argument("--storage", type=str, default=None)

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

    feature_columns = load_selected_features(
        path=args.selected_features_json,
        data=data,
        all_features=build.feature_columns,
    )

    active_days = pd.to_datetime(data["timestamp_utc"], utc=True).dt.date.nunique()
    active_local_days = pd.to_datetime(data["date_local"]).dt.date.nunique()

    if args.fold_mode == "auto":
        params = auto_walk_forward_params(active_days)
        train_window_days = int(params["train_window_days"])
        gap_days = int(params["gap_days"])
        val_days = int(params["val_days"])
        step_days = int(params["step_days"])
        window_mode = str(params["window_mode"])
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

    out_dir = args.output_dir / args.target
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 120)
    print("OPTUNA XGBOOST TUNING")
    print("=" * 120)
    print("Target:", args.target)
    print("Target kind:", spec.target_kind)
    print("Rows:", len(data))
    print("Active UTC days:", active_days)
    print("Active local days:", active_local_days)
    print("Fold date basis: timestamp_utc")
    print("Target column:", build.target_column_name)
    print("Features used:", len(feature_columns))
    print(
        "Fold setup:",
        {
            "window_mode": window_mode,
            "train_window_days": train_window_days,
            "gap_days": gap_days,
            "val_days": val_days,
            "step_days": step_days,
            "max_folds": args.max_folds,
            "folds": len(folds),
            "fold_date_basis": "timestamp_utc",
        },
    )
    print("Trials:", args.n_trials)
    print("Optuna n_jobs:", args.n_jobs)
    print("CPU count:", os.cpu_count())
    print("Objective metric:", args.objective_metric)

    trial_records = []

    def objective(trial: optuna.Trial) -> float:
        params = make_xgb_params(trial)
        t0 = time.perf_counter()

        metrics = evaluate_trial(
            params=params,
            data=data,
            spec=spec,
            target_column=build.target_column_name,
            feature_columns=feature_columns,
            folds=folds,
            objective_metric=args.objective_metric,
            max_allowed_overfit_ratio=args.max_allowed_overfit_ratio,
            overfit_penalty_weight=args.overfit_penalty_weight,
            threshold_quantile=args.threshold_quantile,
        )

        elapsed = time.perf_counter() - t0

        for k, v in metrics.items():
            trial.set_user_attr(k, v)

        trial.set_user_attr("elapsed_seconds", elapsed)

        record = {
            "trial": trial.number,
            "elapsed_seconds": elapsed,
            **metrics,
            **params,
        }
        trial_records.append(record)

        print(
            f"[trial {trial.number}] score={metrics['score']:.4f}, "
            f"rmse={metrics['mean_val_rmse']:.4f}, "
            f"overfit={metrics['mean_overfit_ratio_rmse']:.3f}, "
            f"dir={metrics['mean_directional_accuracy']:.3f}, "
            f"signal_spearman={metrics['mean_signal_spearman']:.3f}, "
            f"pnl={metrics['total_pnl']:.2f}, "
            f"elapsed={elapsed:.1f}s",
            flush=True,
        )

        return metrics["score"]

    study = optuna.create_study(
        direction="minimize",
        storage=args.storage,
        study_name=f"xgboost_{args.target}",
        load_if_exists=True if args.storage else False,
    )

    study.optimize(objective, n_trials=args.n_trials, n_jobs=args.n_jobs)

    trials_df = pd.DataFrame(trial_records)
    trials_path = out_dir / "optuna_trials.csv"
    trials_df.to_csv(trials_path, index=False)

    best_params = dict(study.best_trial.params)

    final_params = {
        **best_params,
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "random_state": 42,
        "n_jobs": 1,
    }

    best_payload = {
        "target": args.target,
        "target_kind": spec.target_kind,
        "target_column": build.target_column_name,
        "model_type": "xgboost",
        "objective_metric": args.objective_metric,
        "best_score": float(study.best_value),
        "best_trial_number": int(study.best_trial.number),
        "best_params": final_params,
        "selected_features_json": str(args.selected_features_json) if args.selected_features_json else None,
        "n_features": len(feature_columns),
        "features": feature_columns,
        "fold_setup": {
            "window_mode": window_mode,
            "train_window_days": train_window_days,
            "gap_days": gap_days,
            "val_days": val_days,
            "step_days": step_days,
            "max_folds": args.max_folds,
            "folds": len(folds),
            "fold_date_basis": "timestamp_utc",
        },
        "tuning_setup": {
            "n_trials": args.n_trials,
            "n_jobs": args.n_jobs,
            "max_allowed_overfit_ratio": args.max_allowed_overfit_ratio,
            "overfit_penalty_weight": args.overfit_penalty_weight,
            "threshold_quantile": args.threshold_quantile,
        },
        "best_user_attrs": study.best_trial.user_attrs,
    }

    best_path = out_dir / "best_xgboost_params.json"

    with open(best_path, "w", encoding="utf-8") as f:
        json.dump(best_payload, f, indent=2)

    elapsed_total = time.perf_counter() - run_start

    print("=" * 120)
    print("OPTUNA COMPLETE")
    print("=" * 120)
    print("Best score:", study.best_value)
    print("Best trial:", study.best_trial.number)
    print("Best params saved:", best_path)
    print("Trials saved:", trials_path)
    print(f"Elapsed: {elapsed_total / 60:.2f} minutes")


if __name__ == "__main__":
    main()
