from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from energy_ida.config import MASTER_FILE
from energy_ida.features.build_features import build_feature_dataset, list_available_targets
from energy_ida.model_selection.feature_selectors import (
    ElasticNetStabilitySelector,
    SeasonalNaiveFeatureSelector,
    XGBoostStabilitySelector,
    auto_walk_forward_params,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build target-specific features and run conservative feature selection."
    )

    parser.add_argument("--target", choices=list_available_targets(), required=True)
    parser.add_argument("--model", choices=["seasonal_naive", "elasticnet", "xgboost"], required=True)

    parser.add_argument("--master-file", type=Path, default=MASTER_FILE)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("model_artifacts/feature_selection"),
    )

    parser.add_argument("--start-date-local", type=str, default=None)
    parser.add_argument("--end-date-local", type=str, default=None)

    parser.add_argument("--fold-mode", choices=["manual", "auto"], default="manual")
    parser.add_argument("--window-mode", choices=["expanding", "rolling", "rolling_gap_aware", "rolling_contiguous"], default="expanding")
    parser.add_argument("--train-window-days", type=int, default=365)
    parser.add_argument("--gap-days", type=int, default=1)
    parser.add_argument("--val-days", type=int, default=30)
    parser.add_argument("--step-days", type=int, default=30)
    parser.add_argument("--max-folds", type=int, default=None)

    parser.add_argument("--min-selection-frequency", type=float, default=None)

    parser.add_argument("--n-jobs", type=int, default=2)

    parser.add_argument("--fast-mode", action="store_true")
    parser.add_argument("--max-train-rows-per-fold", type=int, default=30000)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    build = build_feature_dataset(
        target_name=args.target,
        master_file=args.master_file,
        start_date_local=args.start_date_local,
        end_date_local=args.end_date_local,
    )

    active_days = pd.to_datetime(build.data["timestamp_utc"], utc=True).dt.date.nunique()
    active_local_days = pd.to_datetime(build.data["date_local"]).dt.date.nunique()

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

    max_train_rows = args.max_train_rows_per_fold
    if max_train_rows <= 0:
        max_train_rows = None

    print("=" * 120)
    print("FEATURE DATASET")
    print("=" * 120)
    print("Target:", args.target)
    print("Target column:", build.target_column_name)
    print("Rows:", len(build.data))
    print("Active UTC target days:", active_days)
    print("Active local target days:", active_local_days)
    print("Fold date basis: timestamp_utc")
    print("Feature columns:", len(build.feature_columns))
    print("Coverage:", build.data["timestamp_utc"].min(), "->", build.data["timestamp_utc"].max())
    print("Non-null target rows:", build.data[build.target_column_name].notna().sum())
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
    print("Parallel n_jobs:", args.n_jobs)
    print("Fast mode:", args.fast_mode)
    print("Max train rows per fold:", max_train_rows)

    if len(build.data) == 0:
        raise SystemExit("No rows available after feature build.")

    if args.model == "seasonal_naive":
        selector = SeasonalNaiveFeatureSelector(
            target_name=args.target,
            target_column=build.target_column_name,
            feature_columns=build.feature_columns,
        )

    elif args.model == "elasticnet":
        selector = ElasticNetStabilitySelector(
            target_name=args.target,
            target_column=build.target_column_name,
            feature_columns=build.feature_columns,
            train_window_days=train_window_days,
            gap_days=gap_days,
            val_days=val_days,
            step_days=step_days,
            window_mode=window_mode,
            max_folds=args.max_folds,
            min_selection_frequency=(
                args.min_selection_frequency
                if args.min_selection_frequency is not None
                else 0.65
            ),
            max_train_rows_per_fold=max_train_rows,
            fast_mode=args.fast_mode,
            n_jobs=args.n_jobs,
        )

    elif args.model == "xgboost":
        selector = XGBoostStabilitySelector(
            target_name=args.target,
            target_column=build.target_column_name,
            feature_columns=build.feature_columns,
            train_window_days=train_window_days,
            gap_days=gap_days,
            val_days=val_days,
            step_days=step_days,
            window_mode=window_mode,
            max_folds=args.max_folds,
            min_selection_frequency=(
                args.min_selection_frequency
                if args.min_selection_frequency is not None
                else 0.65
            ),
            n_jobs=args.n_jobs,
        )

    else:
        raise ValueError(f"Unsupported model: {args.model}")

    result = selector.select(build.data)

    out_dir = args.output_dir / args.target / args.model
    result.save(out_dir)

    feature_data_path = out_dir / "feature_dataset_preview.csv"
    preview_cols = build.metadata_columns + result.selected_features + [build.target_column_name]
    preview_cols = [c for c in preview_cols if c in build.data.columns]
    preview_cols = list(dict.fromkeys(preview_cols))

    build.data[preview_cols].tail(1000).to_csv(feature_data_path, index=False)

    print("\n" + "=" * 120)
    print("SELECTION RESULT")
    print("=" * 120)
    print("Model:", result.model_type)
    print("Selected features:", len(result.selected_features))
    print("Output directory:", out_dir)

    print("\nSelected features:")
    for f in result.selected_features:
        print("  -", f)

    print("\nTop feature summary:")
    print(result.feature_summary.head(40).to_string(index=False))

    if not result.fold_metrics.empty:
        print("\nFold metrics:")
        print(result.fold_metrics.to_string(index=False))


if __name__ == "__main__":
    main()
