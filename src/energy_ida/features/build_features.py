from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from energy_ida.config import MASTER_FILE, LOCAL_TZ
from energy_ida.features.target_config import (
    COMMON_CALENDAR_FEATURES,
    DA_FORECAST_FEATURES,
    DA_PRICE_FEATURES,
    FORECAST_REVISION_FEATURES,
    ID_FORECAST_FEATURES,
    TARGET_SPECS,
    TargetSpec,
    get_target_spec,
)


@dataclass
class FeatureBuildResult:
    target_name: str
    target_column: str
    data: pd.DataFrame
    feature_columns: list[str]
    target_column_name: str
    metadata_columns: list[str]


def dedupe(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))


def load_master_for_features(master_file: Path = MASTER_FILE) -> pd.DataFrame:
    if not master_file.exists():
        raise FileNotFoundError(f"Master parquet not found: {master_file}")

    df = pd.read_parquet(master_file)

    if "timestamp_utc" not in df.columns:
        raise ValueError("Master parquet must contain timestamp_utc.")

    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp_utc"])
    df = df.sort_values("timestamp_utc").drop_duplicates("timestamp_utc", keep="last")
    df = df.loc[:, ~df.columns.duplicated()].copy()
    df = df.reset_index(drop=True)

    return df


def add_base_time_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    local_ts = out["timestamp_utc"].dt.tz_convert(LOCAL_TZ)

    out["timestamp_local"] = local_ts
    out["local_time_key"] = local_ts.dt.tz_localize(None)
    out["date_local"] = local_ts.dt.date
    out["hour_local"] = local_ts.dt.hour
    out["minute_local"] = local_ts.dt.minute
    out["quarter_hour_local"] = out["minute_local"] // 15
    out["weekday_local"] = local_ts.dt.weekday
    out["month_local"] = local_ts.dt.month
    out["is_weekend"] = out["weekday_local"].isin([5, 6]).astype(int)

    out["hour_sin"] = np.sin(2 * np.pi * out["hour_local"] / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour_local"] / 24.0)

    qh_of_day = out["hour_local"] * 4 + out["quarter_hour_local"]
    out["quarter_sin"] = np.sin(2 * np.pi * qh_of_day / 96.0)
    out["quarter_cos"] = np.cos(2 * np.pi * qh_of_day / 96.0)

    out["weekday_sin"] = np.sin(2 * np.pi * out["weekday_local"] / 7.0)
    out["weekday_cos"] = np.cos(2 * np.pi * out["weekday_local"] / 7.0)

    out["month_sin"] = np.sin(2 * np.pi * (out["month_local"] - 1) / 12.0)
    out["month_cos"] = np.cos(2 * np.pi * (out["month_local"] - 1) / 12.0)

    return out


def add_missing_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived columns used by modelling.

    Important behavior:
      * If a derived column already exists but has gaps, fill only the missing
        values from the available source columns.
      * Existing non-null values are preserved.
    """
    out = df.copy()

    def fill_or_create(col: str, values: pd.Series) -> None:
        if col not in out.columns:
            out[col] = values
        else:
            out[col] = out[col].where(out[col].notna(), values)

    if "wind_onshore_forecast_da_mw" in out.columns or "wind_offshore_forecast_da_mw" in out.columns:
        cols = [
            c
            for c in ["wind_onshore_forecast_da_mw", "wind_offshore_forecast_da_mw"]
            if c in out.columns
        ]
        if cols:
            fill_or_create("wind_total_forecast_da_mw", out[cols].sum(axis=1, min_count=1))

    if "wind_onshore_forecast_id_mw" in out.columns or "wind_offshore_forecast_id_mw" in out.columns:
        cols = [
            c
            for c in ["wind_onshore_forecast_id_mw", "wind_offshore_forecast_id_mw"]
            if c in out.columns
        ]
        if cols:
            fill_or_create("wind_total_forecast_id_mw", out[cols].sum(axis=1, min_count=1))

    if "solar_forecast_da_mw" in out.columns or "wind_total_forecast_da_mw" in out.columns:
        cols = [c for c in ["solar_forecast_da_mw", "wind_total_forecast_da_mw"] if c in out.columns]
        if cols:
            fill_or_create("renewable_total_forecast_da_mw", out[cols].sum(axis=1, min_count=1))

    if "solar_forecast_id_mw" in out.columns or "wind_total_forecast_id_mw" in out.columns:
        cols = [c for c in ["solar_forecast_id_mw", "wind_total_forecast_id_mw"] if c in out.columns]
        if cols:
            fill_or_create("renewable_total_forecast_id_mw", out[cols].sum(axis=1, min_count=1))

    if "load_forecast_da_mw" in out.columns and "renewable_total_forecast_da_mw" in out.columns:
        fill_or_create(
            "residual_load_forecast_da_mw",
            out["load_forecast_da_mw"] - out["renewable_total_forecast_da_mw"],
        )

    if "load_forecast_da_mw" in out.columns and "renewable_total_forecast_id_mw" in out.columns:
        fill_or_create(
            "residual_load_forecast_id_approx_mw",
            out["load_forecast_da_mw"] - out["renewable_total_forecast_id_mw"],
        )

    revision_pairs = [
        ("solar_forecast_id_mw", "solar_forecast_da_mw", "solar_forecast_revision_id_minus_da_mw"),
        (
            "wind_onshore_forecast_id_mw",
            "wind_onshore_forecast_da_mw",
            "wind_onshore_forecast_revision_id_minus_da_mw",
        ),
        (
            "wind_offshore_forecast_id_mw",
            "wind_offshore_forecast_da_mw",
            "wind_offshore_forecast_revision_id_minus_da_mw",
        ),
        ("wind_total_forecast_id_mw", "wind_total_forecast_da_mw", "wind_total_forecast_revision_id_minus_da_mw"),
        (
            "renewable_total_forecast_id_mw",
            "renewable_total_forecast_da_mw",
            "renewable_total_forecast_revision_id_minus_da_mw",
        ),
        (
            "residual_load_forecast_id_approx_mw",
            "residual_load_forecast_da_mw",
            "residual_load_revision_id_minus_da_approx_mw",
        ),
    ]

    for id_col, da_col, out_col in revision_pairs:
        if id_col in out.columns and da_col in out.columns:
            fill_or_create(out_col, out[id_col] - out[da_col])

    for product in ["ida1", "ida2", "ida3"]:
        price_col = f"{product}_price_eur_mwh"
        spread_col = f"{product}_minus_da_spread_eur_mwh"

        if price_col in out.columns and "da_price_eur_mwh" in out.columns:
            fill_or_create(spread_col, out[price_col] - out["da_price_eur_mwh"])

    continuous_price_candidates = [
        "continuous_15min_weighted_avg_price_eur_mwh",
        "continuous_15min_price_eur_mwh",
    ]

    continuous_price = next((c for c in continuous_price_candidates if c in out.columns), None)

    if continuous_price is not None and "da_price_eur_mwh" in out.columns:
        fill_or_create(
            "continuous_15min_minus_da_spread_eur_mwh",
            out[continuous_price] - out["da_price_eur_mwh"],
        )

    out = out.loc[:, ~out.columns.duplicated()].copy()

    return out

def add_lag_features_by_local_time(
    df: pd.DataFrame,
    columns: list[str],
    lags_days: list[int],
) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()
    created = []

    available_cols = [c for c in dedupe(columns) if c in out.columns]

    if not available_cols:
        return out, created

    if "timestamp_utc" in out.columns:
        out = out.sort_values("timestamp_utc").drop_duplicates("timestamp_utc", keep="last")

    for lag in lags_days:
        source = out[["local_time_key"] + available_cols].copy()

        source = (
            source.sort_values("local_time_key")
            .groupby("local_time_key", as_index=False)
            .last()
        )

        source["local_time_key"] = source["local_time_key"] + pd.Timedelta(days=lag)

        rename = {c: f"{c}_lag_{lag}d_same_local_qh" for c in available_cols}
        source = source.rename(columns=rename)

        if source["local_time_key"].duplicated().any():
            duplicated = int(source["local_time_key"].duplicated().sum())
            raise RuntimeError(
                f"Lag source has {duplicated} duplicate local_time_key values after "
                f"lag={lag}. Refusing to merge because this would multiply rows."
            )

        rows_before = len(out)
        out = out.merge(source, on="local_time_key", how="left", validate="many_to_one")
        rows_after = len(out)

        if rows_after != rows_before:
            raise RuntimeError(
                f"Lag merge unexpectedly changed row count for lag={lag}: "
                f"{rows_before} -> {rows_after}. This indicates duplicate merge keys."
            )

        created.extend(rename.values())

    out = out.loc[:, ~out.columns.duplicated()].copy()

    return out, created


def add_rolling_same_slot_features(
    df: pd.DataFrame,
    columns: list[str],
    windows_days: list[int],
) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()
    created = []

    available_cols = [c for c in dedupe(columns) if c in out.columns]

    if not available_cols:
        return out, created

    out = out.sort_values("timestamp_utc").reset_index(drop=True)

    for col in available_cols:
        grouped = out.groupby(["hour_local", "minute_local"], sort=False)[col]
        shifted = grouped.shift(1)

        for window in windows_days:
            min_periods = max(2, int(np.ceil(window / 3)))

            mean_col = f"{col}_rolling_mean_{window}d_same_local_qh"
            median_col = f"{col}_rolling_median_{window}d_same_local_qh"
            std_col = f"{col}_rolling_std_{window}d_same_local_qh"

            out[mean_col] = (
                shifted.groupby([out["hour_local"], out["minute_local"]])
                .rolling(window=window, min_periods=min_periods)
                .mean()
                .reset_index(level=[0, 1], drop=True)
            )

            out[median_col] = (
                shifted.groupby([out["hour_local"], out["minute_local"]])
                .rolling(window=window, min_periods=min_periods)
                .median()
                .reset_index(level=[0, 1], drop=True)
            )

            out[std_col] = (
                shifted.groupby([out["hour_local"], out["minute_local"]])
                .rolling(window=window, min_periods=min_periods)
                .std()
                .reset_index(level=[0, 1], drop=True)
            )

            created.extend([mean_col, median_col, std_col])

    out = out.loc[:, ~out.columns.duplicated()].copy()

    return out, created


def add_lagged_forecast_error_features(
    df: pd.DataFrame,
    lags_days: list[int],
    rolling_windows_days: list[int],
) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()
    error_cols = []

    pairs = [
        ("actual_generation_solar_actual_aggregated", "solar_forecast_da_mw", "solar_da_forecast_error_mw"),
        ("actual_generation_solar_actual_aggregated", "solar_forecast_id_mw", "solar_id_forecast_error_mw"),
        (
            "actual_generation_wind_onshore_actual_aggregated",
            "wind_onshore_forecast_da_mw",
            "wind_onshore_da_forecast_error_mw",
        ),
        (
            "actual_generation_wind_onshore_actual_aggregated",
            "wind_onshore_forecast_id_mw",
            "wind_onshore_id_forecast_error_mw",
        ),
        (
            "actual_generation_wind_offshore_actual_aggregated",
            "wind_offshore_forecast_da_mw",
            "wind_offshore_da_forecast_error_mw",
        ),
        (
            "actual_generation_wind_offshore_actual_aggregated",
            "wind_offshore_forecast_id_mw",
            "wind_offshore_id_forecast_error_mw",
        ),
        ("actual_load_actual_load", "load_forecast_da_mw", "load_da_forecast_error_mw"),
    ]

    for actual_col, forecast_col, error_col in pairs:
        if actual_col in out.columns and forecast_col in out.columns:
            out[error_col] = out[actual_col] - out[forecast_col]
            error_cols.append(error_col)

    out, lagged_cols = add_lag_features_by_local_time(out, error_cols, lags_days)
    out, rolling_cols = add_rolling_same_slot_features(out, error_cols, rolling_windows_days)

    return out, lagged_cols + rolling_cols


def add_regime_indicators(df: pd.DataFrame, spec: TargetSpec) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()
    created = []

    candidates = [
        spec.spread_column,
        spec.price_column,
        "da_price_eur_mwh",
        "renewable_total_forecast_revision_id_minus_da_mw",
        "wind_total_forecast_revision_id_minus_da_mw",
        "residual_load_revision_id_minus_da_approx_mw",
    ]

    for base in candidates:
        std_28 = f"{base}_rolling_std_28d_same_local_qh"
        mean_28 = f"{base}_rolling_mean_28d_same_local_qh"
        median_28 = f"{base}_rolling_median_28d_same_local_qh"
        std_7 = f"{base}_rolling_std_7d_same_local_qh"
        mean_7 = f"{base}_rolling_mean_7d_same_local_qh"

        if std_28 in out.columns:
            col = f"{base}_regime_vol_28d"
            out[col] = out[std_28]
            created.append(col)

        if std_7 in out.columns and std_28 in out.columns:
            col = f"{base}_regime_vol_7d_over_28d"
            out[col] = out[std_7] / out[std_28].replace(0, np.nan)
            created.append(col)

        if mean_7 in out.columns and mean_28 in out.columns and std_28 in out.columns:
            col = f"{base}_regime_mean_shift_7d_vs_28d_z"
            out[col] = (out[mean_7] - out[mean_28]) / out[std_28].replace(0, np.nan)
            created.append(col)

        if median_28 in out.columns and std_28 in out.columns:
            col = f"{base}_regime_abs_median_28d_over_vol"
            out[col] = out[median_28].abs() / out[std_28].replace(0, np.nan)
            created.append(col)

    out = out.loc[:, ~out.columns.duplicated()].copy()

    return out, created


def get_current_feature_columns(spec: TargetSpec, df_columns: list[str]) -> list[str]:
    columns = []

    if "calendar" in spec.current_feature_families:
        columns.extend(COMMON_CALENDAR_FEATURES)

    if "da_price" in spec.current_feature_families:
        columns.extend(DA_PRICE_FEATURES)

    if "da_forecast" in spec.current_feature_families:
        columns.extend(DA_FORECAST_FEATURES)

    if "id_forecast" in spec.current_feature_families:
        columns.extend(ID_FORECAST_FEATURES)

    if "forecast_revision" in spec.current_feature_families:
        columns.extend(FORECAST_REVISION_FEATURES)

    blocked = set(spec.blocked_current_columns)

    columns = [c for c in dedupe(columns) if c not in blocked]
    columns = [c for c in columns if c in df_columns]

    return columns


def get_own_product_history_columns(spec: TargetSpec, df_columns: list[str]) -> list[str]:
    cols = [
        spec.spread_column,
        spec.price_column,
        spec.target_column,
        "da_price_eur_mwh",
    ]

    if spec.volume_column is not None:
        cols.append(spec.volume_column)

    return [c for c in dedupe(cols) if c in df_columns]


def build_feature_dataset(
    target_name: str,
    master_file: Path = MASTER_FILE,
    lags_days: list[int] | None = None,
    rolling_windows_days: list[int] | None = None,
    start_date_local: str | None = None,
    end_date_local: str | None = None,
    drop_missing_target: bool = True,
) -> FeatureBuildResult:
    if lags_days is None:
        lags_days = [1, 2, 7, 14]

    if rolling_windows_days is None:
        rolling_windows_days = [3, 7, 14, 28]

    spec = get_target_spec(target_name)

    df = load_master_for_features(master_file)
    df = add_base_time_columns(df)
    df = add_missing_derived_columns(df)

    if start_date_local is not None:
        start_d = pd.Timestamp(start_date_local).date()
        df = df[df["date_local"] >= start_d].copy()

    if end_date_local is not None:
        end_d = pd.Timestamp(end_date_local).date()
        df = df[df["date_local"] <= end_d].copy()

    if spec.target_column not in df.columns:
        raise ValueError(f"Target column {spec.target_column} missing for target {target_name}.")

    current_features = get_current_feature_columns(spec, df.columns.tolist())

    own_product_history = get_own_product_history_columns(spec, df.columns.tolist())

    forecast_history_columns = (
        DA_FORECAST_FEATURES
        + ID_FORECAST_FEATURES
        + FORECAST_REVISION_FEATURES
    )
    forecast_history_columns = [c for c in dedupe(forecast_history_columns) if c in df.columns]

    lag_base_columns = dedupe(own_product_history + forecast_history_columns)

    df, lag_features = add_lag_features_by_local_time(
        df=df,
        columns=lag_base_columns,
        lags_days=lags_days,
    )

    rolling_base = [
        spec.spread_column,
        spec.price_column,
        "da_price_eur_mwh",
        "renewable_total_forecast_revision_id_minus_da_mw",
        "wind_total_forecast_revision_id_minus_da_mw",
        "residual_load_revision_id_minus_da_approx_mw",
    ]
    rolling_base = [c for c in dedupe(rolling_base) if c in df.columns]

    df, rolling_features = add_rolling_same_slot_features(
        df=df,
        columns=rolling_base,
        windows_days=rolling_windows_days,
    )

    df, lagged_error_features = add_lagged_forecast_error_features(
        df=df,
        lags_days=lags_days,
        rolling_windows_days=rolling_windows_days,
    )

    df, regime_features = add_regime_indicators(df, spec)

    feature_columns = dedupe(
        current_features
        + lag_features
        + rolling_features
        + lagged_error_features
        + regime_features
    )

    blocked = set(spec.blocked_current_columns)
    feature_columns = [c for c in feature_columns if c not in blocked]

    feature_columns = [
        c for c in feature_columns
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c])
    ]

    metadata_columns = [
        "timestamp_utc",
        "timestamp_local",
        "date_local",
        "hour_local",
        "minute_local",
        "quarter_hour_local",
        "weekday_local",
        "month_local",
    ]

    if spec.spread_column in df.columns and spec.spread_column != spec.target_column:
        metadata_columns.append(spec.spread_column)

    if spec.price_column in df.columns and spec.price_column != spec.target_column:
        metadata_columns.append(spec.price_column)

    if "da_price_eur_mwh" in df.columns:
        metadata_columns.append("da_price_eur_mwh")

    final_cols = dedupe(metadata_columns + feature_columns + [spec.target_column])
    model_df = df[final_cols].copy()
    model_df = model_df.loc[:, ~model_df.columns.duplicated()].copy()

    if drop_missing_target:
        required_target_cols = [spec.target_column]

        # Price targets are evaluated as implied spreads:
        #     predicted_product_price - known_DA_price.
        # Rows without DA price are therefore not usable for comparable
        # trading/backtest evaluation, even if the product price itself exists.
        if spec.target_kind == "price":
            required_target_cols.append("da_price_eur_mwh")

        # Spread targets should require the spread column itself. This is
        # usually the target column, but keep the explicit requirement for
        # clarity and for future target definitions.
        if spec.target_kind == "spread":
            required_target_cols.append(spec.spread_column)

        required_target_cols = [
            c for c in dedupe(required_target_cols)
            if c in model_df.columns
        ]

        model_df = model_df.dropna(subset=required_target_cols).copy()

    model_df = model_df.sort_values("timestamp_utc").drop_duplicates("timestamp_utc", keep="last")
    model_df = model_df.reset_index(drop=True)

    return FeatureBuildResult(
        target_name=target_name,
        target_column=spec.target_column,
        data=model_df,
        feature_columns=feature_columns,
        target_column_name=spec.target_column,
        metadata_columns=dedupe(metadata_columns),
    )


def list_available_targets() -> list[str]:
    return sorted(TARGET_SPECS.keys())
