from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from energy_ida.config import MASTER_FILE, LOCAL_TZ


DEFAULT_REPORT_DIR = Path("reports_local")
DEFAULT_AUDIT_LOG = Path("data/audit/entsoe_document_log.parquet")


FEATURE_REGISTRY: list[dict[str, Any]] = [
    # ------------------------------------------------------------
    # Targets / prices
    # ------------------------------------------------------------
    {
        "column": "ida1_minus_da_spread_eur_mwh",
        "group": "target",
        "source": "derived",
        "document": "derived from IDA1 and DA",
        "initial_status_for_ida1_tplus1": "target_only",
        "notes": "Target for spread model. Never use as contemporaneous feature.",
    },
    {
        "column": "ida1_price_eur_mwh",
        "group": "target_price",
        "source": "EPEX",
        "document": "IDA1 auction result",
        "initial_status_for_ida1_tplus1": "target_only",
        "notes": "Target for price model. Lagged values are safe; same-delivery value is not.",
    },
    {
        "column": "da_price_eur_mwh",
        "group": "known_price",
        "source": "ENTSO-E/EPEX",
        "document": "A44 day-ahead price",
        "initial_status_for_ida1_tplus1": "likely_safe",
        "notes": "Known after DA publication. Must be available before IDA1 forecast origin.",
    },

    # ------------------------------------------------------------
    # DA forecasts
    # ------------------------------------------------------------
    {
        "column": "solar_forecast_da_mw",
        "group": "da_forecast",
        "source": "ENTSO-E",
        "document": "A69 processType A01",
        "initial_status_for_ida1_tplus1": "likely_safe",
        "notes": "DA renewables forecast. Need createdDateTime audit for strict point-in-time proof.",
    },
    {
        "column": "wind_offshore_forecast_da_mw",
        "group": "da_forecast",
        "source": "ENTSO-E",
        "document": "A69 processType A01",
        "initial_status_for_ida1_tplus1": "likely_safe",
        "notes": "DA renewables forecast. Need createdDateTime audit for strict point-in-time proof.",
    },
    {
        "column": "wind_onshore_forecast_da_mw",
        "group": "da_forecast",
        "source": "ENTSO-E",
        "document": "A69 processType A01",
        "initial_status_for_ida1_tplus1": "likely_safe",
        "notes": "DA renewables forecast. Need createdDateTime audit for strict point-in-time proof.",
    },
    {
        "column": "wind_total_forecast_da_mw",
        "group": "da_forecast",
        "source": "ENTSO-E",
        "document": "A69 processType A01 / derived",
        "initial_status_for_ida1_tplus1": "likely_safe",
        "notes": "Derived from onshore + offshore DA wind forecasts.",
    },
    {
        "column": "renewable_total_forecast_da_mw",
        "group": "da_forecast",
        "source": "ENTSO-E",
        "document": "A69 processType A01 / derived",
        "initial_status_for_ida1_tplus1": "likely_safe",
        "notes": "Derived from solar + wind DA forecasts.",
    },
    {
        "column": "load_forecast_da_mw",
        "group": "da_forecast",
        "source": "ENTSO-E",
        "document": "A65 processType A01",
        "initial_status_for_ida1_tplus1": "likely_safe",
        "notes": "DA load forecast. Need createdDateTime audit for strict point-in-time proof.",
    },
    {
        "column": "residual_load_forecast_da_mw",
        "group": "da_forecast_derived",
        "source": "derived",
        "document": "load_forecast_da_mw - renewable_total_forecast_da_mw",
        "initial_status_for_ida1_tplus1": "likely_safe",
        "notes": "Safe if load DA and renewable DA forecasts are safe.",
    },
    {
        "column": "total_generation_forecast_da_mw",
        "group": "da_forecast",
        "source": "ENTSO-E",
        "document": "A71 processType A01",
        "initial_status_for_ida1_tplus1": "likely_safe",
        "notes": "Total generation forecast if available. Need createdDateTime audit.",
    },

    # ------------------------------------------------------------
    # Intraday forecasts / revisions
    # ------------------------------------------------------------
    {
        "column": "solar_forecast_id_mw",
        "group": "id_forecast",
        "source": "ENTSO-E",
        "document": "A69 processType A40",
        "initial_status_for_ida1_tplus1": "audit_required",
        "notes": "Intraday forecast. Must verify createdDateTime is before forecast origin.",
    },
    {
        "column": "wind_offshore_forecast_id_mw",
        "group": "id_forecast",
        "source": "ENTSO-E",
        "document": "A69 processType A40",
        "initial_status_for_ida1_tplus1": "audit_required",
        "notes": "Intraday forecast. Must verify createdDateTime is before forecast origin.",
    },
    {
        "column": "wind_onshore_forecast_id_mw",
        "group": "id_forecast",
        "source": "ENTSO-E",
        "document": "A69 processType A40",
        "initial_status_for_ida1_tplus1": "audit_required",
        "notes": "Intraday forecast. Must verify createdDateTime is before forecast origin.",
    },
    {
        "column": "wind_total_forecast_id_mw",
        "group": "id_forecast",
        "source": "ENTSO-E",
        "document": "A69 processType A40 / derived",
        "initial_status_for_ida1_tplus1": "audit_required",
        "notes": "Derived intraday wind forecast. Must verify underlying A40 timing.",
    },
    {
        "column": "renewable_total_forecast_id_mw",
        "group": "id_forecast",
        "source": "ENTSO-E",
        "document": "A69 processType A40 / derived",
        "initial_status_for_ida1_tplus1": "audit_required",
        "notes": "Derived intraday renewables forecast. Must verify underlying A40 timing.",
    },
    {
        "column": "solar_forecast_revision_id_minus_da_mw",
        "group": "forecast_revision",
        "source": "derived",
        "document": "A69 A40 minus A69 A01",
        "initial_status_for_ida1_tplus1": "audit_required",
        "notes": "Safe only if A40 forecast is available before forecast origin.",
    },
    {
        "column": "wind_onshore_forecast_revision_id_minus_da_mw",
        "group": "forecast_revision",
        "source": "derived",
        "document": "A69 A40 minus A69 A01",
        "initial_status_for_ida1_tplus1": "audit_required",
        "notes": "Safe only if A40 forecast is available before forecast origin.",
    },
    {
        "column": "wind_offshore_forecast_revision_id_minus_da_mw",
        "group": "forecast_revision",
        "source": "derived",
        "document": "A69 A40 minus A69 A01",
        "initial_status_for_ida1_tplus1": "audit_required",
        "notes": "Safe only if A40 forecast is available before forecast origin.",
    },
    {
        "column": "wind_total_forecast_revision_id_minus_da_mw",
        "group": "forecast_revision",
        "source": "derived",
        "document": "A69 A40 minus A69 A01",
        "initial_status_for_ida1_tplus1": "audit_required",
        "notes": "Safe only if A40 forecast is available before forecast origin.",
    },
    {
        "column": "renewable_total_forecast_revision_id_minus_da_mw",
        "group": "forecast_revision",
        "source": "derived",
        "document": "A69 A40 minus A69 A01",
        "initial_status_for_ida1_tplus1": "audit_required",
        "notes": "Safe only if A40 forecast is available before forecast origin.",
    },
    {
        "column": "residual_load_forecast_id_approx_mw",
        "group": "id_forecast_derived",
        "source": "derived",
        "document": "load DA forecast minus renewable ID forecast",
        "initial_status_for_ida1_tplus1": "audit_required",
        "notes": "Approximate ID residual load. Safe only if renewable ID forecast is safe.",
    },
    {
        "column": "residual_load_revision_id_minus_da_approx_mw",
        "group": "forecast_revision",
        "source": "derived",
        "document": "residual load ID approx minus residual load DA",
        "initial_status_for_ida1_tplus1": "audit_required",
        "notes": "Safe only if underlying A40 features are available before forecast origin.",
    },

    # ------------------------------------------------------------
    # EPEX non-target products
    # ------------------------------------------------------------
    {
        "column": "ida2_price_eur_mwh",
        "group": "later_market_price",
        "source": "EPEX",
        "document": "IDA2 auction result",
        "initial_status_for_ida1_tplus1": "unsafe_for_ida1",
        "notes": "Useful for separate IDA2 model, not for IDA1 forecast.",
    },
    {
        "column": "continuous_15min_price_eur_mwh",
        "group": "later_market_price",
        "source": "EPEX",
        "document": "Continuous 15-min final/weighted average",
        "initial_status_for_ida1_tplus1": "unsafe_for_ida1",
        "notes": "Final continuous value is not known before IDA1. Later use only live snapshots if implemented.",
    },
]


def _as_utc_datetime(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, utc=True, errors="coerce")


def _excel_safe(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    out = df.copy()

    for col in out.columns:
        s = out[col]

        if pd.api.types.is_datetime64_any_dtype(s):
            if getattr(s.dt, "tz", None) is not None:
                out[col] = s.dt.strftime("%Y-%m-%d %H:%M:%S%z")
            else:
                out[col] = s.dt.strftime("%Y-%m-%d %H:%M:%S")

        elif s.dtype == "object":
            if s.map(lambda x: isinstance(x, pd.Timestamp)).any():
                out[col] = s.map(
                    lambda x: x.isoformat()
                    if isinstance(x, pd.Timestamp) and not pd.isna(x)
                    else x
                )

    return out


def _write_sheet(writer: pd.ExcelWriter, df: pd.DataFrame, sheet_name: str) -> None:
    _excel_safe(df).to_excel(writer, sheet_name=sheet_name, index=False)


def load_master(master_file: Path) -> pd.DataFrame:
    if not master_file.exists():
        raise FileNotFoundError(f"Master parquet not found: {master_file}")

    df = pd.read_parquet(master_file)

    if "timestamp_utc" not in df.columns:
        raise ValueError("Master parquet must contain timestamp_utc column.")

    df["timestamp_utc"] = _as_utc_datetime(df["timestamp_utc"])
    df = df.dropna(subset=["timestamp_utc"])
    df = df.sort_values("timestamp_utc").reset_index(drop=True)

    return df


def add_time_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    local_ts = out["timestamp_utc"].dt.tz_convert(LOCAL_TZ)

    out["date_utc"] = out["timestamp_utc"].dt.date
    out["date_local"] = local_ts.dt.date
    out["hour_local"] = local_ts.dt.hour
    out["quarter_hour_local"] = local_ts.dt.minute // 15
    out["weekday_local"] = local_ts.dt.weekday

    return out


def completeness_by_day(
    df: pd.DataFrame,
    cols: list[str],
    date_col: str = "date_local",
    tail_days: int = 45,
) -> pd.DataFrame:
    existing_cols = [c for c in cols if c in df.columns]

    if not existing_cols:
        return pd.DataFrame()

    daily = (
        df.groupby(date_col)[existing_cols]
        .apply(lambda x: x.notna().sum())
        .reset_index()
        .sort_values(date_col)
        .tail(tail_days)
    )

    return daily


def build_feature_coverage(df: pd.DataFrame) -> pd.DataFrame:
    max_ts = df["timestamp_utc"].max()
    rows = []

    for item in FEATURE_REGISTRY:
        col = item["column"]
        exists = col in df.columns

        if exists:
            sub = df[["timestamp_utc", col]].dropna(subset=[col])
            non_null_rows = len(sub)
            first_non_null = sub["timestamp_utc"].min() if non_null_rows else pd.NaT
            last_non_null = sub["timestamp_utc"].max() if non_null_rows else pd.NaT
            non_null_last_2d = (
                sub[sub["timestamp_utc"] >= max_ts - pd.Timedelta(days=2)].shape[0]
                if non_null_rows
                else 0
            )
            non_null_last_7d = (
                sub[sub["timestamp_utc"] >= max_ts - pd.Timedelta(days=7)].shape[0]
                if non_null_rows
                else 0
            )
            latest_lag_hours = (
                (max_ts - last_non_null).total_seconds() / 3600.0
                if pd.notna(last_non_null)
                else pd.NA
            )
        else:
            non_null_rows = 0
            first_non_null = pd.NaT
            last_non_null = pd.NaT
            non_null_last_2d = 0
            non_null_last_7d = 0
            latest_lag_hours = pd.NA

        rows.append(
            {
                "column": col,
                "exists_in_master": exists,
                "group": item["group"],
                "source": item["source"],
                "document": item["document"],
                "initial_status_for_ida1_tplus1": item["initial_status_for_ida1_tplus1"],
                "non_null_rows": non_null_rows,
                "first_non_null": first_non_null,
                "last_non_null": last_non_null,
                "latest_lag_hours_vs_master_end": latest_lag_hours,
                "non_null_last_2d": non_null_last_2d,
                "non_null_last_7d": non_null_last_7d,
                "notes": item["notes"],
            }
        )

    out = pd.DataFrame(rows)

    return out.sort_values(
        ["initial_status_for_ida1_tplus1", "group", "column"],
        ascending=True,
    ).reset_index(drop=True)


def build_group_summary(coverage: pd.DataFrame) -> pd.DataFrame:
    if coverage.empty:
        return pd.DataFrame()

    out = (
        coverage.groupby(["group", "initial_status_for_ida1_tplus1"], dropna=False)
        .agg(
            n_features=("column", "count"),
            n_existing=("exists_in_master", "sum"),
            total_non_null_rows=("non_null_rows", "sum"),
            latest_feature_timestamp=("last_non_null", "max"),
        )
        .reset_index()
        .sort_values(["initial_status_for_ida1_tplus1", "group"])
    )

    return out


def load_optional_audit_log(audit_log: Path) -> pd.DataFrame:
    if not audit_log.exists():
        return pd.DataFrame(
            [
                {
                    "message": (
                        f"No metadata audit log found at {audit_log}. "
                        "This is expected before we add ENTSO-E document metadata logging. "
                        "The current report can audit non-null availability, but not strict createdDateTime."
                    )
                }
            ]
        )

    audit = pd.read_parquet(audit_log)

    for col in ["query_time_utc", "createdDateTime", "delivery_start_utc", "delivery_end_utc"]:
        if col in audit.columns:
            audit[col] = pd.to_datetime(audit[col], utc=True, errors="coerce")

    return audit.sort_values(
        [c for c in ["query_time_utc", "documentType", "processType"] if c in audit.columns]
    ).reset_index(drop=True)


def summarize_audit_log(audit: pd.DataFrame) -> pd.DataFrame:
    if audit.empty or "message" in audit.columns:
        return audit

    group_cols = [c for c in ["documentType", "processType", "source_name"] if c in audit.columns]

    if not group_cols:
        return pd.DataFrame([{"message": "Audit log exists, but expected metadata columns are not present."}])

    agg = {
        "n_rows": ("documentType", "count") if "documentType" in audit.columns else (group_cols[0], "count"),
    }

    for c in ["query_time_utc", "createdDateTime", "delivery_start_utc", "delivery_end_utc"]:
        if c in audit.columns:
            agg[f"first_{c}"] = (c, "min")
            agg[f"last_{c}"] = (c, "max")

    return audit.groupby(group_cols, dropna=False).agg(**agg).reset_index()


def infer_preliminary_feature_set(coverage: pd.DataFrame) -> pd.DataFrame:
    """
    Conservative initial decision, before metadata createdDateTime audit.

    This does not select final model features; it only labels candidates.
    """
    rows = []

    for _, r in coverage.iterrows():
        status = r["initial_status_for_ida1_tplus1"]
        exists = bool(r["exists_in_master"])
        non_null = int(r["non_null_rows"])

        if not exists or non_null == 0:
            decision = "exclude_now_missing"
            reason = "Column missing or fully null in master."
        elif status == "likely_safe":
            decision = "candidate_tier_1"
            reason = "Likely known before IDA1; still verify metadata for strict point-in-time audit."
        elif status == "audit_required":
            decision = "hold_until_metadata_audit"
            reason = "Potentially useful, but must verify createdDateTime before forecast origin."
        elif status == "unsafe_for_ida1":
            decision = "exclude_for_ida1"
            reason = "Not known before IDA1 forecast origin."
        elif status == "target_only":
            decision = "target_or_lag_only"
            reason = "Can be target or lagged feature, not same-timestamp feature."
        else:
            decision = "review"
            reason = "Unknown status."

        rows.append(
            {
                "column": r["column"],
                "group": r["group"],
                "source": r["source"],
                "document": r["document"],
                "decision": decision,
                "reason": reason,
                "non_null_rows": r["non_null_rows"],
                "first_non_null": r["first_non_null"],
                "last_non_null": r["last_non_null"],
            }
        )

    return pd.DataFrame(rows)


def write_text_report(
    path: Path,
    df: pd.DataFrame,
    coverage: pd.DataFrame,
    group_summary: pd.DataFrame,
    preliminary_feature_set: pd.DataFrame,
    audit_summary: pd.DataFrame,
    daily_counts: pd.DataFrame,
) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("=" * 120 + "\n")
        f.write("FEATURE AVAILABILITY AUDIT\n")
        f.write("=" * 120 + "\n")
        f.write(f"Master file: {MASTER_FILE}\n")
        f.write(f"Shape: {df.shape}\n")
        f.write(f"Coverage: {df['timestamp_utc'].min()} -> {df['timestamp_utc'].max()}\n")
        f.write(f"Duplicate timestamps: {df['timestamp_utc'].duplicated().sum()}\n")

        f.write("\n" + "=" * 120 + "\n")
        f.write("GROUP SUMMARY\n")
        f.write("=" * 120 + "\n")
        f.write(group_summary.to_string(index=False))

        f.write("\n\n" + "=" * 120 + "\n")
        f.write("FEATURE COVERAGE\n")
        f.write("=" * 120 + "\n")
        f.write(coverage.to_string(index=False))

        f.write("\n\n" + "=" * 120 + "\n")
        f.write("PRELIMINARY FEATURE SET DECISIONS\n")
        f.write("=" * 120 + "\n")
        f.write(preliminary_feature_set.to_string(index=False))

        f.write("\n\n" + "=" * 120 + "\n")
        f.write("AUDIT LOG SUMMARY / CREATEDDATETIME METADATA\n")
        f.write("=" * 120 + "\n")
        f.write(audit_summary.to_string(index=False))

        f.write("\n\n" + "=" * 120 + "\n")
        f.write("RECENT DAILY COMPLETENESS COUNTS\n")
        f.write("=" * 120 + "\n")
        f.write(daily_counts.to_string(index=False))


def run_feature_availability_audit(
    master_file: Path,
    report_dir: Path,
    audit_log: Path,
    tail_days: int,
) -> dict[str, Path]:
    report_dir.mkdir(parents=True, exist_ok=True)

    df = load_master(master_file)
    df = add_time_columns(df)

    coverage = build_feature_coverage(df)
    group_summary = build_group_summary(coverage)
    preliminary_feature_set = infer_preliminary_feature_set(coverage)

    daily_cols = [
        c for c in coverage.loc[coverage["exists_in_master"], "column"].tolist()
        if c in df.columns
    ]
    daily_counts = completeness_by_day(df, daily_cols, date_col="date_local", tail_days=tail_days)

    audit = load_optional_audit_log(audit_log)
    audit_summary = summarize_audit_log(audit)

    report_txt = report_dir / "feature_availability_audit.txt"
    report_xlsx = report_dir / "feature_availability_audit.xlsx"

    write_text_report(
        path=report_txt,
        df=df,
        coverage=coverage,
        group_summary=group_summary,
        preliminary_feature_set=preliminary_feature_set,
        audit_summary=audit_summary,
        daily_counts=daily_counts,
    )

    with pd.ExcelWriter(report_xlsx, engine="openpyxl") as writer:
        summary = pd.DataFrame(
            [
                {"metric": "master_file", "value": str(master_file)},
                {"metric": "rows", "value": len(df)},
                {"metric": "columns", "value": df.shape[1]},
                {"metric": "coverage_start", "value": df["timestamp_utc"].min().isoformat()},
                {"metric": "coverage_end", "value": df["timestamp_utc"].max().isoformat()},
                {"metric": "duplicate_timestamps", "value": int(df["timestamp_utc"].duplicated().sum())},
                {"metric": "audit_log", "value": str(audit_log)},
                {"metric": "audit_log_exists", "value": audit_log.exists()},
            ]
        )

        _write_sheet(writer, summary, "summary")
        _write_sheet(writer, group_summary, "group_summary")
        _write_sheet(writer, coverage, "feature_coverage")
        _write_sheet(writer, preliminary_feature_set, "prelim_feature_set")
        _write_sheet(writer, daily_counts, "daily_completeness")
        _write_sheet(writer, audit_summary, "audit_log_summary")

        if not audit.empty and "message" not in audit.columns:
            _write_sheet(writer, audit.tail(500), "audit_log_tail")

    return {
        "txt": report_txt,
        "xlsx": report_xlsx,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect feature availability for IDA1 forecast modelling.")

    parser.add_argument(
        "--master-file",
        type=Path,
        default=MASTER_FILE,
        help=f"Path to master parquet. Default: {MASTER_FILE}",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=DEFAULT_REPORT_DIR,
        help=f"Output report directory. Default: {DEFAULT_REPORT_DIR}",
    )
    parser.add_argument(
        "--audit-log",
        type=Path,
        default=DEFAULT_AUDIT_LOG,
        help=f"Optional ENTSO-E metadata audit log. Default: {DEFAULT_AUDIT_LOG}",
    )
    parser.add_argument(
        "--tail-days",
        type=int,
        default=45,
        help="Number of recent local dates to include in daily completeness table.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    outputs = run_feature_availability_audit(
        master_file=args.master_file,
        report_dir=args.report_dir,
        audit_log=args.audit_log,
        tail_days=args.tail_days,
    )

    print("Feature availability audit complete.")
    print(f"Saved text report:  {outputs['txt']}")
    print(f"Saved Excel report: {outputs['xlsx']}")


if __name__ == "__main__":
    main()
