import argparse
from datetime import timedelta

import pandas as pd

from energy_ida.config import MASTER_FILE, EPEX_PRODUCT_GROUPS
from energy_ida.data_sources.epex_client import fetch_epex_germany_for_dates
from energy_ida.data_sources.entsoe_client import fetch_entsoe_features_for_day
from energy_ida.storage import load_master, upsert_save_master


# ============================================================
# DATE WINDOWS
# ============================================================

def build_date_window(days_back: int = 2, days_forward: int = 1):
    now_local = pd.Timestamp.now(tz="Europe/Berlin")
    start = now_local.date() - timedelta(days=days_back)
    end = now_local.date() + timedelta(days=days_forward)
    return start, end


# ============================================================
# EPEX WIDE TRANSFORM
# ============================================================

def make_product_name(row: pd.Series) -> str:
    area = str(row["epex_market_area"]).lower().replace("-", "_")
    modality = str(row["epex_modality"]).lower()
    sub = str(row["epex_sub_modality"]).lower()
    auction = str(row["epex_auction"]).lower()
    product = str(row["epex_product"]).lower()

    parts = ["epex", area, modality]

    if sub and sub not in {"nan", "none", "<na>"}:
        parts.append(sub)

    if auction and auction not in {"nan", "none", "<na>"}:
        parts.append(auction)

    if product and product not in {"nan", "none", "<na>"}:
        parts.append(f"{product}min")

    return "_".join(parts)


def expand_hourly_da_to_15min(out: pd.DataFrame) -> pd.DataFrame:
    """
    DA prices are hourly. The master table is 15-min.
    This expands each DA hourly value into four quarter-hour rows.
    """
    if "da_price_eur_mwh" not in out.columns:
        return out

    out = out.copy()
    out["timestamp_utc"] = pd.to_datetime(out["timestamp_utc"], utc=True)

    da_cols = [
        c for c in [
            "da_price_eur_mwh",
            "da_buy_volume_mwh",
            "da_sell_volume_mwh",
            "da_volume_mwh",
        ]
        if c in out.columns
    ]

    da = out[["timestamp_utc"] + da_cols].dropna(
        subset=["da_price_eur_mwh"]
    ).copy()

    if da.empty:
        return out

    da = (
        da.sort_values("timestamp_utc")
          .drop_duplicates("timestamp_utc", keep="last")
          .set_index("timestamp_utc")
          .resample("15min")
          .ffill(limit=3)
          .reset_index()
    )

    out = out.drop(columns=da_cols)
    out = out.merge(da, on="timestamp_utc", how="outer")

    return out.sort_values("timestamp_utc").reset_index(drop=True)


def wide_epex(epex_long: pd.DataFrame) -> pd.DataFrame:
    if epex_long.empty:
        return pd.DataFrame()

    epex_long = epex_long.copy()
    epex_long["timestamp_utc"] = pd.to_datetime(epex_long["timestamp_utc"], utc=True)
    epex_long["product_name"] = epex_long.apply(make_product_name, axis=1)

    value_cols = [
        "price_eur_mwh",
        "buy_volume_mwh",
        "sell_volume_mwh",
        "volume_mwh",
        "low_price_eur_mwh",
        "high_price_eur_mwh",
        "last_price_eur_mwh",
        "weighted_avg_price_eur_mwh",
    ]

    wide_parts = []

    for value in value_cols:
        if value not in epex_long.columns:
            continue

        tmp = (
            epex_long.pivot_table(
                index="timestamp_utc",
                columns="product_name",
                values=value,
                aggfunc="last",
            )
            .reset_index()
        )

        tmp.columns.name = None

        rename = {
            c: f"{c}_{value}"
            for c in tmp.columns
            if c != "timestamp_utc"
        }

        tmp = tmp.rename(columns=rename)
        wide_parts.append(tmp)

    if not wide_parts:
        return pd.DataFrame()

    out = wide_parts[0]

    for part in wide_parts[1:]:
        out = out.merge(part, on="timestamp_utc", how="outer")

    alias_map = {
        # Day-ahead MRC
        "epex_de_lu_auction_dayahead_mrc_60min_price_eur_mwh": "da_price_eur_mwh",
        "epex_de_lu_auction_dayahead_mrc_60min_buy_volume_mwh": "da_buy_volume_mwh",
        "epex_de_lu_auction_dayahead_mrc_60min_sell_volume_mwh": "da_sell_volume_mwh",
        "epex_de_lu_auction_dayahead_mrc_60min_volume_mwh": "da_volume_mwh",

        # IDA1
        "epex_de_lu_auction_intraday_ida1_price_eur_mwh": "ida1_price_eur_mwh",
        "epex_de_lu_auction_intraday_ida1_buy_volume_mwh": "ida1_buy_volume_mw",
        "epex_de_lu_auction_intraday_ida1_sell_volume_mwh": "ida1_sell_volume_mw",
        "epex_de_lu_auction_intraday_ida1_volume_mwh": "ida1_volume_mw",

        # IDA2
        "epex_de_lu_auction_intraday_ida2_price_eur_mwh": "ida2_price_eur_mwh",
        "epex_de_lu_auction_intraday_ida2_buy_volume_mwh": "ida2_buy_volume_mw",
        "epex_de_lu_auction_intraday_ida2_sell_volume_mwh": "ida2_sell_volume_mw",
        "epex_de_lu_auction_intraday_ida2_volume_mwh": "ida2_volume_mw",

        # IDA3
        "epex_de_lu_auction_intraday_ida3_price_eur_mwh": "ida3_price_eur_mwh",
        "epex_de_lu_auction_intraday_ida3_buy_volume_mwh": "ida3_buy_volume_mw",
        "epex_de_lu_auction_intraday_ida3_sell_volume_mwh": "ida3_sell_volume_mw",
        "epex_de_lu_auction_intraday_ida3_volume_mwh": "ida3_volume_mw",

        # Continuous 15-min
        "epex_de_continuous_15min_price_eur_mwh": "continuous_15min_price_eur_mwh",
        "epex_de_continuous_15min_buy_volume_mwh": "continuous_15min_buy_volume_mwh",
        "epex_de_continuous_15min_sell_volume_mwh": "continuous_15min_sell_volume_mwh",
        "epex_de_continuous_15min_volume_mwh": "continuous_15min_volume_mwh",
        "epex_de_continuous_15min_low_price_eur_mwh": "continuous_15min_low_price_eur_mwh",
        "epex_de_continuous_15min_high_price_eur_mwh": "continuous_15min_high_price_eur_mwh",
        "epex_de_continuous_15min_last_price_eur_mwh": "continuous_15min_last_price_eur_mwh",
        "epex_de_continuous_15min_weighted_avg_price_eur_mwh": "continuous_15min_weighted_avg_price_eur_mwh",
    }

    for src, dst in alias_map.items():
        if src in out.columns and dst not in out.columns:
            out[dst] = out[src]

    out = expand_hourly_da_to_15min(out)

    if "ida1_price_eur_mwh" in out.columns and "da_price_eur_mwh" in out.columns:
        out["ida1_minus_da_spread_eur_mwh"] = (
            out["ida1_price_eur_mwh"] - out["da_price_eur_mwh"]
        )

    return out.sort_values("timestamp_utc").reset_index(drop=True)


# ============================================================
# DATA FETCHERS
# ============================================================

def fetch_epex_update(
    product_group: str,
    days_back: int,
    days_forward: int,
) -> pd.DataFrame:
    start_date, end_date = build_date_window(
        days_back=days_back,
        days_forward=days_forward,
    )

    products = EPEX_PRODUCT_GROUPS[product_group]

    print(f"EPEX product group: {product_group}")
    print(f"EPEX update window: {start_date} -> {end_date}")

    epex_long = fetch_epex_germany_for_dates(
        start_date=start_date,
        end_date=end_date,
        products=products,
        headless=True,
        browser_name="firefox",
        pause_seconds=5.0,
    )

    return wide_epex(epex_long)


def fetch_entsoe_update(
    days_back: int,
    days_forward: int,
) -> pd.DataFrame:
    start_date, end_date = build_date_window(
        days_back=days_back,
        days_forward=days_forward,
    )

    parts = []

    for day in pd.date_range(start_date, end_date, freq="D"):
        print(f"ENTSO-E features for {day.date()}")
        part = fetch_entsoe_features_for_day(day.date())

        if not part.empty:
            parts.append(part)

    if not parts:
        return pd.DataFrame()

    out = pd.concat(parts, ignore_index=True)
    out["timestamp_utc"] = pd.to_datetime(out["timestamp_utc"], utc=True)
    out = out.drop_duplicates("timestamp_utc", keep="last")

    return out.sort_values("timestamp_utc").reset_index(drop=True)


# ============================================================
# DERIVED COLUMNS
# ============================================================

def recompute_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.copy()

    if "timestamp_utc" in df.columns:
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)

    if "ida1_price_eur_mwh" in df.columns and "da_price_eur_mwh" in df.columns:
        df["ida1_minus_da_spread_eur_mwh"] = (
            df["ida1_price_eur_mwh"] - df["da_price_eur_mwh"]
        )

    if "load_forecast_da_mw" in df.columns and "renewable_total_forecast_da_mw" in df.columns:
        df["residual_load_forecast_da_mw"] = (
            df["load_forecast_da_mw"] - df["renewable_total_forecast_da_mw"]
        )

    if "load_forecast_da_mw" in df.columns and "renewable_total_forecast_id_mw" in df.columns:
        df["residual_load_forecast_id_approx_mw"] = (
            df["load_forecast_da_mw"] - df["renewable_total_forecast_id_mw"]
        )

    if (
        "residual_load_forecast_id_approx_mw" in df.columns
        and "residual_load_forecast_da_mw" in df.columns
    ):
        df["residual_load_revision_id_minus_da_approx_mw"] = (
            df["residual_load_forecast_id_approx_mw"]
            - df["residual_load_forecast_da_mw"]
        )

    return df


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source",
        choices=["epex", "entsoe", "all"],
        default="all",
    )

    parser.add_argument(
        "--epex-product-group",
        choices=list(EPEX_PRODUCT_GROUPS.keys()),
        default="ida1",
    )

    parser.add_argument("--days-back", type=int, default=2)
    parser.add_argument("--days-forward", type=int, default=1)

    args = parser.parse_args()

    updates = []

    if args.source in {"epex", "all"}:
        epex_update = fetch_epex_update(
            product_group=args.epex_product_group,
            days_back=args.days_back,
            days_forward=args.days_forward,
        )

        if not epex_update.empty:
            updates.append(epex_update)

    if args.source in {"entsoe", "all"}:
        entsoe_update = fetch_entsoe_update(
            days_back=args.days_back,
            days_forward=args.days_forward,
        )

        if not entsoe_update.empty:
            updates.append(entsoe_update)

    if not updates:
        print("No new data available. Exiting gracefully.")
        return

    update = updates[0]

    for part in updates[1:]:
        update = update.merge(part, on="timestamp_utc", how="outer")

    update = recompute_derived_columns(update)

    existing = load_master(MASTER_FILE)

    combined = upsert_save_master(
        existing=existing,
        new=update,
        path=MASTER_FILE,
    )

    combined = recompute_derived_columns(combined)

    combined = upsert_save_master(
        existing=pd.DataFrame(),
        new=combined,
        path=MASTER_FILE,
    )

    duplicate_count = combined["timestamp_utc"].duplicated().sum()
    assert duplicate_count == 0, f"Duplicate timestamps after upsert: {duplicate_count}"

    print(f"Saved master: {MASTER_FILE}")
    print(f"Rows before: {len(existing)}")
    print(f"Rows update batch: {len(update)}")
    print(f"Rows after: {len(combined)}")
    print(f"Duplicate timestamps after save: {duplicate_count}")
    print(f"Coverage: {combined['timestamp_utc'].min()} -> {combined['timestamp_utc'].max()}")


if __name__ == "__main__":
    main()
