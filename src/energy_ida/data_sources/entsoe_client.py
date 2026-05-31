import io
import os
import time
import zipfile
import xml.etree.ElementTree as ET
from datetime import date, timedelta
from typing import Any, Dict, Iterable, Optional

import pandas as pd
import requests

from energy_ida.config import LOCAL_TZ


ENTSOE_API_URL = "https://web-api.tp.entsoe.eu/api"

DE_LU_BZN = "10Y1001A1001A82H"

GERMAN_CONTROL_AREAS = {
    "50hertz": "10YDE-VE-------2",
    "amprion": "10YDE-RWENET---I",
    "tennet_de": "10YDE-EON------1",
    "transnetbw": "10YDE-ENBW-----N",
}

PSR_MAP = {
    "B16": "solar",
    "B18": "wind_offshore",
    "B19": "wind_onshore",
}


# ============================================================
# LOW-LEVEL HELPERS
# ============================================================

def get_entsoe_api_key() -> str:
    api_key = os.getenv("ENTSOE_API_KEY")

    if not api_key:
        raise RuntimeError(
            "Missing ENTSOE_API_KEY. Set it with:\n"
            "  export ENTSOE_API_KEY='your_key'\n"
            "or add it as a GitHub Actions secret."
        )

    return api_key


def local_delivery_day_window_utc(delivery_day: date) -> tuple[pd.Timestamp, pd.Timestamp]:
    """
    Return UTC interval covering one German local delivery day.

    Example in summer:
      2026-05-31 00:00 Europe/Berlin -> 2026-05-30 22:00 UTC
      2026-06-01 00:00 Europe/Berlin -> 2026-05-31 22:00 UTC
    """
    start_local = pd.Timestamp(delivery_day).tz_localize(LOCAL_TZ)
    end_local = (pd.Timestamp(delivery_day) + pd.Timedelta(days=1)).tz_localize(LOCAL_TZ)

    return start_local.tz_convert("UTC"), end_local.tz_convert("UTC")


def fmt_entsoe_time(ts: pd.Timestamp) -> str:
    ts = pd.Timestamp(ts).tz_convert("UTC")
    return ts.strftime("%Y%m%d%H%M")


def strip_ns(tag: str) -> str:
    return tag.split("}", 1)[-1] if "}" in tag else tag


def child_text(elem: ET.Element, local_name: str) -> Optional[str]:
    for child in list(elem):
        if strip_ns(child.tag) == local_name:
            return child.text
    return None


def first_desc_text(elem: ET.Element, local_name: str) -> Optional[str]:
    for child in elem.iter():
        if strip_ns(child.tag) == local_name:
            return child.text
    return None


def iter_children(elem: ET.Element, local_name: str) -> Iterable[ET.Element]:
    for child in list(elem):
        if strip_ns(child.tag) == local_name:
            yield child


def iter_desc(elem: ET.Element, local_name: str) -> Iterable[ET.Element]:
    for child in elem.iter():
        if strip_ns(child.tag) == local_name:
            yield child


def parse_resolution(resolution: str) -> pd.Timedelta:
    if resolution == "PT15M":
        return pd.Timedelta(minutes=15)
    if resolution == "PT30M":
        return pd.Timedelta(minutes=30)
    if resolution == "PT60M":
        return pd.Timedelta(hours=1)
    if resolution == "P1D":
        return pd.Timedelta(days=1)

    # fallback for ISO-ish PTxxM / PTxxH
    if resolution.startswith("PT") and resolution.endswith("M"):
        return pd.Timedelta(minutes=int(resolution[2:-1]))
    if resolution.startswith("PT") and resolution.endswith("H"):
        return pd.Timedelta(hours=int(resolution[2:-1]))

    raise ValueError(f"Unsupported resolution: {resolution}")


def extract_xml_payload(response: requests.Response) -> str:
    content = response.content

    if content[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            xml_names = [name for name in zf.namelist() if name.lower().endswith(".xml")]

            if not xml_names:
                # Some ENTSO-E ZIPs contain XML without .xml extension.
                xml_names = zf.namelist()

            with zf.open(xml_names[0]) as f:
                return f.read().decode("utf-8", errors="replace")

    return response.text


def parse_acknowledgement_errors(xml_text: str) -> list[dict]:
    try:
        root = ET.fromstring(xml_text)
    except Exception:
        return []

    if strip_ns(root.tag) != "Acknowledgement_MarketDocument":
        return []

    errors = []

    for reason in iter_desc(root, "Reason"):
        code = first_desc_text(reason, "code")
        text = first_desc_text(reason, "text")

        if code or text:
            errors.append({"code": code, "text": text})

    return errors


def request_entsoe(params: dict, sleep_seconds: float = 0.2) -> Optional[str]:
    api_key = get_entsoe_api_key()

    full_params = dict(params)
    full_params["securityToken"] = api_key

    try:
        r = requests.get(ENTSOE_API_URL, params=full_params, timeout=60)
    except Exception as exc:
        print(f"ENTSO-E request failed: {exc}")
        return None

    time.sleep(sleep_seconds)

    xml_text = extract_xml_payload(r)

    errors = parse_acknowledgement_errors(xml_text)
    if errors:
        print(f"ENTSO-E no data/error for params={params}: {errors}")
        return None

    if r.status_code >= 400:
        print(f"ENTSO-E HTTP {r.status_code} for params={params}")
        print(xml_text[:500])
        return None

    return xml_text


# ============================================================
# XML PARSING
# ============================================================

def parse_timeseries_points(xml_text: str) -> pd.DataFrame:
    """
    Generic ENTSO-E TimeSeries parser.

    Extracts:
      timestamp_utc
      quantity / price
      businessType
      psrType
      inBiddingZone / outBiddingZone
      resolution
    """
    root = ET.fromstring(xml_text)

    rows = []

    document_type = first_desc_text(root, "type")
    process_type = first_desc_text(root, "process.processType")
    created_datetime = first_desc_text(root, "createdDateTime")

    for ts_idx, ts in enumerate(iter_desc(root, "TimeSeries"), start=1):
        business_type = child_text(ts, "businessType")
        ts_process_type = child_text(ts, "process.processType")
        in_zone = child_text(ts, "in_Domain.mRID") or child_text(ts, "inBiddingZone_Domain.mRID")
        out_zone = child_text(ts, "out_Domain.mRID") or child_text(ts, "outBiddingZone_Domain.mRID")

        psr_type = None
        mkt_psr = next(iter_desc(ts, "MktPSRType"), None)
        if mkt_psr is not None:
            psr_type = child_text(mkt_psr, "psrType")

        # Some documents may put psrType directly.
        if psr_type is None:
            psr_type = first_desc_text(ts, "psrType")

        for period_idx, period in enumerate(iter_desc(ts, "Period"), start=1):
            interval = next(iter_children(period, "timeInterval"), None)

            if interval is None:
                continue

            start_text = child_text(interval, "start")
            end_text = child_text(interval, "end")
            resolution_text = child_text(period, "resolution")

            if not start_text or not resolution_text:
                continue

            period_start = pd.Timestamp(start_text).tz_convert("UTC")
            resolution = parse_resolution(resolution_text)

            for point in iter_children(period, "Point"):
                position_text = child_text(point, "position")

                if not position_text:
                    continue

                position = int(position_text)
                timestamp_utc = period_start + (position - 1) * resolution

                quantity = child_text(point, "quantity")
                price = child_text(point, "price.amount")

                value = None
                value_type = None

                if price is not None:
                    value = float(price)
                    value_type = "price"
                elif quantity is not None:
                    value = float(quantity)
                    value_type = "quantity"

                rows.append(
                    {
                        "timestamp_utc": timestamp_utc,
                        "value": value,
                        "value_type": value_type,
                        "quantity": float(quantity) if quantity is not None else pd.NA,
                        "price": float(price) if price is not None else pd.NA,
                        "documentType": document_type,
                        "documentProcessType": process_type,
                        "createdDateTime": created_datetime,
                        "timeseries": ts_idx,
                        "businessType": business_type,
                        "processType": ts_process_type,
                        "psrType": psr_type,
                        "inBiddingZone": in_zone,
                        "outBiddingZone": out_zone,
                        "period": period_idx,
                        "period_start": period_start,
                        "period_end": pd.Timestamp(end_text).tz_convert("UTC") if end_text else pd.NaT,
                        "resolution": resolution_text,
                        "position": position,
                    }
                )

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    out["timestamp_utc"] = pd.to_datetime(out["timestamp_utc"], utc=True)
    return out.sort_values("timestamp_utc").reset_index(drop=True)


# ============================================================
# SPECIFIC FETCHERS
# ============================================================

def fetch_day_ahead_prices(delivery_day: date) -> pd.DataFrame:
    """
    ENTSO-E day-ahead prices for DE-LU bidding zone.

    Returns hourly rows:
      timestamp_utc
      da_price_eur_mwh
    """
    start_utc, end_utc = local_delivery_day_window_utc(delivery_day)

    params = {
        "documentType": "A44",
        "in_Domain": DE_LU_BZN,
        "out_Domain": DE_LU_BZN,
        "periodStart": fmt_entsoe_time(start_utc),
        "periodEnd": fmt_entsoe_time(end_utc),
    }

    xml_text = request_entsoe(params)

    if xml_text is None:
        return pd.DataFrame(columns=["timestamp_utc", "da_price_eur_mwh"])

    points = parse_timeseries_points(xml_text)

    if points.empty:
        return pd.DataFrame(columns=["timestamp_utc", "da_price_eur_mwh"])

    out = (
        points[["timestamp_utc", "price"]]
        .rename(columns={"price": "da_price_eur_mwh"})
        .dropna(subset=["da_price_eur_mwh"])
        .drop_duplicates("timestamp_utc", keep="last")
        .sort_values("timestamp_utc")
        .reset_index(drop=True)
    )

    return out


def expand_hourly_da_prices_to_15min(da: pd.DataFrame) -> pd.DataFrame:
    if da.empty:
        return da

    da = da.copy()
    da["timestamp_utc"] = pd.to_datetime(da["timestamp_utc"], utc=True)

    out = (
        da.sort_values("timestamp_utc")
          .drop_duplicates("timestamp_utc", keep="last")
          .set_index("timestamp_utc")
          .resample("15min")
          .ffill(limit=3)
          .reset_index()
    )

    return out


def fetch_wind_solar_forecast(delivery_day: date, process_type: str) -> pd.DataFrame:
    """
    A69 wind/solar forecast.

    process_type:
      A01 = day-ahead
      A40 = intraday
    """
    start_utc, end_utc = local_delivery_day_window_utc(delivery_day)

    params = {
        "documentType": "A69",
        "processType": process_type,
        "in_Domain": DE_LU_BZN,
        "periodStart": fmt_entsoe_time(start_utc),
        "periodEnd": fmt_entsoe_time(end_utc),
    }

    xml_text = request_entsoe(params)

    if xml_text is None:
        return pd.DataFrame()

    points = parse_timeseries_points(xml_text)

    if points.empty:
        return pd.DataFrame()

    suffix = "da" if process_type == "A01" else "id"

    wide_parts = []

    for psr, name in PSR_MAP.items():
        sub = points[points["psrType"] == psr].copy()

        if sub.empty:
            continue

        col = f"{name}_forecast_{suffix}_mw"

        tmp = (
            sub[["timestamp_utc", "quantity"]]
            .rename(columns={"quantity": col})
            .dropna(subset=[col])
            .drop_duplicates("timestamp_utc", keep="last")
        )

        wide_parts.append(tmp)

    if not wide_parts:
        return pd.DataFrame()

    out = wide_parts[0]

    for part in wide_parts[1:]:
        out = out.merge(part, on="timestamp_utc", how="outer")

    # Totals
    wind_cols = [
        c for c in [
            f"wind_offshore_forecast_{suffix}_mw",
            f"wind_onshore_forecast_{suffix}_mw",
        ]
        if c in out.columns
    ]

    if wind_cols:
        out[f"wind_total_forecast_{suffix}_mw"] = out[wind_cols].sum(axis=1, min_count=1)

    renewable_cols = [
        c for c in [
            f"solar_forecast_{suffix}_mw",
            f"wind_total_forecast_{suffix}_mw",
        ]
        if c in out.columns
    ]

    if renewable_cols:
        out[f"renewable_total_forecast_{suffix}_mw"] = out[renewable_cols].sum(axis=1, min_count=1)

    out["timestamp_utc"] = pd.to_datetime(out["timestamp_utc"], utc=True)
    return out.sort_values("timestamp_utc").reset_index(drop=True)


def fetch_total_generation_forecast_da(delivery_day: date) -> pd.DataFrame:
    """
    A71 total generation forecast, usually day-ahead only.
    """
    start_utc, end_utc = local_delivery_day_window_utc(delivery_day)

    param_variants = [
        {
            "documentType": "A71",
            "processType": "A01",
            "in_Domain": DE_LU_BZN,
            "periodStart": fmt_entsoe_time(start_utc),
            "periodEnd": fmt_entsoe_time(end_utc),
        },
        {
            "documentType": "A71",
            "processType": "A01",
            "outBiddingZone_Domain": DE_LU_BZN,
            "periodStart": fmt_entsoe_time(start_utc),
            "periodEnd": fmt_entsoe_time(end_utc),
        },
    ]

    for params in param_variants:
        xml_text = request_entsoe(params)

        if xml_text is None:
            continue

        points = parse_timeseries_points(xml_text)

        if points.empty:
            continue

        out = (
            points[["timestamp_utc", "quantity"]]
            .rename(columns={"quantity": "total_generation_forecast_da_mw"})
            .dropna(subset=["total_generation_forecast_da_mw"])
            .drop_duplicates("timestamp_utc", keep="last")
            .sort_values("timestamp_utc")
            .reset_index(drop=True)
        )

        if not out.empty:
            return out

    return pd.DataFrame(columns=["timestamp_utc", "total_generation_forecast_da_mw"])


def fetch_load_forecast_da(delivery_day: date) -> pd.DataFrame:
    """
    ENTSO-E total load day-ahead forecast.

    This usually corresponds to documentType A65, processType A01.
    Parameter naming differs by endpoint, so we try a few variants.
    """
    start_utc, end_utc = local_delivery_day_window_utc(delivery_day)

    param_variants = [
        {
            "documentType": "A65",
            "processType": "A01",
            "outBiddingZone_Domain": DE_LU_BZN,
            "periodStart": fmt_entsoe_time(start_utc),
            "periodEnd": fmt_entsoe_time(end_utc),
        },
        {
            "documentType": "A65",
            "processType": "A01",
            "in_Domain": DE_LU_BZN,
            "periodStart": fmt_entsoe_time(start_utc),
            "periodEnd": fmt_entsoe_time(end_utc),
        },
        {
            "documentType": "A65",
            "processType": "A01",
            "out_Domain": DE_LU_BZN,
            "periodStart": fmt_entsoe_time(start_utc),
            "periodEnd": fmt_entsoe_time(end_utc),
        },
    ]

    for params in param_variants:
        xml_text = request_entsoe(params)

        if xml_text is None:
            continue

        points = parse_timeseries_points(xml_text)

        if points.empty:
            continue

        out = (
            points[["timestamp_utc", "quantity"]]
            .rename(columns={"quantity": "load_forecast_da_mw"})
            .dropna(subset=["load_forecast_da_mw"])
            .drop_duplicates("timestamp_utc", keep="last")
            .sort_values("timestamp_utc")
            .reset_index(drop=True)
        )

        if not out.empty:
            return out

    return pd.DataFrame(columns=["timestamp_utc", "load_forecast_da_mw"])


def fetch_imbalance_volume_for_control_area(
    delivery_day: date,
    control_area: str,
    control_area_code: str,
) -> pd.DataFrame:
    """
    A86 total imbalance volumes per German control area.
    """
    start_utc, end_utc = local_delivery_day_window_utc(delivery_day)

    params = {
        "documentType": "A86",
        "controlArea_Domain": control_area_code,
        "periodStart": fmt_entsoe_time(start_utc),
        "periodEnd": fmt_entsoe_time(end_utc),
    }

    xml_text = request_entsoe(params)

    col = f"imbalance_volume_{control_area}_mw"

    if xml_text is None:
        return pd.DataFrame(columns=["timestamp_utc", col])

    points = parse_timeseries_points(xml_text)

    if points.empty:
        return pd.DataFrame(columns=["timestamp_utc", col])

    out = (
        points[["timestamp_utc", "quantity"]]
        .rename(columns={"quantity": col})
        .dropna(subset=[col])
        .drop_duplicates("timestamp_utc", keep="last")
        .sort_values("timestamp_utc")
        .reset_index(drop=True)
    )

    return out


def fetch_imbalance_volume_germany(delivery_day: date) -> pd.DataFrame:
    parts = []

    for name, code in GERMAN_CONTROL_AREAS.items():
        part = fetch_imbalance_volume_for_control_area(
            delivery_day=delivery_day,
            control_area=name,
            control_area_code=code,
        )

        if not part.empty:
            parts.append(part)

    if not parts:
        return pd.DataFrame()

    out = parts[0]

    for part in parts[1:]:
        out = out.merge(part, on="timestamp_utc", how="outer")

    imbalance_cols = [c for c in out.columns if c.startswith("imbalance_volume_") and c.endswith("_mw")]

    if imbalance_cols:
        out["imbalance_volume_germany_sum_mw"] = out[imbalance_cols].sum(axis=1, min_count=1)

    return out.sort_values("timestamp_utc").reset_index(drop=True)


# ============================================================
# FEATURE ASSEMBLER
# ============================================================

def add_forecast_revisions(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.copy()

    pairs = [
        ("solar", "solar_forecast_revision_id_minus_da_mw"),
        ("wind_onshore", "wind_onshore_forecast_revision_id_minus_da_mw"),
        ("wind_offshore", "wind_offshore_forecast_revision_id_minus_da_mw"),
        ("wind_total", "wind_total_forecast_revision_id_minus_da_mw"),
        ("renewable_total", "renewable_total_forecast_revision_id_minus_da_mw"),
    ]

    for base, out_col in pairs:
        da_col = f"{base}_forecast_da_mw"
        id_col = f"{base}_forecast_id_mw"

        if da_col in df.columns and id_col in df.columns:
            df[out_col] = df[id_col] - df[da_col]

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


def fetch_entsoe_features_for_day(delivery_day: date) -> pd.DataFrame:
    """
    Main public function used by the update pipeline.

    Returns 15-min master-compatible rows where possible.
    """
    parts = []

    print(f"ENTSO-E DA prices for {delivery_day}")
    da = fetch_day_ahead_prices(delivery_day)
    da_15 = expand_hourly_da_prices_to_15min(da)

    if not da_15.empty:
        parts.append(da_15)

    print(f"ENTSO-E wind/solar DA forecast for {delivery_day}")
    ws_da = fetch_wind_solar_forecast(delivery_day, process_type="A01")
    if not ws_da.empty:
        parts.append(ws_da)

    print(f"ENTSO-E wind/solar ID forecast for {delivery_day}")
    ws_id = fetch_wind_solar_forecast(delivery_day, process_type="A40")
    if not ws_id.empty:
        parts.append(ws_id)

    print(f"ENTSO-E total generation DA forecast for {delivery_day}")
    gen_da = fetch_total_generation_forecast_da(delivery_day)
    if not gen_da.empty:
        parts.append(gen_da)

    print(f"ENTSO-E load DA forecast for {delivery_day}")
    load_da = fetch_load_forecast_da(delivery_day)
    if not load_da.empty:
        parts.append(load_da)

    print(f"ENTSO-E imbalance volumes for {delivery_day}")
    imbalance = fetch_imbalance_volume_germany(delivery_day)
    if not imbalance.empty:
        parts.append(imbalance)

    if not parts:
        return pd.DataFrame()

    out = parts[0]

    for part in parts[1:]:
        out = out.merge(part, on="timestamp_utc", how="outer")

    out["timestamp_utc"] = pd.to_datetime(out["timestamp_utc"], utc=True)
    out = out.drop_duplicates("timestamp_utc", keep="last")
    out = out.sort_values("timestamp_utc").reset_index(drop=True)

    out = add_forecast_revisions(out)

    return out
