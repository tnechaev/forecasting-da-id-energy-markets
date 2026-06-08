import io
import json
import os
import time
import zipfile
import xml.etree.ElementTree as ET
from datetime import date
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
import requests

from energy_ida.config import LOCAL_TZ

from dotenv import load_dotenv

load_dotenv()


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

ENTSOE_AUDIT_LOG = Path("data/audit/entsoe_document_log.parquet")

AUDIT_KEY_COLS = [
    "source_name",
    "documentType",
    "processType",
    "revisionNumber",
    "createdDateTime",
    "doc_period_start_utc",
    "doc_period_end_utc",
    "timeseries",
    "businessType",
    "psrType",
    "inBiddingZone",
    "outBiddingZone",
    "resolution",
    "period_start_utc",
    "period_end_utc",
]


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

    if resolution.startswith("PT") and resolution.endswith("M"):
        return pd.Timedelta(minutes=int(resolution[2:-1]))
    if resolution.startswith("PT") and resolution.endswith("H"):
        return pd.Timedelta(hours=int(resolution[2:-1]))

    raise ValueError(f"Unsupported resolution: {resolution}")


def parse_utc_or_nat(value: Optional[str]) -> pd.Timestamp:
    if not value:
        return pd.NaT

    try:
        return pd.Timestamp(value).tz_convert("UTC")
    except Exception:
        try:
            return pd.Timestamp(value).tz_localize("UTC")
        except Exception:
            return pd.NaT


def extract_xml_payload(response: requests.Response) -> str:
    content = response.content

    if content[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            xml_names = [name for name in zf.namelist() if name.lower().endswith(".xml")]

            if not xml_names:
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


def params_without_token(params: dict) -> dict:
    return {k: v for k, v in params.items() if k != "securityToken"}


# ============================================================
# METADATA LOGGING
# ============================================================

def _top_level_time_interval(root: ET.Element) -> tuple[pd.Timestamp, pd.Timestamp]:
    interval = next(iter_children(root, "time_Period.timeInterval"), None)

    if interval is None:
        interval = next(iter_desc(root, "timeInterval"), None)

    if interval is None:
        return pd.NaT, pd.NaT

    start = parse_utc_or_nat(child_text(interval, "start"))
    end = parse_utc_or_nat(child_text(interval, "end"))

    return start, end


def extract_entsoe_metadata_rows(
    xml_text: str,
    params: dict,
    source_name: str,
    query_time_utc: pd.Timestamp,
) -> pd.DataFrame:
    """
    Extract document/time-series metadata from an ENTSO-E XML response.

    This does not parse values for modelling. It records document provenance:
      - createdDateTime
      - revisionNumber
      - documentType/processType
      - delivery period
      - TimeSeries businessType/psrType/resolution
      - query time and API parameters
    """
    try:
        root = ET.fromstring(xml_text)
    except Exception as exc:
        return pd.DataFrame(
            [
                {
                    "source_name": source_name,
                    "query_time_utc": query_time_utc,
                    "metadata_parse_error": str(exc),
                    "api_params_json": json.dumps(params_without_token(params), sort_keys=True),
                }
            ]
        )

    if strip_ns(root.tag) == "Acknowledgement_MarketDocument":
        return pd.DataFrame()

    document_type = child_text(root, "type")
    process_type = child_text(root, "process.processType")
    revision_number = child_text(root, "revisionNumber")
    created_datetime = parse_utc_or_nat(child_text(root, "createdDateTime"))

    doc_start, doc_end = _top_level_time_interval(root)

    query_period_start = parse_utc_or_nat(params.get("periodStart"))
    query_period_end = parse_utc_or_nat(params.get("periodEnd"))

    rows = []

    for ts_idx, ts in enumerate(iter_desc(root, "TimeSeries"), start=1):
        business_type = child_text(ts, "businessType")
        ts_process_type = child_text(ts, "process.processType")

        effective_process_type = process_type or ts_process_type

        in_zone = (
            child_text(ts, "in_Domain.mRID")
            or child_text(ts, "inBiddingZone_Domain.mRID")
            or params.get("in_Domain")
            or params.get("inBiddingZone_Domain")
        )

        out_zone = (
            child_text(ts, "out_Domain.mRID")
            or child_text(ts, "outBiddingZone_Domain.mRID")
            or params.get("out_Domain")
            or params.get("outBiddingZone_Domain")
        )

        psr_type = None
        mkt_psr = next(iter_desc(ts, "MktPSRType"), None)
        if mkt_psr is not None:
            psr_type = child_text(mkt_psr, "psrType")

        if psr_type is None:
            psr_type = first_desc_text(ts, "psrType")

        for period_idx, period in enumerate(iter_desc(ts, "Period"), start=1):
            interval = next(iter_children(period, "timeInterval"), None)

            period_start = pd.NaT
            period_end = pd.NaT

            if interval is not None:
                period_start = parse_utc_or_nat(child_text(interval, "start"))
                period_end = parse_utc_or_nat(child_text(interval, "end"))

            resolution_text = child_text(period, "resolution")
            n_points = sum(1 for _ in iter_children(period, "Point"))

            rows.append(
                {
                    "source_name": source_name,
                    "query_time_utc": query_time_utc,
                    "first_seen_utc": query_time_utc,
                    "last_seen_utc": query_time_utc,
                    "seen_count": 1,
                    "documentType": document_type,
                    "processType": effective_process_type,
                    "revisionNumber": revision_number,
                    "createdDateTime": created_datetime,
                    "doc_period_start_utc": doc_start,
                    "doc_period_end_utc": doc_end,
                    "timeseries": ts_idx,
                    "period": period_idx,
                    "businessType": business_type,
                    "psrType": psr_type,
                    "inBiddingZone": in_zone,
                    "outBiddingZone": out_zone,
                    "resolution": resolution_text,
                    "n_points": n_points,
                    "period_start_utc": period_start,
                    "period_end_utc": period_end,
                    "query_period_start_utc": query_period_start,
                    "query_period_end_utc": query_period_end,
                    "api_params_json": json.dumps(params_without_token(params), sort_keys=True),
                    "metadata_parse_error": pd.NA,
                }
            )

    if not rows:
        rows.append(
            {
                "source_name": source_name,
                "query_time_utc": query_time_utc,
                "first_seen_utc": query_time_utc,
                "last_seen_utc": query_time_utc,
                "seen_count": 1,
                "documentType": document_type,
                "processType": process_type,
                "revisionNumber": revision_number,
                "createdDateTime": created_datetime,
                "doc_period_start_utc": doc_start,
                "doc_period_end_utc": doc_end,
                "timeseries": pd.NA,
                "period": pd.NA,
                "businessType": pd.NA,
                "psrType": pd.NA,
                "inBiddingZone": params.get("in_Domain") or params.get("inBiddingZone_Domain"),
                "outBiddingZone": params.get("out_Domain") or params.get("outBiddingZone_Domain"),
                "resolution": pd.NA,
                "n_points": 0,
                "period_start_utc": pd.NaT,
                "period_end_utc": pd.NaT,
                "query_period_start_utc": query_period_start,
                "query_period_end_utc": query_period_end,
                "api_params_json": json.dumps(params_without_token(params), sort_keys=True),
                "metadata_parse_error": pd.NA,
            }
        )

    out = pd.DataFrame(rows)

    datetime_cols = [
        "query_time_utc",
        "first_seen_utc",
        "last_seen_utc",
        "createdDateTime",
        "doc_period_start_utc",
        "doc_period_end_utc",
        "period_start_utc",
        "period_end_utc",
        "query_period_start_utc",
        "query_period_end_utc",
    ]

    for c in datetime_cols:
        if c in out.columns:
            out[c] = pd.to_datetime(out[c], utc=True, errors="coerce")

    return out


def upsert_entsoe_metadata_log(
    new_rows: pd.DataFrame,
    log_path: Path = ENTSOE_AUDIT_LOG,
) -> None:
    if new_rows.empty:
        return

    log_path.parent.mkdir(parents=True, exist_ok=True)

    new_rows = new_rows.copy()

    for c in AUDIT_KEY_COLS:
        if c not in new_rows.columns:
            new_rows[c] = pd.NA

    datetime_cols = [
        "query_time_utc",
        "first_seen_utc",
        "last_seen_utc",
        "createdDateTime",
        "doc_period_start_utc",
        "doc_period_end_utc",
        "period_start_utc",
        "period_end_utc",
        "query_period_start_utc",
        "query_period_end_utc",
    ]

    for c in datetime_cols:
        if c in new_rows.columns:
            new_rows[c] = pd.to_datetime(new_rows[c], utc=True, errors="coerce")

    if log_path.exists():
        old = pd.read_parquet(log_path)
    else:
        old = pd.DataFrame(columns=new_rows.columns)

    for c in new_rows.columns:
        if c not in old.columns:
            old[c] = pd.NA

    for c in old.columns:
        if c not in new_rows.columns:
            new_rows[c] = pd.NA

    combined = pd.concat([old, new_rows[old.columns]], ignore_index=True)

    for c in datetime_cols:
        if c in combined.columns:
            combined[c] = pd.to_datetime(combined[c], utc=True, errors="coerce")

    # Convert key columns to strings for stable grouping with NaNs/NaTs.
    key_tmp_cols = []
    for c in AUDIT_KEY_COLS:
        tmp = f"__key_{c}"
        key_tmp_cols.append(tmp)

        if c in combined.columns:
            if pd.api.types.is_datetime64_any_dtype(combined[c]):
                combined[tmp] = combined[c].dt.strftime("%Y-%m-%dT%H:%M:%S%z").fillna("<NA>")
            else:
                combined[tmp] = combined[c].astype("string").fillna("<NA>")
        else:
            combined[tmp] = "<NA>"

    agg_spec = {}

    for c in combined.columns:
        if c in key_tmp_cols:
            continue

        if c == "first_seen_utc":
            agg_spec[c] = "min"
        elif c == "last_seen_utc":
            agg_spec[c] = "max"
        elif c == "query_time_utc":
            agg_spec[c] = "max"
        elif c == "seen_count":
            agg_spec[c] = "sum"
        else:
            agg_spec[c] = "last"

    deduped = (
        combined.groupby(key_tmp_cols, dropna=False, as_index=False)
        .agg(agg_spec)
    )

    deduped = deduped.drop(columns=key_tmp_cols)

    # Nice ordering.
    preferred_cols = [
        "source_name",
        "query_time_utc",
        "first_seen_utc",
        "last_seen_utc",
        "seen_count",
        "documentType",
        "processType",
        "revisionNumber",
        "createdDateTime",
        "doc_period_start_utc",
        "doc_period_end_utc",
        "timeseries",
        "period",
        "businessType",
        "psrType",
        "inBiddingZone",
        "outBiddingZone",
        "resolution",
        "n_points",
        "period_start_utc",
        "period_end_utc",
        "query_period_start_utc",
        "query_period_end_utc",
        "api_params_json",
        "metadata_parse_error",
    ]

    ordered = [c for c in preferred_cols if c in deduped.columns]
    rest = [c for c in deduped.columns if c not in ordered]
    deduped = deduped[ordered + rest]

    deduped = deduped.sort_values(
        [c for c in ["query_time_utc", "source_name", "documentType", "processType"] if c in deduped.columns]
    ).reset_index(drop=True)

    deduped.to_parquet(log_path, index=False)


def log_entsoe_metadata_if_possible(
    xml_text: str,
    params: dict,
    source_name: str,
    query_time_utc: pd.Timestamp,
) -> None:
    try:
        rows = extract_entsoe_metadata_rows(
            xml_text=xml_text,
            params=params,
            source_name=source_name,
            query_time_utc=query_time_utc,
        )
        upsert_entsoe_metadata_log(rows, ENTSOE_AUDIT_LOG)
        print(f"Saved ENTSO-E metadata log rows: {len(rows)} -> {ENTSOE_AUDIT_LOG}")
    except Exception as exc:
        print(f"WARNING: Could not save ENTSO-E metadata log for {source_name}: {exc}")


def request_entsoe(
    params: dict,
    sleep_seconds: float = 0.2,
    source_name: str = "entsoe_unknown",
    log_metadata: bool = True,
) -> Optional[str]:
    api_key = get_entsoe_api_key()

    full_params = dict(params)
    full_params["securityToken"] = api_key

    query_time_utc = pd.Timestamp.utcnow()

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

    if log_metadata:
        log_entsoe_metadata_if_possible(
            xml_text=xml_text,
            params=params,
            source_name=source_name,
            query_time_utc=query_time_utc,
        )

    return xml_text


# ============================================================
# XML VALUE PARSING
# ============================================================

def parse_timeseries_points(xml_text: str) -> pd.DataFrame:
    root = ET.fromstring(xml_text)

    rows = []

    document_type = child_text(root, "type")
    process_type = child_text(root, "process.processType")
    created_datetime = child_text(root, "createdDateTime")
    revision_number = child_text(root, "revisionNumber")

    for ts_idx, ts in enumerate(iter_desc(root, "TimeSeries"), start=1):
        business_type = child_text(ts, "businessType")
        ts_process_type = child_text(ts, "process.processType")

        in_zone = child_text(ts, "in_Domain.mRID") or child_text(ts, "inBiddingZone_Domain.mRID")
        out_zone = child_text(ts, "out_Domain.mRID") or child_text(ts, "outBiddingZone_Domain.mRID")

        psr_type = None
        mkt_psr = next(iter_desc(ts, "MktPSRType"), None)
        if mkt_psr is not None:
            psr_type = child_text(mkt_psr, "psrType")

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
                        "revisionNumber": revision_number,
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
    start_utc, end_utc = local_delivery_day_window_utc(delivery_day)

    params = {
        "documentType": "A44",
        "in_Domain": DE_LU_BZN,
        "out_Domain": DE_LU_BZN,
        "periodStart": fmt_entsoe_time(start_utc),
        "periodEnd": fmt_entsoe_time(end_utc),
    }

    xml_text = request_entsoe(params, source_name="day_ahead_prices_A44")

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


def ensure_da_prices_15min(da: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure DA prices are on 15-min grid.

    Before SDAC 15-min go-live, DA prices may be hourly and need expansion.
    After SDAC 15-min go-live, ENTSO-E A44 may already return 96 quarter-hour rows.
    """
    if da.empty:
        return da

    da = da.copy()
    da["timestamp_utc"] = pd.to_datetime(da["timestamp_utc"], utc=True)
    da = da.sort_values("timestamp_utc").drop_duplicates("timestamp_utc", keep="last")

    deltas = da["timestamp_utc"].diff().dropna()

    if deltas.empty:
        return da.reset_index(drop=True)

    most_common_delta = deltas.mode().iloc[0]

    if most_common_delta == pd.Timedelta(minutes=15):
        return da.reset_index(drop=True)

    if most_common_delta == pd.Timedelta(hours=1):
        return (
            da.set_index("timestamp_utc")
              .resample("15min")
              .ffill(limit=3)
              .reset_index()
        )

    print(f"Warning: unexpected DA price resolution: {most_common_delta}. Returning unchanged.")
    return da.reset_index(drop=True)


# Backward-compatible alias, in case other code imports the old name.
def expand_hourly_da_prices_to_15min(da: pd.DataFrame) -> pd.DataFrame:
    return ensure_da_prices_15min(da)


def fetch_wind_solar_forecast(delivery_day: date, process_type: str) -> pd.DataFrame:
    start_utc, end_utc = local_delivery_day_window_utc(delivery_day)

    params = {
        "documentType": "A69",
        "processType": process_type,
        "in_Domain": DE_LU_BZN,
        "periodStart": fmt_entsoe_time(start_utc),
        "periodEnd": fmt_entsoe_time(end_utc),
    }

    xml_text = request_entsoe(
        params,
        source_name=f"wind_solar_forecast_A69_{process_type}",
    )

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
        xml_text = request_entsoe(
            params,
            source_name="total_generation_forecast_DA_A71_A01",
        )

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
        xml_text = request_entsoe(
            params,
            source_name="load_forecast_DA_A65_A01",
        )

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
    start_utc, end_utc = local_delivery_day_window_utc(delivery_day)

    params = {
        "documentType": "A86",
        "controlArea_Domain": control_area_code,
        "periodStart": fmt_entsoe_time(start_utc),
        "periodEnd": fmt_entsoe_time(end_utc),
    }

    xml_text = request_entsoe(
        params,
        source_name=f"imbalance_volume_A86_{control_area}",
    )

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
    parts = []

    print(f"ENTSO-E DA prices for {delivery_day}")
    da = fetch_day_ahead_prices(delivery_day)
    da_15 = ensure_da_prices_15min(da)

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
