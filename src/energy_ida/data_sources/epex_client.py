import re
import time
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from playwright.sync_api import sync_playwright

from energy_ida.config import EPEX_MARKET_AREA, LOCAL_TZ


EPEX_COLUMNS = [
    "timestamp_utc",
    "epex_market_area",
    "epex_modality",
    "epex_sub_modality",
    "epex_auction",
    "epex_product",
    "buy_volume_mwh",
    "sell_volume_mwh",
    "volume_mwh",
    "price_eur_mwh",
    "low_price_eur_mwh",
    "high_price_eur_mwh",
    "last_price_eur_mwh",
    "weighted_avg_price_eur_mwh",
]


# ============================================================
# BASIC HELPERS
# ============================================================

def empty_epex_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=EPEX_COLUMNS)


def parse_number(text: Any) -> float:
    if text is None:
        return float("nan")

    if isinstance(text, (int, float)):
        return float(text)

    text = str(text).strip()

    if not text or text in {"-", "–", "nan", "None", "null"}:
        return float("nan")

    text = re.sub(r"[^\d,\.\-]+", "", text)

    if not text:
        return float("nan")

    return float(text.replace(" ", "").replace(",", ""))


def normalize_time_string(value: Any) -> Optional[str]:
    if value is None:
        return None

    s = str(value).strip()
    match = re.search(r"(\d{1,2}:\d{2})", s)

    if match:
        return match.group(1)

    return None


def local_delivery_time_to_utc(delivery_date: date, start_time: str) -> pd.Timestamp:
    start_time = str(start_time).strip()

    if "-" in start_time:
        start_time = start_time.split("-")[0].strip()

    local_naive = pd.Timestamp(f"{delivery_date} {start_time}")

    try:
        return (
            local_naive
            .tz_localize(LOCAL_TZ, ambiguous="infer", nonexistent="shift_forward")
            .tz_convert("UTC")
        )
    except Exception:
        return (
            local_naive
            .tz_localize(LOCAL_TZ, ambiguous="NaT", nonexistent="shift_forward")
            .tz_convert("UTC")
        )


# ============================================================
# URL BUILDER
# ============================================================

def build_epex_url(
    market_area: str,
    delivery_date: date,
    modality: str,
    auction: str = "",
    sub_modality: str = "",
    product: str = "",
    data_mode: str = "table",
    trading_date: Optional[date | str] = "auto",
) -> str:
    """
    Examples:

    IDA1:
        modality="Auction"
        sub_modality="Intraday"
        auction="IDA1"
        market_area="DE-LU"
        product=""

    Continuous 15-min:
        modality="Continuous"
        sub_modality=""
        auction=""
        market_area="DE"
        product="15"
        trading_date=""
    """
    delivery_str = delivery_date.strftime("%Y-%m-%d")

    if trading_date == "auto":
        if modality.lower() == "auction":
            trading_str = delivery_str
        else:
            trading_str = ""
    elif trading_date is None:
        trading_str = ""
    elif isinstance(trading_date, date):
        trading_str = trading_date.strftime("%Y-%m-%d")
    else:
        trading_str = str(trading_date)

    return (
        "https://www.epexspot.com/en/market-results"
        f"?market_area={market_area}"
        f"&auction={auction}"
        f"&trading_date={trading_str}"
        f"&delivery_date={delivery_str}"
        f"&underlying_year="
        f"&modality={modality}"
        f"&sub_modality={sub_modality}"
        f"&technology="
        f"&data_mode={data_mode}"
        f"&period="
        f"&production_period="
        f"&product={product}"
    )


# ============================================================
# DEBUG / PAGE HELPERS
# ============================================================

def ensure_debug_dir() -> Path:
    debug_dir = Path("debug_epex")
    debug_dir.mkdir(exist_ok=True)
    return debug_dir


def save_debug_artifacts(page, prefix: str) -> None:
    debug_dir = ensure_debug_dir()

    try:
        png_path = debug_dir / f"{prefix}.png"
        html_path = debug_dir / f"{prefix}.html"

        page.screenshot(path=str(png_path), full_page=True)

        with open(html_path, "w", encoding="utf-8") as f:
            f.write(page.content())

        print(f"Saved debug screenshot: {png_path}")
        print(f"Saved debug HTML: {html_path}")

    except Exception as exc:
        print(f"Could not save debug artifacts: {exc}")


def is_human_verification_page(page) -> bool:
    title = ""
    body = ""

    try:
        title = page.title().lower()
    except Exception:
        pass

    try:
        body = page.locator("body").inner_text(timeout=5000).lower()
    except Exception:
        pass

    markers = [
        "human verification",
        "confirm you are human",
        "security check",
        "verify you are human",
    ]

    return any(marker in title or marker in body for marker in markers)


def print_page_diagnostics(page) -> None:
    print("\nPage diagnostics:")
    print("Title:", page.title())
    print("URL:", page.url)

    selectors = [
        "table",
        "tr",
        "td",
        "div.js-table-values",
        "div.fixed-column",
        "div.js-md-widget",
        "div.fixed-column.js-table-times ul li.child",
        "div.fixed-column.js-table-times ul li.lvl-1",
        "div.fixed-column.js-table-times ul li.lvl-2",
        "div.js-table-values table tbody tr.child",
        "div.js-table-values table tbody tr.lvl-1",
        "div.js-table-values table tbody tr.lvl-2",
        "[class*='table']",
        "[class*='market']",
    ]

    for selector in selectors:
        try:
            print(f"  {selector}: {page.locator(selector).count()}")
        except Exception as exc:
            print(f"  {selector}: error {exc}")

    try:
        body_text = page.locator("body").inner_text(timeout=5000)
        print("\nBody text preview:")
        print(body_text[:2000])
    except Exception as exc:
        print("Could not get body text:", exc)


def accept_cookies_if_present(page, timeout_ms: int = 5000) -> None:
    selectors = [
        "#onetrust-accept-btn-handler",
        "button:has-text('Accept all')",
        "button:has-text('Accept All')",
        "button:has-text('Allow all')",
        "button:has-text('I agree')",
        "button:has-text('Accept')",
    ]

    for selector in selectors:
        try:
            button = page.locator(selector).first
            button.wait_for(state="visible", timeout=timeout_ms)
            button.click(timeout=timeout_ms)
            print("Accepted cookies.")
            page.wait_for_timeout(1000)
            return
        except Exception:
            continue

    print("No cookie banner accepted/needed.")


def click_data_disclaimer_if_present(page, timeout_ms: int = 5000) -> None:
    """
    Sometimes EPEX shows a data-use disclaimer/welcome overlay.
    Click the accept/access button if present.
    """
    selectors = [
        "#edit-acceptationbutton",
        "#edit-acceptationbuttonmobile",
        "button.data-use-acceptation-button",
        "input.data-use-acceptation-button",
    ]

    for selector in selectors:
        try:
            button = page.locator(selector).first
            button.wait_for(state="visible", timeout=timeout_ms)
            button.click(timeout=timeout_ms)
            print("Accepted EPEX data-use disclaimer.")
            page.wait_for_timeout(3000)
            return
        except Exception:
            continue


def wait_for_page_settled(page) -> None:
    try:
        page.wait_for_load_state("domcontentloaded", timeout=60000)
    except Exception:
        pass

    try:
        page.wait_for_load_state("networkidle", timeout=30000)
    except Exception:
        pass

    page.wait_for_timeout(5000)


# ============================================================
# TABLE EXTRACTION
# ============================================================

def extract_auction_rows(page) -> tuple[list[str], list[list[str]]]:
    """
    Auction pages such as IDA1/IDA2 usually have exactly one row per delivery interval:
      time rows: li.child
      data rows: tr.child
    """
    time_intervals = [
        txt.strip()
        for txt in page.locator("div.fixed-column.js-table-times ul li.child a").all_inner_texts()
        if txt.strip()
    ]

    row_locator = page.locator("div.js-table-values table tbody tr.child")
    row_count = row_locator.count()

    table_rows: list[list[str]] = []

    for i in range(row_count):
        row = row_locator.nth(i)
        cols = [txt.strip() for txt in row.locator("td").all_inner_texts()]
        if cols:
            table_rows.append(cols)

    return time_intervals, table_rows


def extract_continuous_rows(page) -> tuple[list[str], list[list[str]]]:
    """
    Continuous 15-min page is hierarchical:
      level 0: hour, e.g. 00 - 01
      level 1: 30-min, e.g. 00:00 - 00:30
      level 2: 15-min, e.g. 00:00 - 00:15

    We only want lvl-2 rows for product=15.
    """
    time_intervals = [
        txt.strip()
        for txt in page.locator("div.fixed-column.js-table-times ul li.lvl-2 a").all_inner_texts()
        if txt.strip()
    ]

    row_locator = page.locator("div.js-table-values table tbody tr.lvl-2")
    row_count = row_locator.count()

    table_rows: list[list[str]] = []

    for i in range(row_count):
        row = row_locator.nth(i)
        cols = [txt.strip() for txt in row.locator("td").all_inner_texts()]
        if cols:
            table_rows.append(cols)

    return time_intervals, table_rows


def parse_table_rows(
    time_intervals: list[str],
    table_rows: list[list[str]],
    delivery_date: date,
    market_area: str,
    modality: str,
    sub_modality: str,
    auction: str,
    product: str,
) -> pd.DataFrame:
    if not time_intervals or not table_rows:
        return empty_epex_frame()

    n = min(len(time_intervals), len(table_rows))

    if len(time_intervals) != len(table_rows):
        print(
            f"Warning: time interval count ({len(time_intervals)}) "
            f"!= row count ({len(table_rows)}). Using first {n} rows."
        )

    rows: List[Dict] = []
    is_continuous = modality.lower() == "continuous"

    for interval, cols in zip(time_intervals[:n], table_rows[:n]):
        start_time = normalize_time_string(interval)

        if start_time is None:
            continue

        timestamp_utc = local_delivery_time_to_utc(delivery_date, start_time)

        if pd.isna(timestamp_utc):
            continue

        if is_continuous:
            # Continuous table columns:
            # Low, High, Last, Weight Avg., Buy Volume, Sell Volume, Volume
            if len(cols) < 7:
                continue

            low_price = parse_number(cols[0])
            high_price = parse_number(cols[1])
            last_price = parse_number(cols[2])
            weighted_avg_price = parse_number(cols[3])
            buy_volume = parse_number(cols[4])
            sell_volume = parse_number(cols[5])
            volume = parse_number(cols[6])

            # Canonical continuous price: weighted average.
            price = weighted_avg_price

        else:
            # Auction table columns:
            # Buy Volume, Sell Volume, Volume, Price
            if len(cols) < 4:
                continue

            buy_volume = parse_number(cols[0])
            sell_volume = parse_number(cols[1])
            volume = parse_number(cols[2])
            price = parse_number(cols[3])

            low_price = float("nan")
            high_price = float("nan")
            last_price = float("nan")
            weighted_avg_price = float("nan")

        rows.append(
            {
                "timestamp_utc": timestamp_utc,
                "epex_market_area": market_area,
                "epex_modality": modality,
                "epex_sub_modality": sub_modality,
                "epex_auction": auction,
                "epex_product": product,
                "buy_volume_mwh": buy_volume,
                "sell_volume_mwh": sell_volume,
                "volume_mwh": volume,
                "price_eur_mwh": price,
                "low_price_eur_mwh": low_price,
                "high_price_eur_mwh": high_price,
                "last_price_eur_mwh": last_price,
                "weighted_avg_price_eur_mwh": weighted_avg_price,
            }
        )

    return pd.DataFrame(rows, columns=EPEX_COLUMNS)


def finalize_output(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return empty_epex_frame()

    df = df.copy()

    for col in EPEX_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA

    df = df[EPEX_COLUMNS]

    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp_utc", "price_eur_mwh"])

    df = (
        df
        .drop_duplicates(
            subset=[
                "timestamp_utc",
                "epex_market_area",
                "epex_modality",
                "epex_sub_modality",
                "epex_auction",
                "epex_product",
            ],
            keep="last",
        )
        .sort_values("timestamp_utc")
        .reset_index(drop=True)
    )

    return df


# ============================================================
# PUBLIC SCRAPER FUNCTIONS
# ============================================================

def scrape_epex_table(
    delivery_date: date,
    auction: str = "",
    sub_modality: str = "",
    market_area: str = EPEX_MARKET_AREA,
    modality: str = "Auction",
    product: str = "",
    trading_date: Optional[date | str] = "auto",
    headless: bool = True,
    browser_name: str = "firefox",
) -> pd.DataFrame:
    """
    Examples:

    IDA1:
        scrape_epex_table(
            delivery_date=date(2026, 5, 30),
            modality="Auction",
            sub_modality="Intraday",
            auction="IDA1",
            market_area="DE-LU",
        )

    Continuous 15-min:
        scrape_epex_table(
            delivery_date=date(2026, 5, 30),
            modality="Continuous",
            market_area="DE",
            product="15",
            auction="",
            sub_modality="",
            trading_date="",
        )
    """
    url = build_epex_url(
        market_area=market_area,
        delivery_date=delivery_date,
        modality=modality,
        auction=auction,
        sub_modality=sub_modality,
        product=product,
        trading_date=trading_date,
    )

    safe_auction = auction or "NA"
    safe_sub = sub_modality or "NA"
    safe_product = product or "NA"
    prefix = f"epex_debug_{market_area}_{modality}_{safe_sub}_{safe_auction}_{safe_product}_{delivery_date}"

    with sync_playwright() as p:
        if browser_name == "firefox":
            browser_type = p.firefox
        elif browser_name == "chromium":
            browser_type = p.chromium
        else:
            raise ValueError("browser_name must be 'firefox' or 'chromium'")

        browser = browser_type.launch(headless=headless)

        context = browser.new_context(
            viewport={"width": 1920, "height": 1080},
            locale="en-US",
            timezone_id=LOCAL_TZ,
            user_agent=(
                "Mozilla/5.0 (X11; Linux x86_64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Firefox/150.0 Safari/537.36"
            ),
        )

        page = context.new_page()
        page.set_default_timeout(60000)

        try:
            print("Opening:", url)
            page.goto(url, wait_until="domcontentloaded", timeout=90000)

            accept_cookies_if_present(page)
            click_data_disclaimer_if_present(page)
            wait_for_page_settled(page)

            if is_human_verification_page(page):
                print("EPEX human verification page detected. Returning empty dataframe; retry later.")
                save_debug_artifacts(page, prefix=prefix)
                return empty_epex_frame()

            if modality.lower() == "continuous":
                time_intervals, table_rows = extract_continuous_rows(page)
                print(
                    f"Continuous lvl-2 extraction: "
                    f"time_intervals={len(time_intervals)}, table_rows={len(table_rows)}"
                )
            else:
                time_intervals, table_rows = extract_auction_rows(page)
                print(
                    f"Auction extraction: "
                    f"time_intervals={len(time_intervals)}, table_rows={len(table_rows)}"
                )

            out = parse_table_rows(
                time_intervals=time_intervals,
                table_rows=table_rows,
                delivery_date=delivery_date,
                market_area=market_area,
                modality=modality,
                sub_modality=sub_modality,
                auction=auction,
                product=product,
            )

            out = finalize_output(out)

            if not out.empty:
                print(
                    f"EPEX scrape complete: "
                    f"{market_area} {modality} {sub_modality} {auction} product={product} "
                    f"{delivery_date}, rows={len(out)}"
                )
                return out

            print_page_diagnostics(page)
            save_debug_artifacts(page, prefix=prefix)
            print("No EPEX data parsed.")
            return empty_epex_frame()

        finally:
            context.close()
            browser.close()


def normalize_product_spec(spec: tuple | dict) -> dict:
    """
    Backward compatible product spec handling.

    Old form:
        ("Intraday", "IDA1")

    New form:
        {
            "modality": "Continuous",
            "sub_modality": "",
            "auction": "",
            "market_area": "DE",
            "product": "15",
            "trading_date": "",
        }
    """
    if isinstance(spec, tuple):
        sub_modality, auction = spec

        return {
            "modality": "Auction",
            "sub_modality": sub_modality,
            "auction": auction,
            "market_area": EPEX_MARKET_AREA,
            "product": "",
            "trading_date": "auto",
        }

    if isinstance(spec, dict):
        return {
            "modality": spec.get("modality", "Auction"),
            "sub_modality": spec.get("sub_modality", ""),
            "auction": spec.get("auction", ""),
            "market_area": spec.get("market_area", EPEX_MARKET_AREA),
            "product": spec.get("product", ""),
            "trading_date": spec.get("trading_date", "auto"),
        }

    raise TypeError("Product spec must be tuple or dict.")


def fetch_epex_germany_for_dates(
    start_date: date,
    end_date: date,
    products: list[tuple[str, str] | dict],
    headless: bool = True,
    browser_name: str = "firefox",
    pause_seconds: float = 60.0,
) -> pd.DataFrame:
    """
    Inclusive date range.

    Example products:

        products = [
            ("Intraday", "IDA1"),
            ("Intraday", "IDA2"),
            ("Intraday", "IDA3"),
            {
                "modality": "Continuous",
                "sub_modality": "",
                "auction": "",
                "market_area": "DE",
                "product": "15",
                "trading_date": "",
            },
        ]
    """
    all_parts = []

    normalized_products = [normalize_product_spec(p) for p in products]

    for delivery_day in pd.date_range(start_date, end_date, freq="D"):
        d = delivery_day.date()

        for spec in normalized_products:
            print(
                f"EPEX {spec['market_area']} | {spec['modality']} "
                f"{spec['sub_modality']} {spec['auction']} product={spec['product']} | {d}"
            )

            try:
                part = scrape_epex_table(
                    delivery_date=d,
                    auction=spec["auction"],
                    sub_modality=spec["sub_modality"],
                    market_area=spec["market_area"],
                    modality=spec["modality"],
                    product=spec["product"],
                    trading_date=spec["trading_date"],
                    headless=headless,
                    browser_name=browser_name,
                )

                if not part.empty:
                    all_parts.append(part)
                else:
                    print(f"No rows returned for {spec} {d}")

            except Exception as exc:
                print(f"Missing/not ready/failed for {spec} {d}: {exc}")

            print(f"Sleeping {pause_seconds:.1f}s before next EPEX request...")
            time.sleep(pause_seconds)

    if not all_parts:
        return empty_epex_frame()

    out = pd.concat(all_parts, ignore_index=True)

    out = (
        out
        .drop_duplicates(
            subset=[
                "timestamp_utc",
                "epex_market_area",
                "epex_modality",
                "epex_sub_modality",
                "epex_auction",
                "epex_product",
            ],
            keep="last",
        )
        .sort_values(
            [
                "timestamp_utc",
                "epex_market_area",
                "epex_modality",
                "epex_sub_modality",
                "epex_auction",
                "epex_product",
            ]
        )
        .reset_index(drop=True)
    )

    return out
