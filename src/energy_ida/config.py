from pathlib import Path

# ============================================================
# TIME / MARKET CONSTANTS
# ============================================================

LOCAL_TZ = "Europe/Berlin"

ENTSOE_DE_LU = "10Y1001A1001A82H"

EPEX_AUCTION_MARKET_AREA = "DE-LU"
EPEX_CONTINUOUS_MARKET_AREA = "DE"

# Backward-compatible default
EPEX_MARKET_AREA = EPEX_AUCTION_MARKET_AREA


# ============================================================
# PATHS
# ============================================================

DATA_DIR = Path("data")
MASTER_DIR = DATA_DIR / "master"
RAW_DIR = DATA_DIR / "raw"
FEATURES_DIR = DATA_DIR / "features"
FORECASTS_DIR = DATA_DIR / "forecasts"
REPORTS_DIR = DATA_DIR / "reports"

MASTER_FILE = MASTER_DIR / "germany_ida_master_15min.parquet"

for path in [
    DATA_DIR,
    MASTER_DIR,
    RAW_DIR,
    FEATURES_DIR,
    FORECASTS_DIR,
    REPORTS_DIR,
]:
    path.mkdir(parents=True, exist_ok=True)


# ============================================================
# EPEX PRODUCT GROUPS
# ============================================================

EPEX_PRODUCT_GROUPS = {
    "ida1": [
        {
            "modality": "Auction",
            "sub_modality": "Intraday",
            "auction": "IDA1",
            "market_area": "DE-LU",
            "product": "",
            "trading_date": "auto",
        }
    ],
    "ida2": [
        {
            "modality": "Auction",
            "sub_modality": "Intraday",
            "auction": "IDA2",
            "market_area": "DE-LU",
            "product": "",
            "trading_date": "auto",
        }
    ],
    "ida3": [
        {
            "modality": "Auction",
            "sub_modality": "Intraday",
            "auction": "IDA3",
            "market_area": "DE-LU",
            "product": "",
            "trading_date": "auto",
        }
    ],
    "continuous_15": [
        {
            "modality": "Continuous",
            "sub_modality": "",
            "auction": "",
            "market_area": "DE",
            "product": "15",
            "trading_date": "",
        }
    ],
}
