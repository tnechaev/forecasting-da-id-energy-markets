import pandas as pd
from pathlib import Path


def load_master(path: Path) -> pd.DataFrame:
    if path.exists():
        df = pd.read_parquet(path)
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
        return df.sort_values("timestamp_utc").reset_index(drop=True)

    return pd.DataFrame()


def upsert_by_timestamp(existing: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """
    Merge existing and new data by timestamp_utc.

    Important behavior:
    - no duplicate timestamps in final file
    - new non-null values update existing values
    - existing non-null values are preserved when new value is null
    - new columns are added automatically
    """
    if existing.empty and new.empty:
        return pd.DataFrame()

    if existing.empty:
        out = new.copy()
        out["timestamp_utc"] = pd.to_datetime(out["timestamp_utc"], utc=True)
        out = out.sort_values("timestamp_utc")
        out = out.groupby("timestamp_utc", as_index=False).last()
        return out.reset_index(drop=True)

    if new.empty:
        out = existing.copy()
        out["timestamp_utc"] = pd.to_datetime(out["timestamp_utc"], utc=True)
        out = out.sort_values("timestamp_utc")
        out = out.groupby("timestamp_utc", as_index=False).last()
        return out.reset_index(drop=True)

    existing = existing.copy()
    new = new.copy()

    existing["timestamp_utc"] = pd.to_datetime(existing["timestamp_utc"], utc=True)
    new["timestamp_utc"] = pd.to_datetime(new["timestamp_utc"], utc=True)

    # Remove duplicate timestamps inside each frame first.
    # For existing, keep last stored version.
    existing = (
        existing.sort_values("timestamp_utc")
        .groupby("timestamp_utc", as_index=False)
        .last()
    )

    # For new, combine duplicates using last non-null value per column.
    new = (
        new.sort_values("timestamp_utc")
        .groupby("timestamp_utc", as_index=False)
        .last()
    )

    existing = existing.set_index("timestamp_utc")
    new = new.set_index("timestamp_utc")

    # Union of timestamps and columns.
    all_index = existing.index.union(new.index).sort_values()
    all_columns = existing.columns.union(new.columns)

    existing = existing.reindex(index=all_index, columns=all_columns)
    new = new.reindex(index=all_index, columns=all_columns)

    # Key behavior:
    # Use new value if available, otherwise keep existing.
    combined = existing.combine_first(new)

    # Wait: combine_first keeps existing over new.
    # We want new over existing where new is non-null.
    combined = new.combine_first(existing)

    combined = combined.reset_index()
    combined = combined.sort_values("timestamp_utc").reset_index(drop=True)

    return combined


def save_master(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = df.copy()
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    df = df.sort_values("timestamp_utc").reset_index(drop=True)
    df.to_parquet(path, index=False)


def upsert_save_master(existing: pd.DataFrame, new: pd.DataFrame, path: Path) -> pd.DataFrame:
    combined = upsert_by_timestamp(existing, new)
    save_master(combined, path)
    return combined
