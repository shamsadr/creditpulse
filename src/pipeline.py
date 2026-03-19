"""
pipeline.py — Core ETL logic for CreditPulse.

Four stages:
  1. fetch_series()     — Pull raw FRED data
  2. validate_data()    — Check for gaps, types, expected shape
  3. engineer_features()— Lags, rolling stats, recession flag
  4. export_data()      — Write CSV + metadata JSON
"""

import json
import os
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# FRED series to pull
# ---------------------------------------------------------------------------

FRED_SERIES: dict[str, str] = {
    "delinquency_credit_card": "DRCCLACBS",  # CC delinquency rate (%)
    "delinquency_commercial": "DRBLACBS",  # C&I loan delinquency rate (%)
    "chargeoff_credit_card": "CORCCACBS",  # CC charge-off rate (%)
    "unemployment_rate": "UNRATE",  # Unemployment rate (%)
    "credit_spread_baa": "BAA10Y",  # BAA-10yr spread (pp)
    "fed_funds_rate": "FEDFUNDS",  # Federal funds rate (%)
    "recession_indicator": "USREC",  # NBER recession dummy (0/1)
}

MAX_GAP_MONTHS = 3  # gaps longer than this get flagged in metadata


# ---------------------------------------------------------------------------
# Stage 1: Fetch
# ---------------------------------------------------------------------------


def fetch_series(
    start: str = "2000-01-01",
    end: Optional[str] = None,
    api_key: Optional[str] = None,
) -> pd.DataFrame:
    """
    Pull all FRED series and align to a common monthly DatetimeIndex.

    Args:
        start:   First date to pull, ISO format string.
        end:     Last date to pull (defaults to today).
        api_key: FRED API key. Falls back to FRED_API_KEY env var.

    Returns:
        Raw DataFrame with one column per indicator, monthly frequency.

    Raises:
        ValueError: If API key is missing.
        ImportError: If fredapi is not installed.
    """
    try:
        from fredapi import Fred
    except ImportError as exc:
        raise ImportError("fredapi not installed. Run: pip install fredapi") from exc

    key = api_key or os.environ.get("FRED_API_KEY")
    if not key:
        raise ValueError(
            "FRED API key required. Set FRED_API_KEY env var or pass --api-key.\n"
            "Get a free key at: https://fred.stlouisfed.org/docs/api/api_key.html"
        )

    end = end or datetime.today().strftime("%Y-%m-%d")
    fred = Fred(api_key=key)

    frames: dict[str, pd.Series] = {}
    for name, series_id in FRED_SERIES.items():
        print(f"  Pulling {name} ({series_id})...")
        raw = fred.get_series(series_id, observation_start=start, observation_end=end)
        # Resample to month-end to standardize frequency
        frames[name] = raw.resample("ME").last().ffill()

    df = pd.DataFrame(frames)
    df.index.name = "date"
    df.index = pd.to_datetime(df.index)
    return df


# ---------------------------------------------------------------------------
# Stage 2: Validate
# ---------------------------------------------------------------------------


def validate_data(df: pd.DataFrame) -> dict:
    """
    Check data quality and return a validation report.

    Args:
        df: Raw DataFrame from fetch_series().

    Returns:
        Dictionary with missing value counts, gap flags, and date range.
    """
    assert not df.empty, "DataFrame is empty — check FRED pull."
    assert isinstance(df.index, pd.DatetimeIndex), "Index must be DatetimeIndex."

    report: dict = {
        "date_range": {
            "start": str(df.index.min().date()),
            "end": str(df.index.max().date()),
            "n_months": len(df),
        },
        "missing_values": {},
        "long_gaps_flagged": {},
    }

    for col in df.columns:
        n_missing = df[col].isna().sum()
        report["missing_values"][col] = int(n_missing)

        # Detect runs of consecutive NaNs longer than MAX_GAP_MONTHS
        is_nan = df[col].isna()
        run_lengths = (
            is_nan.groupby((is_nan != is_nan.shift()).cumsum())
            .sum()
            .where(is_nan.groupby((is_nan != is_nan.shift()).cumsum()).first())
            .dropna()
        )
        long_gaps = run_lengths[run_lengths > MAX_GAP_MONTHS]
        if not long_gaps.empty:
            report["long_gaps_flagged"][col] = int(long_gaps.max())

    return report


# ---------------------------------------------------------------------------
# Stage 3: Feature Engineering
# ---------------------------------------------------------------------------


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build modeling-ready features from raw FRED series.

    Features added per indicator:
      - _lag1, _lag3, _lag6   : 1-, 3-, 6-month lags
      - _rolling3_mean        : 3-month rolling average (smoothed trend)
      - _mom                  : month-over-month change (first difference)

    Additional features:
      - recession_flag        : binary, from USREC
      - credit_stress_index   : equal-weight composite of delinquency + spread + unemployment

    Args:
        df: Raw/validated DataFrame.

    Returns:
        DataFrame with original columns plus engineered features.
        Drops recession_indicator (replaced by recession_flag).
    """
    feat = df.copy()

    # Fill short gaps (forward then backward fill)
    feat = feat.ffill(limit=MAX_GAP_MONTHS).bfill(limit=MAX_GAP_MONTHS)

    # Recession flag (clean binary from USREC)
    feat["recession_flag"] = feat["recession_indicator"].fillna(0).astype(int)
    feat = feat.drop(columns=["recession_indicator"])

    # Lag and rolling features for each core indicator
    core_cols = [
        "delinquency_credit_card",
        "delinquency_commercial",
        "chargeoff_credit_card",
        "unemployment_rate",
        "credit_spread_baa",
        "fed_funds_rate",
    ]

    for col in core_cols:
        if col not in feat.columns:
            continue
        feat[f"{col}_lag1"] = feat[col].shift(1)
        feat[f"{col}_lag3"] = feat[col].shift(3)
        feat[f"{col}_lag6"] = feat[col].shift(6)
        feat[f"{col}_rolling3_mean"] = feat[col].rolling(3).mean()
        feat[f"{col}_mom"] = feat[col].diff(1)

    # Credit stress index: normalized average of 3 key stress indicators
    stress_cols = [
        "delinquency_credit_card",
        "credit_spread_baa",
        "unemployment_rate",
    ]
    available = [c for c in stress_cols if c in feat.columns]
    if available:
        normalized = feat[available].apply(lambda x: (x - x.mean()) / x.std(), axis=0)
        feat["credit_stress_index"] = normalized.mean(axis=1)

    # Drop rows that are all NaN (pre-data period)
    feat = feat.dropna(how="all")

    return feat


# ---------------------------------------------------------------------------
# Stage 4: Export
# ---------------------------------------------------------------------------


def export_data(
    df: pd.DataFrame,
    validation_report: dict,
    output_dir: str = "data",
) -> tuple[Path, Path]:
    """
    Write the feature DataFrame and metadata JSON to disk.

    Args:
        df:                Feature-engineered DataFrame.
        validation_report: Output from validate_data().
        output_dir:        Directory to write outputs.

    Returns:
        Tuple of (csv_path, metadata_path).
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    csv_path = out / "creditpulse_features.csv"
    meta_path = out / "metadata.json"

    df.to_csv(csv_path)

    metadata = {
        "pulled_at": datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "n_rows": len(df),
        "n_features": len(df.columns),
        "columns": list(df.columns),
        "validation": validation_report,
        "series_descriptions": {k: v for k, v in FRED_SERIES.items()},
    }
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return csv_path, meta_path


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------


def summarize_results(
    df: pd.DataFrame,
    validation_report: dict,
    csv_path: Path,
    meta_path: Path,
) -> None:
    """
    Print a human-readable pipeline summary to stdout.

    Args:
        df:                Feature-engineered DataFrame.
        validation_report: Validation report dict.
        csv_path:          Path where CSV was saved.
        meta_path:         Path where metadata JSON was saved.
    """
    dr = validation_report["date_range"]
    print("\n" + "=" * 60)
    print("  CreditPulse — Pipeline Complete")
    print("=" * 60)
    print(f"  Date range : {dr['start']} → {dr['end']}  ({dr['n_months']} months)")
    print(f"  Features   : {len(df.columns)} columns")
    print(f"  Rows       : {len(df)}")
    print(f"\n  Output files:")
    print(f"    {csv_path}")
    print(f"    {meta_path}")

    print("\n  Missing values (raw series):")
    for col, n in validation_report["missing_values"].items():
        flag = (
            " ⚠️  LONG GAP"
            if col in validation_report.get("long_gaps_flagged", {})
            else ""
        )
        print(f"    {col:<35} {n:>3} missing{flag}")

    print("\n  Correlation with credit_stress_index (top 5):")
    if "credit_stress_index" in df.columns:
        corr = (
            df.corr()["credit_stress_index"]
            .drop("credit_stress_index")
            .abs()
            .sort_values(ascending=False)
            .head(5)
        )
        for feat, val in corr.items():
            print(f"    {feat:<40} {val:.3f}")

    print("\n  Ready for Project 2: CreditScore (PD model)")
    print("=" * 60 + "\n")
