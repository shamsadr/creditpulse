"""
CreditPulse — Macroeconomic Credit Risk Data Pipeline
Project 1 of 3: Credit Risk Series

Entry point: python src/main.py

Usage:
    python src/main.py                          # full pull, 2000-present
    python src/main.py --start 2010-01-01       # custom start date
    python src/main.py --offline                # use cached data (no API call)
    python src/main.py --api-key YOUR_KEY       # pass key directly (not recommended)
"""

import argparse
import os
import sys
from pathlib import Path

# Allow running from project root as `python src/main.py`
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import (
    engineer_features,
    export_data,
    fetch_series,
    summarize_results,
    validate_data,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="CreditPulse: Pull and engineer FRED credit indicators.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/main.py
  python src/main.py --start 2010-01-01 --end 2023-12-31
  python src/main.py --offline
  FRED_API_KEY=abc123 python src/main.py
        """,
    )
    parser.add_argument(
        "--start",
        default="2000-01-01",
        help="Start date for data pull (default: 2000-01-01)",
    )
    parser.add_argument(
        "--end",
        default=None,
        help="End date for data pull (default: today)",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        dest="api_key",
        help="FRED API key (default: reads FRED_API_KEY env var)",
    )
    parser.add_argument(
        "--output-dir",
        default="data",
        dest="output_dir",
        help="Directory to write outputs (default: data/)",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Skip API pull; use existing data/creditpulse_features.csv",
    )
    return parser.parse_args()


def run_offline(output_dir: str) -> None:
    """Load and summarize existing cached data without hitting the API."""
    import json

    import pandas as pd

    csv_path = Path(output_dir) / "creditpulse_features.csv"
    meta_path = Path(output_dir) / "metadata.json"

    if not csv_path.exists():
        print(
            f"ERROR: No cached data found at {csv_path}.\n"
            "Run without --offline first to pull data."
        )
        sys.exit(1)

    df = pd.read_csv(csv_path, index_col="date", parse_dates=True)
    with open(meta_path) as f:
        metadata = json.load(f)

    print("\n[OFFLINE MODE — using cached data]")
    summarize_results(df, metadata["validation"], csv_path, meta_path)


def main() -> None:
    """Orchestrate the full ETL pipeline."""
    args = parse_args()

    if args.offline:
        run_offline(args.output_dir)
        return

    print("\nCreditPulse — Starting pipeline...")
    print(f"  Date range : {args.start} → {args.end or 'today'}")
    print(f"  Output dir : {args.output_dir}/\n")

    # Stage 1: Fetch
    print("[1/4] Fetching FRED series...")
    raw_df = fetch_series(
        start=args.start,
        end=args.end,
        api_key=args.api_key,
    )

    # Stage 2: Validate
    print("[2/4] Validating data quality...")
    validation_report = validate_data(raw_df)

    # Stage 3: Engineer features
    print("[3/4] Engineering features...")
    features_df = engineer_features(raw_df)

    # Stage 4: Export
    print("[4/4] Exporting outputs...")
    csv_path, meta_path = export_data(features_df, validation_report, args.output_dir)

    # Print summary
    summarize_results(features_df, validation_report, csv_path, meta_path)


if __name__ == "__main__":
    main()
