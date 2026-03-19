"""
Sanity checks for CreditPulse pipeline.

Run with: pytest tests/ -v

Tests are designed to run WITHOUT a FRED API key by injecting
synthetic data that mimics the expected FRED output format.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import (
    FRED_SERIES,
    engineer_features,
    export_data,
    summarize_results,
    validate_data,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_synthetic_fred_df(n_months: int = 60, start: str = "2019-01-01") -> pd.DataFrame:
    """
    Create a synthetic DataFrame that mimics FRED output structure.
    All values are realistic ranges for each indicator.
    """
    np.random.seed(42)
    index = pd.date_range(start=start, periods=n_months, freq="ME")
    rng = np.random.default_rng(42)

    df = pd.DataFrame(
        {
            "delinquency_credit_card": 2.5 + rng.normal(0, 0.3, n_months).cumsum() * 0.05,
            "delinquency_commercial":  1.2 + rng.normal(0, 0.2, n_months).cumsum() * 0.03,
            "chargeoff_credit_card":   3.0 + rng.normal(0, 0.4, n_months).cumsum() * 0.04,
            "unemployment_rate":       4.5 + rng.normal(0, 0.2, n_months).cumsum() * 0.03,
            "credit_spread_baa":       1.8 + rng.normal(0, 0.15, n_months).cumsum() * 0.02,
            "fed_funds_rate":          2.0 + rng.normal(0, 0.1, n_months).cumsum() * 0.02,
            "recession_indicator":     [0] * n_months,
        },
        index=index,
    )
    df.index.name = "date"

    # Clip to realistic ranges
    df["delinquency_credit_card"] = df["delinquency_credit_card"].clip(0.5, 10)
    df["unemployment_rate"] = df["unemployment_rate"].clip(3, 15)
    df["credit_spread_baa"] = df["credit_spread_baa"].clip(0.5, 8)
    df["fed_funds_rate"] = df["fed_funds_rate"].clip(0, 10)
    return df


@pytest.fixture
def raw_df():
    return make_synthetic_fred_df()


@pytest.fixture
def features_df(raw_df):
    return engineer_features(raw_df)


# ---------------------------------------------------------------------------
# Test Stage 2: Validate
# ---------------------------------------------------------------------------

class TestValidate:
    def test_returns_dict_with_expected_keys(self, raw_df):
        report = validate_data(raw_df)
        assert "date_range" in report
        assert "missing_values" in report
        assert "long_gaps_flagged" in report

    def test_date_range_populated(self, raw_df):
        report = validate_data(raw_df)
        assert report["date_range"]["n_months"] == 60
        assert report["date_range"]["start"] < report["date_range"]["end"]

    def test_no_missing_in_clean_data(self, raw_df):
        report = validate_data(raw_df)
        assert all(v == 0 for v in report["missing_values"].values())

    def test_detects_missing_values(self, raw_df):
        dirty = raw_df.copy()
        dirty.iloc[5:10, 0] = np.nan  # inject 5 NaNs
        report = validate_data(dirty)
        first_col = list(dirty.columns)[0]
        assert report["missing_values"][first_col] == 5

    def test_flags_long_gaps(self, raw_df):
        dirty = raw_df.copy()
        dirty.iloc[10:20, 0] = np.nan  # 10-month gap > MAX_GAP_MONTHS
        report = validate_data(dirty)
        first_col = list(dirty.columns)[0]
        assert first_col in report["long_gaps_flagged"]

    def test_raises_on_empty_df(self):
        with pytest.raises(AssertionError):
            validate_data(pd.DataFrame())


# ---------------------------------------------------------------------------
# Test Stage 3: Feature Engineering
# ---------------------------------------------------------------------------

class TestEngineerFeatures:
    def test_output_has_more_columns_than_input(self, raw_df):
        feat = engineer_features(raw_df)
        assert len(feat.columns) > len(raw_df.columns)

    def test_recession_flag_is_binary(self, features_df):
        assert set(features_df["recession_flag"].dropna().unique()).issubset({0, 1})

    def test_recession_indicator_dropped(self, features_df):
        assert "recession_indicator" not in features_df.columns

    def test_lag_columns_created(self, features_df):
        assert "delinquency_credit_card_lag1" in features_df.columns
        assert "delinquency_credit_card_lag3" in features_df.columns
        assert "delinquency_credit_card_lag6" in features_df.columns

    def test_rolling_mean_column_created(self, features_df):
        assert "delinquency_credit_card_rolling3_mean" in features_df.columns

    def test_mom_column_created(self, features_df):
        assert "delinquency_credit_card_mom" in features_df.columns

    def test_credit_stress_index_present(self, features_df):
        assert "credit_stress_index" in features_df.columns

    def test_credit_stress_index_normalized(self, features_df):
        # Normalized → mean ≈ 0, std ≈ 1 (not exact due to averaging 3 cols)
        csi = features_df["credit_stress_index"].dropna()
        assert abs(csi.mean()) < 0.5
        assert 0.3 < csi.std() < 2.0

    def test_no_all_nan_rows(self, features_df):
        all_nan_rows = features_df.isna().all(axis=1).sum()
        assert all_nan_rows == 0

    def test_output_is_dataframe(self, raw_df):
        result = engineer_features(raw_df)
        assert isinstance(result, pd.DataFrame)


# ---------------------------------------------------------------------------
# Test Stage 4: Export
# ---------------------------------------------------------------------------

class TestExport:
    def test_creates_csv_and_metadata(self, features_df, tmp_path):
        raw_df = make_synthetic_fred_df()
        report = validate_data(raw_df)
        csv_path, meta_path = export_data(features_df, report, str(tmp_path))
        assert csv_path.exists()
        assert meta_path.exists()

    def test_csv_is_loadable(self, features_df, tmp_path):
        raw_df = make_synthetic_fred_df()
        report = validate_data(raw_df)
        csv_path, _ = export_data(features_df, report, str(tmp_path))
        loaded = pd.read_csv(csv_path, index_col="date", parse_dates=True)
        assert len(loaded) == len(features_df)
        assert list(loaded.columns) == list(features_df.columns)

    def test_metadata_json_valid(self, features_df, tmp_path):
        raw_df = make_synthetic_fred_df()
        report = validate_data(raw_df)
        _, meta_path = export_data(features_df, report, str(tmp_path))
        with open(meta_path) as f:
            meta = json.load(f)
        assert "pulled_at" in meta
        assert "n_rows" in meta
        assert "n_features" in meta
        assert meta["n_rows"] == len(features_df)


# ---------------------------------------------------------------------------
# Integration test
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_full_pipeline_no_api(self, tmp_path):
        """Full pipeline runs end-to-end with synthetic data (no FRED key needed)."""
        raw = make_synthetic_fred_df()
        report = validate_data(raw)
        features = engineer_features(raw)
        csv_path, meta_path = export_data(features, report, str(tmp_path))

        assert csv_path.exists()
        assert meta_path.exists()

        loaded = pd.read_csv(csv_path, index_col="date", parse_dates=True)
        assert len(loaded) > 0
        assert "credit_stress_index" in loaded.columns
        assert "recession_flag" in loaded.columns
        assert "delinquency_credit_card_lag6" in loaded.columns

    def test_summarize_does_not_crash(self, features_df, tmp_path, capsys):
        raw_df = make_synthetic_fred_df()
        report = validate_data(raw_df)
        csv_path, meta_path = export_data(features_df, report, str(tmp_path))
        summarize_results(features_df, report, csv_path, meta_path)
        captured = capsys.readouterr()
        assert "CreditPulse" in captured.out
        assert "Pipeline Complete" in captured.out
