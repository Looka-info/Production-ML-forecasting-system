"""Tests for data cleaning, transaction validation, and quality reporting."""

import pandas as pd
import pytest

from demandcast.data_validation import clean_and_validate_transactions


def test_clean_and_validate_transactions(sample_transactions_raw):
    """Verify that cancellations, non-merchandise, zero quantities, and duplicates are excluded."""
    cleaned_df, report = clean_and_validate_transactions(sample_transactions_raw)

    assert "series_id" in cleaned_df.columns
    assert "quantity" in cleaned_df.columns
    assert "invoice_date" in cleaned_df.columns

    # Verify exclusions
    assert report["cancellations_count"] == 1
    assert report["exclusions"]["cancelled_invoices"] == 1
    assert report["exclusions"]["non_merchandise_codes"] == 1
    assert report["exclusions"]["non_positive_quantity"] == 1
    assert report["exclusions"]["duplicate_rows"] == 1

    # Verify no non-positive quantities remain
    assert (cleaned_df["quantity"] <= 0).sum() == 0

    # Verify non-merchandise POST is excluded
    assert "POST" not in cleaned_df["series_id"].values

    # Verify customer_id was dropped to prevent customer leakage
    assert "customer_id" not in cleaned_df.columns


def test_missing_required_columns():
    """Verify that an error is raised when required columns are missing."""
    invalid_df = pd.DataFrame({"quantity": [1, 2], "unit_price": [10.0, 20.0]})
    with pytest.raises(ValueError, match="Missing required columns"):
        clean_and_validate_transactions(invalid_df)
