"""Pytest test fixtures for DemandCast unit and integration tests."""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from demandcast.models import DemandForecastModel


@pytest.fixture
def sample_transactions_raw():
    """Synthetic transactions DataFrame mirroring UCI Online Retail II raw structure."""
    base_time = datetime(2010, 1, 4, 10, 30)  # Monday
    records = []

    # Valid merchandise for 2 products over 40 weeks
    for w in range(40):
        t = base_time + timedelta(weeks=w)
        # SKU A
        records.append(
            {
                "invoice": f"INV{1000 + w}",
                "stock_code": "85123A",
                "description": "WHITE HANGING HEART T-LIGHT HOLDER",
                "quantity": 20 + (w % 5),
                "invoice_date": t,
                "unit_price": 2.55,
                "customer_id": 12345,
                "country": "United Kingdom",
            }
        )
        # SKU B
        records.append(
            {
                "invoice": f"INV{2000 + w}",
                "stock_code": "84879",
                "description": "ASSORTED COLOUR BIRD ORNAMENT",
                "quantity": 15 + (w % 4),
                "invoice_date": t + timedelta(days=2),
                "unit_price": 1.69,
                "customer_id": 12346,
                "country": "United Kingdom",
            }
        )

    # Add cancellations (Invoice starting with 'C')
    records.append(
        {
            "invoice": "C10999",
            "stock_code": "85123A",
            "description": "CANCELLATION",
            "quantity": -5,
            "invoice_date": base_time + timedelta(weeks=5),
            "unit_price": 2.55,
            "customer_id": 12345,
            "country": "United Kingdom",
        }
    )

    # Add non-merchandise
    records.append(
        {
            "invoice": "INV9999",
            "stock_code": "POST",
            "description": "POSTAGE",
            "quantity": 1,
            "invoice_date": base_time + timedelta(weeks=6),
            "unit_price": 18.0,
            "customer_id": 12345,
            "country": "United Kingdom",
        }
    )

    # Add non-positive quantity
    records.append(
        {
            "invoice": "INV9998",
            "stock_code": "85123A",
            "description": "ZERO QTY",
            "quantity": 0,
            "invoice_date": base_time + timedelta(weeks=7),
            "unit_price": 2.55,
            "customer_id": 12345,
            "country": "United Kingdom",
        }
    )

    # Add duplicate row
    dup = dict(records[0])
    records.append(dup)

    return pd.DataFrame(records)


@pytest.fixture
def sample_weekly_demand():
    """Synthetic continuous weekly demand dataset for 2 series over 50 weeks."""
    base_date = pd.to_datetime("2010-01-04")  # Monday
    records = []
    skus = ["SKU_ALPHA", "SKU_BETA"]

    for sku in skus:
        base_demand = 50.0 if sku == "SKU_ALPHA" else 30.0
        for i in range(50):
            w_date = base_date + pd.Timedelta(weeks=i)
            # deterministic seasonal & noise pattern
            demand_val = base_demand + 10.0 * np.sin(2 * np.pi * i / 26.0) + (i % 3)
            records.append(
                {
                    "series_id": sku,
                    "week_start": w_date,
                    "demand": float(max(0.0, demand_val)),
                }
            )

    df = pd.DataFrame(records)
    df["week_start"] = pd.to_datetime(df["week_start"])
    return df


@pytest.fixture
def trained_dummy_model(sample_weekly_demand):
    """Pre-trained DemandForecastModel on sample weekly demand."""
    model = DemandForecastModel()
    model.fit(sample_weekly_demand)
    return model
