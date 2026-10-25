"""Dataset acquisition and raw data loading for DemandCast.

Dataset: UCI Online Retail II
Source: https://archive.ics.uci.edu/dataset/502/online+retail+ii
Attribution: Chen, D. (2012). Online Retail II [Dataset]. UCI Machine Learning Repository.
Licence: Creative Commons Attribution 4.0 International (CC BY 4.0)
"""

import hashlib
import json
import logging
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from demandcast.config import get_project_root, load_config

logger = logging.getLogger(__name__)


def calculate_sha256(filepath: Path) -> str:
    """Calculate the SHA-256 checksum of a file."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(65536), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def download_dataset(
    url: str | None = None,
    dest_zip_path: Path | None = None,
    extract_to: Path | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Download the UCI Online Retail II dataset archive and extract it.

    Returns metadata dictionary including checksum, file sizes, and paths.
    """
    config = load_config()
    root = get_project_root()

    url = url or config.data.dataset_url
    dest_zip = dest_zip_path or (root / config.data.raw_zip_path)
    extract_dir = extract_to or (root / "data" / "raw")

    dest_zip.parent.mkdir(parents=True, exist_ok=True)
    extract_dir.mkdir(parents=True, exist_ok=True)

    metadata: dict[str, Any] = {
        "dataset_name": "Online Retail II",
        "source_url": url,
        "licence": "CC BY 4.0",
        "attribution": "Chen, D. (2012). Online Retail II [Dataset]. UCI Machine Learning Repository.",
        "download_timestamp": datetime.now(timezone.utc).isoformat(),
    }

    if not dest_zip.exists() or force:
        logger.info(f"Downloading dataset from {url} to {dest_zip}...")
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "DemandCast-ML-System/1.0 (Academic/Research Use)"},
        )
        with urllib.request.urlopen(req) as response, open(dest_zip, "wb") as out_file:
            # Stream in chunks
            while chunk := response.read(1024 * 1024):
                out_file.write(chunk)
        logger.info("Download completed successfully.")
    else:
        logger.info(f"Using existing archive at {dest_zip}")

    zip_checksum = calculate_sha256(dest_zip)
    zip_size = dest_zip.stat().st_size
    metadata["zip_sha256"] = zip_checksum
    metadata["zip_size_bytes"] = zip_size

    # Extract contents
    extracted_files = []
    with zipfile.ZipFile(dest_zip, "r") as zip_ref:
        for file_info in zip_ref.infolist():
            zip_ref.extract(file_info, extract_dir)
            extracted_path = extract_dir / file_info.filename
            extracted_files.append({
                "filename": file_info.filename,
                "size_bytes": file_info.file_size,
                "sha256": calculate_sha256(extracted_path) if extracted_path.is_file() else None,
            })

    metadata["extracted_files"] = extracted_files

    # Save provenance metadata
    meta_path = extract_dir / "dataset_provenance.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return metadata


def load_raw_transactions(excel_path: Path | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load raw transactions from the multi-sheet Online Retail II Excel file.

    The dataset contains two sheets:
    - 'Year 2009-2010'
    - 'Year 2010-2011'
    """
    config = load_config()
    root = get_project_root()
    path = excel_path or (root / config.data.raw_excel_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Raw dataset file not found at {path}. Please run scripts/download_data.py first."
        )

    logger.info(f"Reading raw Excel file from {path} (both 2009-2010 and 2010-2011 sheets)...")
    xl = pd.ExcelFile(path)
    logger.info(f"Detected sheets: {xl.sheet_names}")

    dfs = []
    sheet_info = {}
    for sheet_name in xl.sheet_names:
        df_sheet = xl.parse(sheet_name)
        sheet_info[sheet_name] = {
            "row_count": len(df_sheet),
            "columns": list(df_sheet.columns),
        }
        dfs.append(df_sheet)

    raw_df = pd.concat(dfs, ignore_index=True)
    raw_df.columns = [str(col).strip().lower().replace(" ", "_") for col in raw_df.columns]

    # Standardize column mapping if needed
    col_mapping = {
        "invoicedate": "invoice_date",
        "stockcode": "stock_code",
        "customer_id": "customer_id",
        "price": "unit_price",
    }
    raw_df = raw_df.rename(columns=col_mapping)

    info = {
        "source_file": str(path),
        "total_raw_rows": len(raw_df),
        "columns": list(raw_df.columns),
        "sheets": sheet_info,
    }
    logger.info(f"Loaded total {len(raw_df):,} raw transaction rows.")
    return raw_df, info
