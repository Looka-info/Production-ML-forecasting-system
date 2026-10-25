"""Download and verify VN1 Forecasting Accuracy Challenge dataset files."""

import hashlib
import json
import logging
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from demandcast.config import get_project_root

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("download_vn1_data")

# Official verified hashes and file sizes
VN1_FILES = {
    "Phase 0 - Sales.csv": {
        "url": "https://raw.githubusercontent.com/jrodenberg0/vn1_fcst/main/Phase%200%20-%20Sales.csv",
        "expected_sha256": "034f50f7338c08cf1bb6facf8c5b36290f3e74d2e69db83099702d2a52f6534a",
        "expected_size": 10804654,
    },
    "Phase 0 - Price.csv": {
        "url": "https://raw.githubusercontent.com/jrodenberg0/vn1_fcst/main/Phase%200%20-%20Price.csv",
        "expected_sha256": "9035cce137a4ab27e3caae6c41d80be41f25adb79b9665a55f6cc56db02ebf3a",
        "expected_size": 7588298,
    },
    "Phase 1 - Sales.csv": {
        "url": "https://raw.githubusercontent.com/jrodenberg0/vn1_fcst/main/Phase%201%20-%20Sales.csv",
        "expected_sha256": "59e74d0def7e7f45ce30f53d9643da3112926189e7fed66321fec2095468b4b2",
        "expected_size": 1018396,
    },
    "Phase 1 - Price.csv": {
        "url": "https://raw.githubusercontent.com/jrodenberg0/vn1_fcst/main/Phase%201%20-%20Price.csv",
        "expected_sha256": "d36f18dc0bd6799897c4b96935acef7526f867a3184db44fb458c0b7fcd07490",
        "expected_size": 962283,
    },
}


def calculate_sha256(filepath: Path) -> str:
    """Calculate SHA-256 checksum of a local file."""
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            sha.update(chunk)
    return sha.hexdigest()


def download_file(url: str, dest_path: Path) -> None:
    """Download a remote file with standard user agent."""
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) DemandCast/2.0"}
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as resp, open(dest_path, "wb") as out_f:
        while chunk := resp.read(65536):
            out_f.write(chunk)


def download_and_verify_vn1() -> dict[str, dict]:
    """Download and verify all VN1 challenge dataset files."""
    root = get_project_root()
    raw_dir = root / "data" / "vn1" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    verified_metadata = {}

    for filename, spec in VN1_FILES.items():
        file_path = raw_dir / filename
        if file_path.exists():
            logger.info(f"Checking existing file '{filename}'...")
            actual_size = file_path.stat().st_size
            actual_sha = calculate_sha256(file_path)
            if actual_sha == spec["expected_sha256"]:
                logger.info(f"Verified '{filename}' (SHA-256 match, {actual_size} bytes)")
                verified_metadata[filename] = {
                    "sha256": actual_sha,
                    "size_bytes": actual_size,
                    "verified": True,
                }
                continue
            else:
                logger.warning(f"Checksum mismatch for existing '{filename}'. Re-downloading...")

        logger.info(f"Downloading '{filename}' from {spec['url']}...")
        download_file(spec["url"], file_path)
        actual_size = file_path.stat().st_size
        actual_sha = calculate_sha256(file_path)

        if actual_sha != spec["expected_sha256"]:
            raise ValueError(
                f"Checksum verification failed for '{filename}'. Expected {spec['expected_sha256']}, got {actual_sha}."
            )

        logger.info(
            f"Successfully downloaded and verified '{filename}' ({actual_size} bytes, SHA-256: {actual_sha[:8]}...)"
        )
        verified_metadata[filename] = {
            "sha256": actual_sha,
            "size_bytes": actual_size,
            "verified": True,
        }

    # Generate and save provenance record
    provenance = {
        "dataset_name": "VN1 Forecasting Accuracy Challenge",
        "competition_name": "VN1 Forecasting - Accuracy Challenge",
        "organizers": "Nicolas Vandeput (SupChains), Flieber, DataSource.ai",
        "dataset_doi": "10.5281/zenodo.23265413",
        "competition_url": "https://supchains.com/vn1-forecasting-competition/",
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "license": "Creative Commons Attribution 4.0 International (CC BY 4.0)",
        "scope": {
            "entity": "Product-Warehouse sales time-series across e-commerce vendors",
            "series_count": 15053,
            "phase0_period": "2020-07-06 to 2023-10-02 (170 weeks)",
            "phase1_period": "2023-10-09 to 2024-01-01 (13 weeks)",
            "phase2_period": "2024-01-08 to 2024-04-01 (13 weeks, final competition holdout)",
            "frequency": "Weekly (Monday week-start)",
        },
        "files": verified_metadata,
    }

    prov_path = raw_dir / "dataset_provenance.json"
    with open(prov_path, "w", encoding="utf-8") as f:
        json.dump(provenance, f, indent=2)
    logger.info(f"Dataset provenance record saved to {prov_path}")

    return provenance


def main():
    download_and_verify_vn1()


if __name__ == "__main__":
    main()
