"""Script to download and verify the real UCI Online Retail II dataset archive."""

import argparse
import logging
import sys

from demandcast.data_loading import download_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("download_data")


def main():
    parser = argparse.ArgumentParser(description="Download UCI Online Retail II dataset.")
    parser.add_argument("--force", action="store_true", help="Force re-download even if zip exists.")
    args = parser.parse_args()

    try:
        logger.info("Starting dataset acquisition...")
        metadata = download_dataset(force=args.force)
        logger.info("=" * 60)
        logger.info("Dataset Download & Provenance Summary:")
        logger.info(f"Source URL: {metadata.get('source_url')}")
        logger.info(f"Licence: {metadata.get('licence')} ({metadata.get('attribution')})")
        logger.info(f"Archive SHA-256: {metadata.get('zip_sha256')}")
        logger.info(f"Archive Size: {metadata.get('zip_size_bytes'):,} bytes")
        for f in metadata.get("extracted_files", []):
            logger.info(f" - Extracted: {f['filename']} ({f['size_bytes']:,} bytes)")
        logger.info("=" * 60)
    except Exception as e:
        logger.error(f"Download failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
