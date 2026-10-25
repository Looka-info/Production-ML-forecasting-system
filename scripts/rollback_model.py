"""Script to roll back active model pointer to a previous verified release."""

import argparse
import logging
import sys

from demandcast.registry import ModelRegistry

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("rollback_model")


def main():
    parser = argparse.ArgumentParser(description="Roll back active model pointer to a previous release.")
    parser.add_argument(
        "--to", dest="target_version", type=str, required=True, help="Target release version to restore."
    )
    parser.add_argument("--reason", type=str, default="Operational rollback requested.", help="Reason for rollback.")
    args = parser.parse_args()

    registry = ModelRegistry()
    current_info = registry.get_current_release_info()
    current_ver = current_info.get("current_version") if current_info else "NONE"

    logger.info(f"Current active version is: {current_ver}")
    logger.info(f"Attempting rollback to version: {args.target_version}...")

    try:
        registry.rollback_to(args.target_version, reason=args.reason)
        new_info = registry.get_current_release_info()
        logger.info("=" * 60)
        logger.info("Model Rollback Succeeded:")
        logger.info(f"Previous Version: {current_ver}")
        logger.info(f"Now Active: {new_info.get('current_version')}")
        logger.info(f"Checksum: {new_info.get('checksum_sha256')}")
        logger.info(f"Reason: {new_info.get('reason')}")
        logger.info("=" * 60)
    except Exception as e:
        logger.error(f"Rollback failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
