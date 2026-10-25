"""Script to evaluate candidate model and produce baseline benchmark report."""

import logging

import pandas as pd

from demandcast.config import get_project_root
from demandcast.evaluate import run_full_evaluation

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("evaluate_model")


def main():
    root = get_project_root()
    comp_file = root / "reports" / "model_comparison.csv"

    if comp_file.exists():
        logger.info(f"Existing model comparison report found at {comp_file}:")
        df = pd.read_csv(comp_file)
        print("\n" + df.to_string(index=False) + "\n")
    else:
        logger.info("Running full evaluation to generate model comparison report...")
        df, _, _ = run_full_evaluation()
        print("\n" + df.to_string(index=False) + "\n")


if __name__ == "__main__":
    main()
