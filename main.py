"""
PubMed summarization project — entry point.

Commands:
  Train model:     uv run python train.py
  Evaluate (ROUGE): uv run python evaluate.py
  View data:       uv run streamlit run data_view/data_view.py
"""

import logging
import sys

# Ensure project root is on path when running as main
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    logger.info("PubMed summarization — commands:")
    logger.info("  train:   uv run python train.py")
    logger.info("  eval:    uv run python evaluate.py")
    logger.info("  data UI: uv run streamlit run data_view/data_view.py")
    logger.info("  plots:   uv run python -m visualize.run_visualizations  (saves to results/figures/)")
    logger.info("  time est: uv run python -m utils.training_time  (estimate training duration from config)")


if __name__ == "__main__":
    main()
