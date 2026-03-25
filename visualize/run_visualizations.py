"""
Generate and save all project visualizations with matplotlib.
Run from project root: uv run python -m visualize.run_visualizations

Saves figures to results/figures/
"""

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Project root = parent of visualize/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def save_fig(name: str, dpi: int = 150):
    """Save current figure to results/figures/ and close."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / f"{name}.png"
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()
    logger.info("Saved: %s", path)


def plot_model_architecture():
    """Draw a simple diagram of the seq2seq + attention model."""
    from model.config import (
        MAX_ARTICLE_LEN,
        MAX_SUMMARY_LEN,
        EMBEDDING_DIM,
        LATENT_DIM,
        VOCAB_SIZE,
    )

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title("Model architecture (seq2seq + attention)", fontsize=14, fontweight="bold")

    boxes = [
        (1, 7, 2, 0.8, f"Encoder Input\n(article, {MAX_ARTICLE_LEN})"),
        (1, 5.5, 2, 0.8, f"Embedding\n({EMBEDDING_DIM}-dim)"),
        (1, 4, 2, 0.8, f"LSTM\n({LATENT_DIM} units)"),
        (1, 2.5, 2, 0.8, "Encoder Outputs\n+ state_h, state_c"),
        (4, 7, 2, 0.8, f"Decoder Input\n(abstract, {MAX_SUMMARY_LEN})"),
        (4, 5.5, 2, 0.8, f"Embedding\n({EMBEDDING_DIM}-dim)"),
        (4, 4, 2, 0.8, "LSTM\n(init=encoder state)"),
        (4, 2.5, 2, 0.8, "Decoder Outputs"),
        (7, 4, 2, 0.8, "Attention\n(decoder, encoder)"),
        (7, 2.5, 2, 0.8, f"Concat +\nDense({VOCAB_SIZE})\n→ softmax"),
    ]
    for x, y, w, h, label in boxes:
        rect = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", fc="lightblue", ec="black")
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=8, wrap=True)

    ax.text(
        5,
        0.5,
        f"Config: MAX_ARTICLE_LEN={MAX_ARTICLE_LEN}, MAX_SUMMARY_LEN={MAX_SUMMARY_LEN}, "
        f"EMBEDDING_DIM={EMBEDDING_DIM}, LATENT_DIM={LATENT_DIM}, VOCAB_SIZE={VOCAB_SIZE}",
        ha="center",
        fontsize=9,
    )
    save_fig("01_model_architecture")


def plot_training_history():
    """Plot loss and accuracy from training_history.json if it exists."""
    history_path = RESULTS_DIR / "training_history.json"
    if not history_path.exists():
        logger.warning("No training_history.json found. Run train.py first. Skipping training plots.")
        return

    with open(history_path) as f:
        hist = json.load(f)

    n_plots = len(hist)
    if n_plots == 0:
        return
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4))
    if n_plots == 1:
        axes = [axes]
    for ax, (key, values) in zip(axes, hist.items()):
        ax.plot(values, color="steelblue", linewidth=2)
        ax.set_title(key)
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.3)
    fig.suptitle("Training history", fontsize=14, fontweight="bold", y=1.02)
    save_fig("02_training_history")


def plot_config_summary():
    """Bar chart of main config values (model and data settings)."""
    try:
        from model.config import (
            MAX_ARTICLE_LEN,
            MAX_SUMMARY_LEN,
            EMBEDDING_DIM,
            LATENT_DIM,
            BATCH_SIZE,
            EPOCHS,
            MAX_TRAIN_SAMPLES,
            MAX_VAL_SAMPLES,
        )
    except Exception:
        logger.warning("Could not import model.config. Skipping config plot.")
        return

    labels = [
        "MAX_ARTICLE_LEN",
        "MAX_SUMMARY_LEN",
        "EMBEDDING_DIM",
        "LATENT_DIM",
        "BATCH_SIZE",
        "EPOCHS",
        "MAX_TRAIN_SAMPLES",
        "MAX_VAL_SAMPLES",
    ]
    values = [
        MAX_ARTICLE_LEN,
        MAX_SUMMARY_LEN,
        EMBEDDING_DIM,
        LATENT_DIM,
        BATCH_SIZE,
        EPOCHS,
        MAX_TRAIN_SAMPLES or 0,
        MAX_VAL_SAMPLES or 0,
    ]
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(labels, values, color="steelblue", alpha=0.8)
    ax.set_xlabel("Value")
    ax.set_title("Project config (model & training)")
    for b, v in zip(bars, values):
        ax.text(v + max(values) * 0.01, b.get_y() + b.get_height() / 2, str(v), va="center", fontsize=9)
    save_fig("03_config_summary")


def plot_data_lengths():
    """Plot article and abstract length distribution from a small data sample."""
    data_path = PROJECT_ROOT / "data" / "train.csv"
    if not data_path.exists():
        logger.warning("data/train.csv not found. Skipping data length plot.")
        return

    try:
        import pandas as pd

        df = pd.read_csv(data_path, nrows=5000, on_bad_lines="skip")
        if "article" not in df.columns or "abstract" not in df.columns:
            logger.warning("train.csv missing article/abstract columns. Skipping.")
            return
        df["article_len"] = df["article"].astype(str).str.split().str.len()
        df["abstract_len"] = df["abstract"].astype(str).str.split().str.len()
    except Exception as e:
        logger.warning("Could not load data for length plot: %s", e)
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(df["article_len"], bins=50, color="steelblue", alpha=0.8, edgecolor="black")
    axes[0].set_xlabel("Article length (words)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Article length (sample 5000)")
    axes[1].hist(df["abstract_len"], bins=50, color="coral", alpha=0.8, edgecolor="black")
    axes[1].set_xlabel("Abstract length (words)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Abstract length (sample 5000)")
    fig.suptitle("Data length distribution", fontsize=14, fontweight="bold", y=1.02)
    save_fig("04_data_lengths")


def main():
    logger.info("Generating visualizations -> %s", FIGURES_DIR)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    plot_model_architecture()
    plot_config_summary()
    plot_data_lengths()
    plot_training_history()

    logger.info("Done. Figures saved in: %s", FIGURES_DIR)


if __name__ == "__main__":
    main()
