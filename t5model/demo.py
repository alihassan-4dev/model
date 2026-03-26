"""
Demo: Run 3 hardcoded medical articles through the fine-tuned T5-Small model.

Usage:
    uv run python t5model/demo.py
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent          # t5model/
SAVE_DIR = BASE_DIR / "saved_model"                 # t5model/saved_model/

# ── Config ────────────────────────────────────────────────────────────────────
MAX_INPUT_LEN  = 512
MAX_OUTPUT_LEN = 128

# ── Hardcoded demo articles ───────────────────────────────────────────────────
DEMO_ARTICLES = [
    {
        "title": "Parkinson's Disease & Anxiety",
        "text": (
            "parkinson s disease is a neurodegenerative disorder affecting dopaminergic neurons . "
            "anxiety affects 50 percent of parkinson patients and reduces quality of life . "
            "this study evaluated 200 patients and found anxiety associated with depression "
            "and disease severity ."
        ),
    },
    {
        "title": "Type 2 Diabetes & Empagliflozin",
        "text": (
            "type 2 diabetes affects 460 million people worldwide . "
            "empagliflozin was compared to placebo in 350 patients over 24 weeks . "
            "hba1c reduced by 0.8 percent in treatment group versus 0.1 percent in placebo group ."
        ),
    },
    {
        "title": "MiR-21 in Breast Cancer",
        "text": (
            "mir 21 expression was examined in 80 breast cancer tissue samples . "
            "mir 21 was overexpressed in cancer tissues and correlated with advanced tumor stage . "
            "inhibition of mir 21 suppressed cell proliferation and invasion ."
        ),
    },
]


def load_model() -> tuple[T5ForConditionalGeneration, T5Tokenizer, torch.device]:
    """Load fine-tuned model and tokenizer from saved_model/."""
    if not SAVE_DIR.exists() or not any(SAVE_DIR.iterdir()):
        logger.error("saved_model/ not found or empty at: %s", SAVE_DIR)
        logger.error("Run training first:  uv run python t5model/train.py")
        sys.exit(1)

    logger.info("[%s] Loading model from %s ...", datetime.now().strftime("%Y-%m-%d %H:%M:%S"), SAVE_DIR)
    try:
        tokenizer = T5Tokenizer.from_pretrained(str(SAVE_DIR))
        model     = T5ForConditionalGeneration.from_pretrained(str(SAVE_DIR))
    except Exception as e:
        logger.error("Failed to load model: %s", e)
        logger.error("Make sure training completed successfully and saved files to %s", SAVE_DIR)
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    logger.info("Model ready. Device: %s", device)
    return model, tokenizer, device


def summarize(
    model: T5ForConditionalGeneration,
    tokenizer: T5Tokenizer,
    text: str,
    device: torch.device,
) -> str:
    """Generate a summary for a single input text."""
    input_text = "summarize: " + text
    enc = tokenizer(
        input_text,
        max_length=MAX_INPUT_LEN,
        truncation=True,
        return_tensors="pt",
    )
    input_ids      = enc.input_ids.to(device)
    attention_mask = enc.attention_mask.to(device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=MAX_OUTPUT_LEN,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=3,
        )

    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


def run_demo() -> None:
    logger.info("=" * 60)
    logger.info("T5-Small Demo  |  %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    logger.info("=" * 60)

    model, tokenizer, device = load_model()

    print()
    for i, article in enumerate(DEMO_ARTICLES, start=1):
        print("=" * 60)
        print(f"  Article {i}: {article['title']}")
        print("=" * 60)
        print(f"INPUT :\n  {article['text']}")
        print()

        try:
            summary = summarize(model, tokenizer, article["text"], device)
        except Exception as e:
            logger.error("Failed to generate summary for article %d: %s", i, e)
            summary = "(Error generating summary)"

        print(f"OUTPUT:\n  {summary}")
        print()

    logger.info("[%s] Demo complete.", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


if __name__ == "__main__":
    run_demo()
