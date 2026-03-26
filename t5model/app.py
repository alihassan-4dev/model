"""
Interactive summarization app for T5-Small.

Loads the fine-tuned model, then loops:
  - User types an article (or pastes text)
  - Model returns a generated summary

Usage:
    uv run python t5model/app.py

Commands during the session:
    exit / quit  →  stop the app
    help         →  show usage tips
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

HELP_TEXT = """
┌─────────────────────────────────────────────────────┐
│  T5-Small Medical Summarizer — Usage Tips           │
├─────────────────────────────────────────────────────┤
│  • Paste or type a medical article and press Enter  │
│  • The model returns a concise summary              │
│  • Type 'exit' or 'quit' to stop                    │
│  • Type 'help' to see this message again            │
└─────────────────────────────────────────────────────┘
"""


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
        logger.error("The saved_model/ folder may be incomplete. Re-run train.py.")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    logger.info("[%s] Model ready. Device: %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"), device)
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


def run_app() -> None:
    logger.info("=" * 60)
    logger.info("T5-Small Interactive Summarizer  |  %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    logger.info("=" * 60)

    model, tokenizer, device = load_model()
    print(HELP_TEXT)

    while True:
        try:
            user_input = input("Article> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[Exiting] Goodbye!")
            break

        if not user_input:
            print("  (empty input — please enter article text, or type 'help')")
            continue

        cmd = user_input.lower()
        if cmd in {"exit", "quit"}:
            print("Exiting. Goodbye!")
            break
        if cmd == "help":
            print(HELP_TEXT)
            continue

        logger.info("[%s] Generating summary ...", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        try:
            summary = summarize(model, tokenizer, user_input, device)
        except Exception as e:
            logger.error("Error during generation: %s", e)
            continue

        if not summary.strip():
            summary = "(No summary generated)"

        print(f"\nSummary> {summary}\n")


if __name__ == "__main__":
    run_app()
