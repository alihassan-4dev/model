"""
Evaluate the fine-tuned T5-Small model on the first 100 rows of test.csv.

Prints ROUGE-1, ROUGE-2, ROUGE-L scores and 3 sample predictions.

Usage:
    uv run python t5model/evaluate.py
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from rouge_score import rouge_scorer
from tqdm import tqdm

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
DATA_DIR = BASE_DIR.parent / "data"                 # ../data/
SAVE_DIR = BASE_DIR / "saved_model"                 # t5model/saved_model/

# ── Config ────────────────────────────────────────────────────────────────────
MAX_INPUT_LEN   = 512
MAX_OUTPUT_LEN  = 128
EVAL_ROWS       = 100   # number of test rows to evaluate
SAMPLE_COUNT    = 3     # number of samples to print


def load_model() -> tuple[T5ForConditionalGeneration, T5Tokenizer, torch.device]:
    """Load fine-tuned model and tokenizer from saved_model/."""
    if not SAVE_DIR.exists() or not any(SAVE_DIR.iterdir()):
        logger.error("saved_model/ not found or empty at: %s", SAVE_DIR)
        logger.error("Run train.py first:  uv run python t5model/train.py")
        sys.exit(1)

    logger.info("[%s] Loading model from %s ...", datetime.now().strftime("%Y-%m-%d %H:%M:%S"), SAVE_DIR)
    tokenizer = T5Tokenizer.from_pretrained(str(SAVE_DIR))
    model     = T5ForConditionalGeneration.from_pretrained(str(SAVE_DIR))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    logger.info("Model loaded. Device: %s", device)
    return model, tokenizer, device


def generate_summary(
    model: T5ForConditionalGeneration,
    tokenizer: T5Tokenizer,
    article: str,
    device: torch.device,
) -> str:
    """Generate a summary for a single article."""
    input_text = "summarize: " + article
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


def evaluate() -> None:
    logger.info("=" * 60)
    logger.info("T5-Small Evaluation  |  %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    logger.info("=" * 60)

    # ── Load data ─────────────────────────────────────────────────────────────
    test_path = DATA_DIR / "test.csv"
    if not test_path.exists():
        logger.error("test.csv not found at: %s", test_path)
        sys.exit(1)

    test_df = pd.read_csv(test_path, nrows=EVAL_ROWS)
    test_df["article"]  = test_df["article"].fillna("").astype(str)
    test_df["abstract"] = test_df["abstract"].fillna("").astype(str)
    logger.info("Evaluating on %d test rows ...", len(test_df))

    # ── Load model ────────────────────────────────────────────────────────────
    model, tokenizer, device = load_model()

    # ── Generate predictions ──────────────────────────────────────────────────
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    r1_scores, r2_scores, rl_scores = [], [], []
    predictions: list[str] = []

    logger.info("[%s] Generating summaries ...", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Evaluating", unit="article"):
        pred = generate_summary(model, tokenizer, row["article"], device)
        ref  = row["abstract"]

        scores = scorer.score(ref, pred)
        r1_scores.append(scores["rouge1"].fmeasure)
        r2_scores.append(scores["rouge2"].fmeasure)
        rl_scores.append(scores["rougeL"].fmeasure)
        predictions.append(pred)

    # ── ROUGE results ─────────────────────────────────────────────────────────
    avg_r1 = sum(r1_scores) / len(r1_scores)
    avg_r2 = sum(r2_scores) / len(r2_scores)
    avg_rl = sum(rl_scores) / len(rl_scores)

    logger.info("=" * 60)
    logger.info("ROUGE Scores (on %d test samples)", EVAL_ROWS)
    logger.info("  ROUGE-1 : %.4f", avg_r1)
    logger.info("  ROUGE-2 : %.4f", avg_r2)
    logger.info("  ROUGE-L : %.4f", avg_rl)
    logger.info("=" * 60)

    # ── Sample predictions ────────────────────────────────────────────────────
    logger.info("Sample Predictions (first %d):", SAMPLE_COUNT)
    for i in range(min(SAMPLE_COUNT, len(predictions))):
        print(f"\n--- Sample {i + 1} ---")
        print(f"REFERENCE : {test_df.iloc[i]['abstract'][:300]}")
        print(f"PREDICTION: {predictions[i][:300]}")

    logger.info("\n[%s] Evaluation complete.", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


if __name__ == "__main__":
    evaluate()
