"""
Evaluate the trained model with ROUGE scores and show sample summaries.
Run from project root: uv run python evaluate.py
"""

import logging
import pickle
import sys

import numpy as np
from rouge_score import rouge_scorer

from model.config import (
    RESULTS_DIR,
    MAX_ARTICLE_LEN,
    MAX_SUMMARY_LEN,
    START_TOKEN,
    END_TOKEN,
)
from model.dataset import load_train_val_test, prepare_sequences, get_vocab_size
from model.main_model import build_model


def setup_logging(level=logging.INFO):
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def load_saved_tokenizer():
    path = RESULTS_DIR / "tokenizer.pkl"
    if not path.exists():
        raise FileNotFoundError("Run train.py first to save tokenizer at results/tokenizer.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)


def decode_sequence(model, tokenizer, encoder_input, index_to_word):
    """Generate summary one token at a time (greedy). Decoder input = [start, t1, t2, ..., 0, 0]."""
    start_idx = tokenizer.word_index.get(START_TOKEN, 1)
    end_idx = tokenizer.word_index.get(END_TOKEN, 2)
    output_tokens = []

    for step in range(MAX_SUMMARY_LEN - 1):
        dec_seq = np.zeros((1, MAX_SUMMARY_LEN), dtype=np.int32)
        dec_seq[0, 0] = start_idx
        for j, t in enumerate(output_tokens):
            if j + 1 < MAX_SUMMARY_LEN:
                dec_seq[0, j + 1] = t
        pred = model.predict([encoder_input, dec_seq], verbose=0)
        next_idx = int(np.argmax(pred[0, step, :]))
        if next_idx == end_idx or next_idx == 0:
            break
        output_tokens.append(next_idx)

    return " ".join(index_to_word.get(i, "<OOV>") for i in output_tokens)


def main():
    setup_logging()
    logger = logging.getLogger(__name__)

    model_path = RESULTS_DIR / "best_model.keras"
    if not model_path.exists():
        logger.error("No trained model at %s. Run train.py first.", model_path)
        return

    logger.info("========== Loading tokenizer and model ==========")
    tokenizer = load_saved_tokenizer()
    vocab_size = get_vocab_size(tokenizer)
    model = build_model(vocab_size=vocab_size)
    model.load_weights(str(model_path))
    logger.info("Model loaded from %s", model_path)

    index_to_word = {v: k for k, v in tokenizer.word_index.items()}

    logger.info("========== Loading test data ==========")
    _, val_df, test_df = load_train_val_test()
    if test_df.empty and not val_df.empty:
        logger.warning("No test data; using validation set for eval.")
        test_df = val_df
    if test_df.empty:
        logger.error("No test/val data available.")
        return

    # Limit for quick eval
    n_eval = min(100, len(test_df))
    test_df = test_df.head(n_eval)
    enc_test, dec_test, _ = prepare_sequences(
        tokenizer,
        test_df["article"].astype(str).tolist(),
        test_df["abstract"].astype(str).tolist(),
    )
    logger.info("Evaluating on %d samples", n_eval)

    logger.info("========== Generating summaries ==========")
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    rouge1_f = []
    rouge2_f = []
    rougeL_f = []

    for i in range(n_eval):
        enc_i = enc_test[i : i + 1]
        pred_summary = decode_sequence(model, tokenizer, enc_i, index_to_word)
        ref = test_df.iloc[i]["abstract"]
        if not isinstance(ref, str):
            ref = str(ref)
        scores = scorer.score(ref, pred_summary)
        rouge1_f.append(scores["rouge1"].fmeasure)
        rouge2_f.append(scores["rouge2"].fmeasure)
        rougeL_f.append(scores["rougeL"].fmeasure)
        if i < 3:
            logger.info("--- Sample %d ---", i + 1)
            logger.info("Reference (first 200 chars): %s...", ref[:200])
            logger.info("Predicted (first 200 chars): %s...", pred_summary[:200])

    logger.info("========== ROUGE results ==========")
    logger.info("ROUGE-1 F: %.4f", np.mean(rouge1_f))
    logger.info("ROUGE-2 F: %.4f", np.mean(rouge2_f))
    logger.info("ROUGE-L F: %.4f", np.mean(rougeL_f))


if __name__ == "__main__":
    main()
