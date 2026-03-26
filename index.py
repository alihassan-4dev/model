"""
Interactive summarization CLI.

Loads:
- results/best_model.keras
- results/tokenizer.pkl

Then runs a loop:
- You enter article text
- Model returns a generated summary
"""

from __future__ import annotations

import logging
import pickle
import sys

import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

from model.config import RESULTS_DIR, MAX_ARTICLE_LEN, MAX_SUMMARY_LEN, START_TOKEN, END_TOKEN
from model.dataset import get_vocab_size
from model.main_model import build_model


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def load_tokenizer():
    tokenizer_path = RESULTS_DIR / "tokenizer.pkl"
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}. Run train.py first.")
    with open(tokenizer_path, "rb") as f:
        return pickle.load(f)


def load_trained_model(tokenizer):
    model_path = RESULTS_DIR / "best_model.keras"
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}. Run train.py first.")

    vocab_size = get_vocab_size(tokenizer)
    model = build_model(vocab_size=vocab_size)
    model.load_weights(str(model_path))
    return model


def encode_article(tokenizer, article_text: str) -> np.ndarray:
    seq = tokenizer.texts_to_sequences([article_text])
    enc = pad_sequences(seq, maxlen=MAX_ARTICLE_LEN, padding="post", truncating="post")
    return enc.astype(np.int32)


def decode_summary(model, tokenizer, encoder_input: np.ndarray, index_to_word: dict[int, str]) -> str:
    start_idx = tokenizer.word_index.get(START_TOKEN, 1)
    end_idx = tokenizer.word_index.get(END_TOKEN, 2)

    output_tokens: list[int] = []
    for step in range(MAX_SUMMARY_LEN - 1):
        dec_seq = np.zeros((1, MAX_SUMMARY_LEN), dtype=np.int32)
        dec_seq[0, 0] = start_idx
        for j, token_id in enumerate(output_tokens):
            if j + 1 < MAX_SUMMARY_LEN:
                dec_seq[0, j + 1] = token_id

        pred = model.predict([encoder_input, dec_seq], verbose=0)
        next_idx = int(np.argmax(pred[0, step, :]))
        if next_idx == 0 or next_idx == end_idx:
            break
        output_tokens.append(next_idx)

    words = [index_to_word.get(token_id, "<OOV>") for token_id in output_tokens]
    return " ".join(words).strip()


def main() -> None:
    setup_logging()
    logger = logging.getLogger(__name__)

    logger.info("Loading tokenizer and model...")
    tokenizer = load_tokenizer()
    model = load_trained_model(tokenizer)
    index_to_word = {idx: word for word, idx in tokenizer.word_index.items()}
    logger.info("Ready. Type article text and press Enter.")
    logger.info("Type 'exit' (or 'quit') to stop.")

    while True:
        try:
            user_input = input("\nArticle> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not user_input:
            print("Please enter non-empty text.")
            continue
        if user_input.lower() in {"exit", "quit"}:
            print("Exiting.")
            break

        encoder_input = encode_article(tokenizer, user_input)
        summary = decode_summary(model, tokenizer, encoder_input, index_to_word)
        if not summary:
            summary = "(No summary generated)"
        print(f"\nSummary> {summary}")


if __name__ == "__main__":
    main()
