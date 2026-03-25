"""
Load and preprocess PubMed article/abstract data.
Builds tokenizer and returns padded encoder/decoder sequences.
"""

import logging
from pathlib import Path

import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from model.config import (
    DATA_DIR,
    ARTICLE_COL,
    ABSTRACT_COL,
    MAX_ARTICLE_LEN,
    MAX_SUMMARY_LEN,
    VOCAB_SIZE,
    OOV_TOKEN,
    START_TOKEN,
    END_TOKEN,
    MAX_TRAIN_SAMPLES,
    MAX_VAL_SAMPLES,
    MAX_TEST_SAMPLES,
)

logger = logging.getLogger(__name__)


def _load_csv(name: str, max_rows: int | None) -> pd.DataFrame:
    """Load one CSV from data/ with optional row limit."""
    path = DATA_DIR / name
    if not path.exists():
        logger.error("Data file not found: %s", path)
        return pd.DataFrame()

    try:
        df = pd.read_csv(path, nrows=max_rows, on_bad_lines="skip")
        logger.info("Loaded %s: %d rows, columns: %s", name, len(df), list(df.columns))
        return df
    except Exception as e:
        logger.exception("Failed to load %s: %s", name, e)
        return pd.DataFrame()


def load_train_val_test():
    """
    Load train, validation, and test DataFrames.
    Uses MAX_TRAIN_SAMPLES / MAX_VAL_SAMPLES / MAX_TEST_SAMPLES from config if set.
    """
    train = _load_csv("train.csv", MAX_TRAIN_SAMPLES)
    val = _load_csv("validation.csv", MAX_VAL_SAMPLES)
    test = _load_csv("test.csv", MAX_TEST_SAMPLES)

    if train.empty:
        raise FileNotFoundError("train.csv not found or empty. Check data/ folder.")

    for df, name in [(train, "train"), (val, "val"), (test, "test")]:
        if df.empty:
            continue
        if ARTICLE_COL not in df.columns or ABSTRACT_COL not in df.columns:
            raise ValueError(f"{name} must have columns: {ARTICLE_COL}, {ABSTRACT_COL}")

    return train, val, test


def build_tokenizer(train_articles: list[str], train_abstracts: list[str]) -> Tokenizer:
    """
    Fit tokenizer with a fixed vocab cap (VOCAB_SIZE) so the model fits in RAM.

    <start> / <end> are included by fitting on strings that contain them on every
    abstract line (so they stay among the top num_words tokens).
    """
    abstracts_with_markers = [
        f"{START_TOKEN} {str(a)} {END_TOKEN}" for a in train_abstracts
    ]
    all_text = train_articles + abstracts_with_markers
    tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token=OOV_TOKEN)
    tokenizer.fit_on_texts(all_text)
    # Force special tokens to stay inside vocab cap
    special_tokens = [OOV_TOKEN, START_TOKEN, END_TOKEN]
    for token in special_tokens:
        if token not in tokenizer.word_index:
            # assign a low index so it's never cut off by num_words
            idx = len(tokenizer.word_index) + 1
            tokenizer.word_index[token] = idx
            tokenizer.index_word[idx] = token

    start_idx = tokenizer.word_index.get(START_TOKEN)
    end_idx = tokenizer.word_index.get(END_TOKEN)
    assert start_idx is not None, "startseq token missing from vocabulary!"
    assert end_idx is not None, "endseq token missing from vocabulary!"
    assert start_idx < VOCAB_SIZE, f"startseq index {start_idx} outside vocab cap {VOCAB_SIZE}!"
    assert end_idx < VOCAB_SIZE, f"endseq index {end_idx} outside vocab cap {VOCAB_SIZE}!"

    logger.info(
        "Tokenizer built: num_words=%d (capped), word_index entries=%d, <start> idx=%s, <end> idx=%s",
        VOCAB_SIZE,
        len(tokenizer.word_index),
        tokenizer.word_index.get(START_TOKEN),
        tokenizer.word_index.get(END_TOKEN),
    )
    return tokenizer


def prepare_sequences(
    tokenizer: Tokenizer,
    articles: list[str],
    abstracts: list[str],
) -> tuple:
    """
    Convert articles and abstracts to padded integer sequences.
    Returns (encoder_input, decoder_input, decoder_target).
    decoder_target is decoder_input shifted by 1 (next token); last position set to 0.
    """
    # Encoder: article -> sequence
    encoder_seq = tokenizer.texts_to_sequences(articles)
    encoder_input = pad_sequences(
        encoder_seq,
        maxlen=MAX_ARTICLE_LEN,
        padding="post",
        truncating="post",
    )

    # Decoder: "<start> abstract <end>"
    abstract_with_tokens = [f"{START_TOKEN} {str(a)} {END_TOKEN}" for a in abstracts]
    decoder_seq = tokenizer.texts_to_sequences(abstract_with_tokens)
    decoder_input = pad_sequences(
        decoder_seq,
        maxlen=MAX_SUMMARY_LEN,
        padding="post",
        truncating="post",
    )

    # Target = next token at each step (shift left by 1; last position = 0)
    import numpy as np

    decoder_target = np.roll(decoder_input, -1, axis=1)
    decoder_target[:, -1] = 0

    logger.info(
        "Sequences prepared: encoder %s, decoder %s, target %s",
        encoder_input.shape,
        decoder_input.shape,
        decoder_target.shape,
    )
    return encoder_input, decoder_input, decoder_target


def get_vocab_size(tokenizer: Tokenizer) -> int:
    """Embedding input_dim: indices are 0..num_words-1 when num_words is set."""
    return int(tokenizer.num_words)
