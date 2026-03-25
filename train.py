"""
Train the PubMed summarization model.
Run from project root: uv run python train.py
"""

import logging
import sys
from pathlib import Path

import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

from model.config import (
    RESULTS_DIR,
    BATCH_SIZE,
    EPOCHS,
    VALIDATION_SPLIT,
    EARLY_STOPPING_PATIENCE,
)
from model.dataset import (
    load_train_val_test,
    build_tokenizer,
    prepare_sequences,
    get_vocab_size,
)
from model.main_model import build_model
from utils.training_time import estimate_training_time, format_duration


def setup_logging(level=logging.INFO):
    """Configure logging so messages are easy to read."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def main():
    setup_logging()
    logger = logging.getLogger(__name__)

    logger.info("========== Loading data ==========")
    train_df, val_df, test_df = load_train_val_test()

    logger.info("Train size: %d, Val size: %d, Test size: %d", len(train_df), len(val_df), len(test_df))

    val_n = len(val_df) if not val_df.empty else 0
    time_est = estimate_training_time(train_samples=len(train_df), val_samples=val_n)
    logger.info("========== Time estimate (CPU, approximate) ==========")
    logger.info("%s", time_est.summary())
    logger.info(
        "  Typical %s (range %s – %s); assumes %.2f s/step → edit ESTIMATED_SECONDS_PER_STEP in model/config.py after epoch 1",
        format_duration(time_est.estimated_seconds),
        format_duration(time_est.estimated_seconds_low),
        format_duration(time_est.estimated_seconds_high),
        time_est.seconds_per_step,
    )

    # Optional: quick stats
    train_df["_article_len"] = train_df["article"].astype(str).str.split().str.len()
    train_df["_abstract_len"] = train_df["abstract"].astype(str).str.split().str.len()
    logger.info(
        "Article length (words) - mean: %.0f, max: %d",
        train_df["_article_len"].mean(),
        train_df["_article_len"].max(),
    )
    logger.info(
        "Abstract length (words) - mean: %.0f, max: %d",
        train_df["_abstract_len"].mean(),
        train_df["_abstract_len"].max(),
    )
    train_df.drop(columns=["_article_len", "_abstract_len"], inplace=True)

    logger.info("========== Building tokenizer ==========")
    tokenizer = build_tokenizer(
        train_df["article"].astype(str).tolist(),
        train_df["abstract"].astype(str).tolist(),
    )
    vocab_size = get_vocab_size(tokenizer)
    logger.info("Vocabulary size for model: %d", vocab_size)

    logger.info("========== Preparing sequences ==========")
    enc_train, dec_train, target_train = prepare_sequences(
        tokenizer,
        train_df["article"].astype(str).tolist(),
        train_df["abstract"].astype(str).tolist(),
    )
    logger.info("Train encoder shape: %s, decoder shape: %s", enc_train.shape, dec_train.shape)

    if not val_df.empty:
        enc_val, dec_val, target_val = prepare_sequences(
            tokenizer,
            val_df["article"].astype(str).tolist(),
            val_df["abstract"].astype(str).tolist(),
        )
        logger.info("Val encoder shape: %s", enc_val.shape)
        validation_data = ([enc_val, dec_val], np.expand_dims(target_val, -1))
    else:
        validation_data = None

    # Target for loss: (batch, max_summary_len, 1) for sparse_categorical_crossentropy
    target_train = np.expand_dims(target_train, -1)

    logger.info("========== Building model ==========")
    model = build_model(vocab_size=vocab_size)
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary(print_fn=logger.info)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = RESULTS_DIR / "best_model.keras"

    callbacks = [
        EarlyStopping(
            monitor="val_loss" if validation_data else "loss",
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        ModelCheckpoint(
            str(checkpoint_path),
            monitor="val_loss" if validation_data else "loss",
            save_best_only=True,
            verbose=1,
        ),
    ]

    logger.info("========== Training ==========")
    logger.info("Batch size: %d, Max epochs: %d", BATCH_SIZE, EPOCHS)

    if validation_data:
        history = model.fit(
            [enc_train, dec_train],
            target_train,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            validation_data=validation_data,
            callbacks=callbacks,
        )
    else:
        history = model.fit(
            [enc_train, dec_train],
            target_train,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            validation_split=VALIDATION_SPLIT,
            callbacks=callbacks,
        )

    logger.info("========== Training complete ==========")
    logger.info("Best model saved to: %s", checkpoint_path)

    # Save tokenizer for inference/eval
    import pickle

    tokenizer_path = RESULTS_DIR / "tokenizer.pkl"
    with open(tokenizer_path, "wb") as f:
        pickle.dump(tokenizer, f)
    logger.info("Tokenizer saved to: %s", tokenizer_path)

    # Save training history for plotting (visualize/run_visualizations.py)
    import json

    history_path = RESULTS_DIR / "training_history.json"
    hist = {k: [float(x) for x in v] for k, v in history.history.items()}
    with open(history_path, "w") as f:
        json.dump(hist, f, indent=2)
    logger.info("Training history saved to: %s", history_path)

    return model, history, tokenizer


if __name__ == "__main__":
    main()
