"""
Fine-tune T5-Small on PubMed article/abstract data.

Usage:
    uv run python t5model/train.py

Saves best model (lowest validation loss) to t5model/saved_model/
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

import time
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5ForConditionalGeneration, AutoTokenizer, get_linear_schedule_with_warmup
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

# ── Hyper-parameters ──────────────────────────────────────────────────────────
MODEL_NAME    = "t5-small"
MAX_INPUT_LEN = 512
MAX_OUTPUT_LEN = 128
BATCH_SIZE    = 8
EPOCHS        = 3
LR            = 5e-5
LOG_STEPS     = 100   # print loss every N steps


# ── Dataset ───────────────────────────────────────────────────────────────────
class SummarizationDataset(Dataset):
    """Wraps a DataFrame into a PyTorch Dataset for T5."""

    def __init__(self, df: pd.DataFrame, tokenizer: AutoTokenizer) -> None:
        self.articles  = df["article"].fillna("").astype(str).tolist()
        self.abstracts = df["abstract"].fillna("").astype(str).tolist()
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.articles)

    def __getitem__(self, idx: int) -> dict:
        article  = "summarize: " + self.articles[idx]
        abstract = self.abstracts[idx]

        input_enc = self.tokenizer(
            article,
            max_length=MAX_INPUT_LEN,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        target_enc = self.tokenizer(
            abstract,
            max_length=MAX_OUTPUT_LEN,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        labels = target_enc.input_ids.squeeze()
        # T5 ignores positions where label == -100 in the loss
        labels = labels.masked_fill(labels == self.tokenizer.pad_token_id, -100)

        return {
            "input_ids":      input_enc.input_ids.squeeze(),
            "attention_mask": input_enc.attention_mask.squeeze(),
            "labels":         labels,
        }


# ── Training loop ─────────────────────────────────────────────────────────────
def run_epoch(
    model: T5ForConditionalGeneration,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    epoch: int,
    phase: str,
    scheduler=None,
) -> float:
    """Run one train or validation epoch. Returns average loss."""
    is_train = phase == "train"
    model.train() if is_train else model.eval()

    total_loss = 0.0
    step = 0
    desc = f"Epoch {epoch}/{EPOCHS} [{phase.capitalize()}]"

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        pbar = tqdm(loader, desc=desc, unit="batch", dynamic_ncols=True)
        for batch in pbar:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

            total_loss += loss.item()
            step += 1

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            if is_train and step % LOG_STEPS == 0:
                avg = total_loss / step
                logger.info(
                    "[%s] Epoch %d | Step %d | Avg Loss: %.4f",
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    epoch,
                    step,
                    avg,
                )

    return total_loss / max(len(loader), 1)


def train() -> None:
    logger.info("=" * 60)
    logger.info("T5-Small Fine-tuning  |  %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    logger.info("=" * 60)

    # ── Validate data paths ───────────────────────────────────────────────────
    for fname in ("train.csv", "validation.csv"):
        if not (DATA_DIR / fname).exists():
            logger.error("Data file not found: %s", DATA_DIR / fname)
            logger.error("Make sure data/ folder contains train.csv and validation.csv")
            sys.exit(1)

    # ── Load data ─────────────────────────────────────────────────────────────
    logger.info("Loading data from %s ...", DATA_DIR)
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    val_df   = pd.read_csv(DATA_DIR / "validation.csv")
    logger.info("Train rows: %d | Validation rows: %d", len(train_df), len(val_df))

    # ── Load tokenizer & model ────────────────────────────────────────────────
    logger.info("Loading %s tokenizer and model ...", MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)
    model.to(device)

    # ── Datasets & loaders ────────────────────────────────────────────────────
    train_dataset = SummarizationDataset(train_df, tokenizer)
    val_dataset   = SummarizationDataset(val_df,   tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=total_steps // 10,
        num_training_steps=total_steps,
    )

    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    # ── Training config summary ────────────────────────────────────────────────
    secs_per_step = 0.5 if str(device) != "cpu" else 8.0
    est_hours = (total_steps * secs_per_step) / 3600
    print("\n" + "═" * 44)
    print("  Training Configuration")
    print("═" * 44)
    print(f"  Model        : {MODEL_NAME}")
    print(f"  Device       : {device}")
    print(f"  Epochs       : {EPOCHS}")
    print(f"  Batch size   : {BATCH_SIZE}")
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples  : {len(val_dataset)}")
    print(f"  Total steps  : {total_steps}")
    print(f"  Est. time    : {est_hours:.1f} hours")
    print("═" * 44 + "\n")

    # ── Training ──────────────────────────────────────────────────────────────
    best_val_loss = float("inf")
    best_epoch = 0
    PATIENCE = 2
    patience_counter = 0
    train_start = time.time()

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()
        train_loss = run_epoch(model, train_loader, optimizer, device, epoch, "train", scheduler)
        val_loss   = run_epoch(model, val_loader,   None,      device, epoch, "val")

        # ── Time tracking ─────────────────────────────────────────────────────
        epoch_mins     = (time.time() - epoch_start) / 60
        elapsed_mins   = (time.time() - train_start) / 60
        remaining_mins = epoch_mins * (EPOCHS - epoch)

        logger.info(
            "[%s] Epoch %d/%d | Train Loss: %.4f | Val Loss: %.4f",
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            epoch, EPOCHS,
            train_loss, val_loss,
        )
        logger.info(
            "  Time elapsed  : %.1f minutes",
            elapsed_mins,
        )
        logger.info(
            "  Time per epoch: %.1f minutes",
            epoch_mins,
        )
        logger.info(
            "  Est. remaining: %.1f minutes",
            remaining_mins,
        )

        # ── Save best / early stopping ────────────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            model.save_pretrained(str(SAVE_DIR))
            tokenizer.save_pretrained(str(SAVE_DIR))
            logger.info(
                "[%s] *** Best model saved to %s (val_loss=%.4f) ***",
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                SAVE_DIR,
                best_val_loss,
            )
        else:
            patience_counter += 1
            logger.info(
                "  No improvement. Patience: %d/%d",
                patience_counter, PATIENCE,
            )
            if patience_counter >= PATIENCE:
                logger.info(
                    "[%s] Early stopping triggered after epoch %d.",
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    epoch,
                )
                break

    # ── Final summary ──────────────────────────────────────────────────────────
    total_mins = (time.time() - train_start) / 60
    logger.info("=" * 60)
    logger.info("Total training time : %.1f minutes (%.1f hours)", total_mins, total_mins / 60)
    logger.info("Total epochs ran    : %d out of %d", epoch, EPOCHS)
    logger.info("Best epoch was      : %d", best_epoch)
    logger.info("Best val loss       : %.4f", best_val_loss)
    logger.info("Model saved to      : t5model/saved_model/")
    logger.info("=" * 60)


if __name__ == "__main__":
    train()
