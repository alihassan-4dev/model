"""
Estimate wall-clock training time from config (CPU; approximate).

Tune ESTIMATED_SECONDS_PER_STEP in model/config.py after you see how long
epoch 1 actually takes (epoch_time / steps_per_epoch).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from model.config import (
    BATCH_SIZE,
    EPOCHS,
    ESTIMATED_SECONDS_PER_STEP,
    MAX_TRAIN_SAMPLES,
    MAX_VAL_SAMPLES,
)


@dataclass(frozen=True)
class TrainingTimeEstimate:
    """Result of estimate_training_time()."""

    steps_per_epoch: int
    val_steps_per_epoch: int
    epochs: int
    train_samples: int
    val_samples: int
    seconds_per_step: float
    # Total training steps (train + val) across all epochs (worst case if no early stop)
    total_train_steps: int
    total_val_steps: int
    estimated_seconds: float
    estimated_seconds_low: float
    estimated_seconds_high: float

    def summary(self) -> str:
        mid = format_duration(self.estimated_seconds)
        lo = format_duration(self.estimated_seconds_low)
        hi = format_duration(self.estimated_seconds_high)
        return (
            f"~{mid} typical ({lo}–{hi} range), "
            f"{self.steps_per_epoch} train steps/epoch + {self.val_steps_per_epoch} val steps/epoch, "
            f"{self.epochs} epochs max (early stopping may finish sooner)"
        )


def format_duration(seconds: float) -> str:
    """Human-readable duration (e.g. '2h 15m' or '45m')."""
    if seconds < 0:
        return "0s"
    if seconds < 60:
        return f"{int(seconds)}s"
    minutes = seconds / 60
    if minutes < 60:
        return f"{minutes:.0f}m"
    hours = minutes / 60
    h = int(hours)
    m = int(round(minutes - h * 60))
    if m == 60:
        h += 1
        m = 0
    if m == 0:
        return f"{h}h"
    return f"{h}h {m}m"


def estimate_training_time(
    train_samples: int | None = None,
    val_samples: int | None = None,
    batch_size: int | None = None,
    epochs: int | None = None,
    seconds_per_step: float | None = None,
    uncertainty_low_factor: float = 0.5,
    uncertainty_high_factor: float = 2.5,
) -> TrainingTimeEstimate:
    """
    Estimate total training time on CPU using current (or overridden) settings.

    Formula (rough):
      steps_per_epoch = ceil(train_samples / batch_size)
      val_steps = ceil(val_samples / batch_size)  (0 if no val)
      time ≈ (steps_per_epoch + val_steps) * seconds_per_step * epochs

    Args:
        train_samples: Defaults to MAX_TRAIN_SAMPLES from config.
        val_samples: Defaults to MAX_VAL_SAMPLES from config.
        batch_size: Defaults to BATCH_SIZE from config.
        epochs: Defaults to EPOCHS from config.
        seconds_per_step: Defaults to ESTIMATED_SECONDS_PER_STEP from config.
        uncertainty_low_factor: Multiply typical time by this for optimistic bound.
        uncertainty_high_factor: Multiply typical time by this for pessimistic bound.
    """
    ts = train_samples if train_samples is not None else MAX_TRAIN_SAMPLES
    vs = val_samples if val_samples is not None else MAX_VAL_SAMPLES
    bs = batch_size if batch_size is not None else BATCH_SIZE
    ep = epochs if epochs is not None else EPOCHS
    sps = seconds_per_step if seconds_per_step is not None else ESTIMATED_SECONDS_PER_STEP

    if ts is None:
        raise ValueError(
            "train_samples is unknown (set MAX_TRAIN_SAMPLES in config or pass train_samples=)"
        )
    if ts <= 0:
        raise ValueError("train_samples must be positive")
    if bs <= 0:
        raise ValueError("batch_size must be positive")
    if ep <= 0:
        raise ValueError("epochs must be positive")
    if sps <= 0:
        raise ValueError("seconds_per_step must be positive")

    steps_per_epoch = math.ceil(ts / bs)
    val_steps = math.ceil(vs / bs) if vs and vs > 0 else 0

    per_epoch_seconds = (steps_per_epoch + val_steps) * sps
    estimated_seconds = per_epoch_seconds * ep

    estimated_seconds_low = estimated_seconds * uncertainty_low_factor
    estimated_seconds_high = estimated_seconds * uncertainty_high_factor

    total_train_steps = steps_per_epoch * ep
    total_val_steps = val_steps * ep

    return TrainingTimeEstimate(
        steps_per_epoch=steps_per_epoch,
        val_steps_per_epoch=val_steps,
        epochs=ep,
        train_samples=ts,
        val_samples=vs or 0,
        seconds_per_step=sps,
        total_train_steps=total_train_steps,
        total_val_steps=total_val_steps,
        estimated_seconds=estimated_seconds,
        estimated_seconds_low=estimated_seconds_low,
        estimated_seconds_high=estimated_seconds_high,
    )


def main_cli():
    """Run: uv run python -m utils.training_time"""
    est = estimate_training_time()
    print("Training time estimate (from config defaults)")
    print(" ", est.summary())
    print(
        f"  Typical: {format_duration(est.estimated_seconds)} "
        f"({format_duration(est.estimated_seconds_low)} – {format_duration(est.estimated_seconds_high)})"
    )
    print(
        f"  Tune ESTIMATED_SECONDS_PER_STEP in model/config.py (currently {est.seconds_per_step}) after epoch 1."
    )


if __name__ == "__main__":
    main_cli()
