"""
Configuration for PubMed summarization model.
All constants in one place so you can tune them easily.

Defaults target ~5 hours wall-clock on a typical laptop (e.g. i5, 8 GB RAM, CPU):
- VOCAB_SIZE capped at 20k (avoids OOM on the final Dense layer)
- 10k train / 800 val rows, batch 16, shorter sequences
"""

from pathlib import Path

# Paths (relative to project root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

# Data columns
ARTICLE_COL = "article"
ABSTRACT_COL = "abstract"

# Sequence lengths (tokens). Slightly shorter = faster + less RAM on CPU.
MAX_ARTICLE_LEN = 600
MAX_SUMMARY_LEN = 150

# Vocabulary — MUST be capped (e.g. 20k) so the output Dense fits in ~8 GB RAM.
# Uncapped vocab → 700k+ words → OOM on MatMul (batch × seq × vocab).
VOCAB_SIZE = 30_000
OOV_TOKEN = "<OOV>"
START_TOKEN = "startseq"
END_TOKEN = "endseq"

# Model — smaller than default for i5 / 8 GB / ~5 h wall-clock on CPU
EMBEDDING_DIM = 128
LATENT_DIM = 256

# Training — tuned for ~5 h on a typical laptop CPU (i5, 8 GB RAM)
BATCH_SIZE = 16
EPOCHS = 15
VALIDATION_SPLIT = 0.1
EARLY_STOPPING_PATIENCE = 3

# Rough seconds per training step on CPU (calibrate after epoch 1: epoch_wall_time / steps)
# Used by utils.training_time.estimate_training_time()
ESTIMATED_SECONDS_PER_STEP = 2.0

# Subset of data: enough to learn, not so large that each epoch takes forever on CPU
MAX_TRAIN_SAMPLES = 10_000
MAX_VAL_SAMPLES = 800
MAX_TEST_SAMPLES = 800
