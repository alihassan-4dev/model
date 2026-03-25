# Project guide — what each file does

This document explains **which file contains which code** and **what happens when you run** `main.py`, `train.py`, and `evaluate.py`.

---

## Quick reference

| Command | What it runs | Main outcome |
|--------|----------------|--------------|
| `uv run python main.py` | Entry point / help | Prints how to run train, evaluate, data view |
| `uv run python train.py` | Full training pipeline | Trains model, saves `results/best_model.keras`, `tokenizer.pkl`, `training_history.json` |
| `uv run python evaluate.py` | Evaluation with ROUGE | Loads saved model, generates summaries on test set, prints ROUGE-1/2/L |
| `uv run streamlit run data_view/data_view.py` | Data viewer app | Opens Streamlit UI to browse train/val/test CSVs |
| `uv run python -m visualize.run_visualizations` | Generate all plots | Saves figures to `results/figures/` |

---

## 1. When you run `main.py`

**File:** `main.py`

**What happens:**

1. Adds project root to `sys.path` so `import model` works.
2. Sets up logging (timestamp, level, message).
3. Calls `main()` which only **prints instructions**:
   - How to run training
   - How to run evaluation
   - How to run the data viewer

**Related code:**

- All logic lives in this single file.
- It does **not** load data, build the model, or train. It is only an entry point that tells you which commands to run.

**When to use:** When you want a reminder of the available commands.

---

## 2. When you run `train.py`

**File:** `train.py`

**What happens (step by step):**

1. **Setup**  
   Configures logging so you see clear messages in the console.

2. **Load data**  
   Uses `model.dataset.load_train_val_test()`  
   → reads `data/train.csv`, `data/validation.csv`, `data/test.csv`  
   → applies row limits from `model.config` (e.g. `MAX_TRAIN_SAMPLES`, `MAX_VAL_SAMPLES`).

3. **Data stats**  
   Computes article/abstract length (word count) on train and logs mean and max.

4. **Tokenizer**  
   Uses `model.dataset.build_tokenizer()`  
   → fits on train articles + abstracts  
   → adds `<start>` and `<end>` tokens.

5. **Sequences**  
   Uses `model.dataset.prepare_sequences()`  
   → turns articles into encoder sequences (padded to `MAX_ARTICLE_LEN`)  
   → turns abstracts into decoder input/target (padded to `MAX_SUMMARY_LEN`).

6. **Model**  
   Uses `model.main_model.build_model(vocab_size)`  
   → builds encoder–decoder LSTM + attention  
   → compiles with `adam`, `sparse_categorical_crossentropy`, `accuracy`.

7. **Training**  
   `model.fit(...)` with:
   - `EarlyStopping` (stops if validation loss does not improve for a few epochs)
   - `ModelCheckpoint` (saves best weights to `results/best_model.keras`).

8. **Save artifacts**  
   - `results/best_model.keras` — best model weights  
   - `results/tokenizer.pkl` — tokenizer (for evaluate and inference)  
   - `results/training_history.json` — loss/accuracy per epoch (for plotting).

**Related code (which file does what):**

| What | File |
|------|------|
| Paths, lengths, batch size, epochs | `model/config.py` |
| Load CSVs, tokenizer, prepare_sequences | `model/dataset.py` |
| Build encoder–decoder + attention | `model/main_model.py` |
| Orchestration, fit, callbacks, saving | `train.py` |

---

## 3. When you run `evaluate.py`

**File:** `evaluate.py`

**What happens (step by step):**

1. **Setup**  
   Configures logging.

2. **Load model and tokenizer**  
   - Reads `results/tokenizer.pkl` (saved by `train.py`).  
   - Builds the same model with `model.main_model.build_model(vocab_size)`.  
   - Loads weights from `results/best_model.keras`.

3. **Load test (or validation) data**  
   Uses `model.dataset.load_train_val_test()`  
   → if test is empty, uses validation set.

4. **Generate summaries**  
   For each sample (up to 100 by default):  
   - Uses `decode_sequence()`: greedy decoding step-by-step  
   - Compares prediction to reference abstract.

5. **ROUGE**  
   Uses `rouge_score.RougeScorer`  
   → computes ROUGE-1, ROUGE-2, ROUGE-L  
   → logs a few example summaries and average ROUGE F scores.

**Related code (which file does what):**

| What | File |
|------|------|
| Paths, MAX_ARTICLE_LEN, MAX_SUMMARY_LEN, tokens | `model/config.py` |
| load_train_val_test, prepare_sequences, get_vocab_size | `model/dataset.py` |
| build_model | `model/main_model.py` |
| load tokenizer/model, decode_sequence, ROUGE loop | `evaluate.py` |

---

## 4. Data viewer — `data_view/data_view.py`

**What happens when you run:**  
`uv run streamlit run data_view/data_view.py`

- Streamlit starts a web app.
- Reads CSVs from `data/` (path resolved from project root).
- Lets you choose train/validation/test, limit rows, and browse the table.
- Shows column info and row detail (article/abstract text).

**Related code:** All in `data_view/data_view.py`; it does not import from `model/`.

---

## 5. Visualizations — `visualize/run_visualizations.py`

**What happens when you run:**  
`uv run python -m visualize.run_visualizations`

- Creates `results/figures/` if needed.
- **01_model_architecture.png** — diagram of encoder → decoder → attention → output (uses `model/config` for labels).
- **02_training_history.png** — loss and accuracy curves (from `results/training_history.json`; only exists after training).
- **03_config_summary.png** — bar chart of main config values from `model/config.py`.
- **04_data_lengths.png** — histograms of article and abstract length from a sample of `data/train.csv`.

You do **not** need to run training first for config and data plots; the training plot appears only after you have run `train.py`.

---

## Folder layout summary

```
karas-model-dev/
├── main.py              # Entry point; prints commands
├── train.py             # Full training; uses model.config, model.dataset, model.main_model
├── evaluate.py          # ROUGE evaluation; uses model.* + results/best_model.keras, tokenizer.pkl
├── PROJECT_GUIDE.md     # This file
├── data/                # train.csv, validation.csv, test.csv
├── data_view/
│   └── data_view.py     # Streamlit data browser
├── model/
│   ├── config.py        # All constants (paths, lengths, batch size, etc.)
│   ├── dataset.py       # Load CSVs, tokenizer, prepare_sequences
│   └── main_model.py    # build_model() — seq2seq + attention
├── visualize/
│   └── run_visualizations.py  # Matplotlib figures → results/figures/
└── results/             # Created by train.py and visualize
    ├── best_model.keras
    ├── tokenizer.pkl
    ├── training_history.json
    └── figures/         # Created by visualize.run_visualizations
        ├── 01_model_architecture.png
        ├── 02_training_history.png
        ├── 03_config_summary.png
        └── 04_data_lengths.png
```

---

## Order of operations (recommended)

1. **View data:** `uv run streamlit run data_view/data_view.py`  
2. **Train:** `uv run python train.py`  
3. **Visualize:** `uv run python -m visualize.run_visualizations`  
4. **Evaluate:** `uv run python evaluate.py`
