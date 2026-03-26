# T5-Small PubMed Summarizer

Fine-tune Google's **T5-Small** transformer model on PubMed biomedical article-abstract pairs.
This folder is a self-contained pipeline: train → evaluate → demo → interactive app.

---

## Project Context

This repository contains **two separate summarization models** built on the same dataset:

| | Existing model (`model/`) | This folder (`t5model/`) |
|---|---|---|
| **Architecture** | Custom Keras seq2seq LSTM + Attention | HuggingFace T5-Small (transformer) |
| **Framework** | TensorFlow / Keras | PyTorch + HuggingFace Transformers |
| **Vocab** | Custom tokenizer (30k words) | T5 SentencePiece tokenizer (built-in) |
| **Approach** | Train from scratch | Fine-tune a pre-trained model |
| **Interactive CLI** | `index.py` (project root) | `t5model/app.py` |

The existing model (`model/config.py`, `model/dataset.py`, `model/main_model.py`, `index.py`) is a **custom encoder-decoder LSTM** with Bahdanau-style attention, trained from scratch using TensorFlow/Keras. It uses a word-level tokenizer (capped at 30k vocab), teacher-forcing during training, and greedy autoregressive decoding at inference time.

This folder replaces the LSTM with **T5-Small**, a pre-trained transformer that already understands language. Fine-tuning it on the same PubMed data typically gives better ROUGE scores with fewer epochs.

---

## Folder Structure

```
t5model/
├── train.py        ← Fine-tune T5-Small on train.csv + validation.csv
├── evaluate.py     ← Compute ROUGE scores on the first 100 rows of test.csv
├── demo.py         ← Run 3 hardcoded medical articles and print summaries
├── app.py          ← Interactive CLI: type an article, get a summary
├── saved_model/    ← Best model checkpoint saved here after training
└── README.md       ← This file
```

Data is loaded from `../data/` (relative to this folder), i.e. the project-level `data/` folder:

```
data/
├── train.csv       (10 000 rows: article, abstract)
├── validation.csv  (  800 rows: article, abstract)
└── test.csv        (  800 rows: article, abstract)
```

---

## Requirements

Install once (from the project root):

```bash
uv add torch transformers rouge-score tqdm pandas sentencepiece
```

> **GPU note**: training on CPU is slow (~several hours for 3 epochs on 10k rows).
> If you have a CUDA GPU, PyTorch will use it automatically.

---

## How to Run

### 1. Train

Fine-tunes T5-Small for 3 epochs and saves the best checkpoint:

```bash
uv run python t5model/train.py
```

What happens:
- Loads `data/train.csv` (10 000 rows) and `data/validation.csv` (800 rows)
- Downloads `t5-small` weights from HuggingFace (~240 MB, one-time)
- Trains for **3 epochs**, batch size **8**, learning rate **5e-5**
- Prints loss every **100 steps** with timestamps
- Saves the best model (lowest validation loss) to `t5model/saved_model/`

Example output:
```
2026-03-26 10:00:00 [INFO] T5-Small Fine-tuning  |  2026-03-26 10:00:00
2026-03-26 10:00:01 [INFO] Train rows: 10000 | Validation rows: 800
Epoch 1/3 [Train]: 100%|████| 1250/1250 [loss=1.8432]
2026-03-26 10:05:00 [INFO] Epoch 1 | Step 100 | Avg Loss: 2.1043
...
2026-03-26 10:30:00 [INFO] *** Best model saved to t5model/saved_model/ (val_loss=1.4201) ***
```

---

### 2. Evaluate

Runs inference on the first 100 rows of `test.csv` and prints ROUGE scores:

```bash
uv run python t5model/evaluate.py
```

What happens:
- Loads the saved model from `t5model/saved_model/`
- Generates summaries for 100 test articles
- Computes **ROUGE-1**, **ROUGE-2**, **ROUGE-L** F1 scores
- Prints 3 sample reference vs. predicted summaries

Example output:
```
ROUGE Scores (on 100 test samples)
  ROUGE-1 : 0.3812
  ROUGE-2 : 0.1547
  ROUGE-L : 0.2943

--- Sample 1 ---
REFERENCE : parkinson s disease is associated with anxiety in 50% of patients...
PREDICTION: anxiety is associated with depression and disease severity in patients with parkinson's disease.
```

> **ROUGE explained**:
> - **ROUGE-1**: overlap of single words (unigrams)
> - **ROUGE-2**: overlap of word pairs (bigrams)
> - **ROUGE-L**: longest common subsequence — captures sentence-level flow

---

### 3. Demo

Runs 3 hardcoded biomedical articles and prints the generated summaries:

```bash
uv run python t5model/demo.py
```

The 3 articles are about:
1. Parkinson's disease and anxiety
2. Type 2 diabetes and empagliflozin
3. MiR-21 in breast cancer

Example output:
```
============================================================
  Article 1: Parkinson's Disease & Anxiety
============================================================
INPUT :
  parkinson s disease is a neurodegenerative disorder affecting dopaminergic neurons ...

OUTPUT:
  anxiety in parkinson's disease is associated with depression and disease severity.
```

---

### 4. Interactive App

Type any medical article and get a summary:

```bash
uv run python t5model/app.py
```

Commands:
- Type or paste article text → press Enter → get summary
- `help` → show usage tips
- `exit` or `quit` → stop

Example session:
```
Article> type 2 diabetes affects 460 million people worldwide . empagliflozin was compared to placebo ...

Summary> empagliflozin significantly reduced hba1c levels compared to placebo in type 2 diabetes patients.
```

---

## How the T5 Model Works

### Architecture (T5-Small)
T5 (Text-to-Text Transfer Transformer) treats every NLP task as a text-to-text problem.
For summarization, input is prefixed with `"summarize: "` and the model generates the summary token-by-token.

```
Input:  "summarize: type 2 diabetes affects 460 million people ..."
Output: "empagliflozin significantly reduced hba1c levels ..."
```

T5-Small has ~60M parameters across:
- **Encoder**: 6 transformer layers with self-attention
- **Decoder**: 6 transformer layers with self + cross-attention over encoder output
- **Vocabulary**: ~32k SentencePiece tokens (subword units)

### Fine-tuning vs Training from Scratch
The existing `model/` LSTM trains entirely from random weights on PubMed data.
T5-Small has already learned general language from C4 (750GB web text).
Fine-tuning updates those weights slightly to specialise on the summarization task — usually reaching better ROUGE scores in fewer steps.

### Input/Output Limits
| Parameter | Value |
|---|---|
| Max input tokens | 512 |
| Max output tokens | 128 |
| Prefix | `"summarize: "` |
| Decoding | Beam search, beam=4 |

### Loss Function
T5 uses **cross-entropy loss** over the decoder's predicted token probabilities.
Padding tokens are masked with `-100` so they don't contribute to the loss.

### Saving Strategy
Only the epoch with the **lowest validation loss** is saved. This prevents overfitting: later epochs may have lower training loss but higher validation loss.

---

## Comparison: Existing LSTM vs T5-Small

| Aspect | LSTM (`model/`) | T5-Small (`t5model/`) |
|---|---|---|
| Parameters | ~15M | ~60M |
| Pre-training | None (random init) | C4 dataset (750GB) |
| Tokenizer | Word-level (30k cap) | SentencePiece subwords |
| Max input | 600 tokens | 512 tokens |
| Max output | 150 tokens | 128 tokens |
| Decoding | Greedy step-by-step | Beam search (beam=4) |
| Training time (CPU) | ~5 hours (15 epochs) | ~3-6 hours (3 epochs) |
| Expected ROUGE-1 | ~0.20–0.28 | ~0.35–0.42 |

---

## Troubleshooting

**`saved_model/ not found or empty`**
→ Run `uv run python t5model/train.py` first.

**`Data file not found: .../data/train.csv`**
→ Make sure you run scripts from the project root (`d:/karas-model-dev`), not from inside `t5model/`.

**CUDA out of memory**
→ Reduce `BATCH_SIZE` from `8` to `4` in `train.py` (line with `BATCH_SIZE = 8`).

**Slow training on CPU**
→ Expected. T5-Small on 10k rows × 3 epochs takes 3–6 hours on a laptop CPU.
   Reduce `MAX_ROWS` in `train.py` or use a GPU/Colab environment.

**`No module named 'rouge_score'`**
→ Run `uv add rouge-score`.

**`No module named 'sentencepiece'`**
→ Run `uv add sentencepiece`.
