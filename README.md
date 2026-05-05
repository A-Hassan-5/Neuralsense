# 🧠 Neural-Sense
### Sentiment Classification of Consumer Reviews using RNNs / LSTMs

---

## Project Structure

```
neural-sense/
├── neural_sense.py         ← All Colab cell code (Optimized for Google Colab)
├── app.py                  ← Streamlit inference app (run locally)
├── requirements.txt        ← System requirements for the streamlit interface
└── README.md
```

After running the Colab notebook you will also have:
```
├── neural_sense_model.h5           ← Trained BiLSTM model
└── neural_sense_meta.npz           ← vocab_size + max_len metadata
```

---

## Part 1 — Google Colab Notebook

### What it does (16 cells)

| Cell | Contents |
|------|----------|
| 1  | Install & import all dependencies |
| 2  | Global hyperparameter config |
| 3  | Load & summarise the IMDb dataset |
| 4  | EDA — length distribution, class balance, top-words |
| 5  | Text preprocessing (tokenise + pad) |
| 6  | Build **SimpleRNN** baseline model |
| 7  | Train SimpleRNN with EarlyStopping + LR scheduler |
| 8  | Build **Bidirectional LSTM** (main model) |
| 9  | Train BiLSTM with checkpointing |
| 10 | Training curves for both models |
| 11 | Confusion matrices |
| 12 | ROC & Precision-Recall curves |
| 13 | Side-by-side comparison table |
| 14 | Qualitative testing (sarcasm / tricky sentences) |
| 15 | Word embedding PCA visualisation |
| 16 | Save model + confirm output files |

### How to use in Colab

1. Open [colab.research.google.com](https://colab.research.google.com)
2. **Runtime → Change runtime type → GPU** (T4 is enough)
3. Create a new notebook
4. Open `NeuralSense_Colab_Notebook.py` and copy each `CELL N` block  
   into a separate Colab code cell (strip the surrounding triple-quotes)
5. Run top-to-bottom with **Shift+Enter**
6. After Cell 16, download both files from the **Files panel** (left sidebar):
   - `neural_sense_model.h5`
   - `neural_sense_meta.npz`

---

## Part 2 — Streamlit App (local)

### Requirements

```bash
pip install -r requirements.txt
```

### Setup

Place these three files in the **same folder**:

```
your_folder/
├── app.py
├── neural_sense_model.h5       ← from Colab
└── neural_sense_meta.npz       ← from Colab
```

### Run

```bash
streamlit run app.py
```

Opens automatically at `http://localhost:8501`

### Features

- **Live sentiment prediction** — type any sentence, get Positive / Negative / Neutral
- **Confidence bar** — visual probability breakdown
- **Adjustable thresholds** — sidebar sliders for pos/neg cutoffs
- **Example sentences** — one-click sarcastic / tricky test cases
- **Session history** — last 10 predictions shown
- **Live stats** — running counts of Positive / Negative / Mixed

---

## Model Architecture

```
Input (max_len=256 tokens)
    │
Embedding (vocab=20 000 → 128-dim, trainable)
    │
SpatialDropout1D (0.2)
    │
Bidirectional LSTM (128 units, return_sequences=True)
    │
Bidirectional LSTM (64 units)
    │
Dense (64, ReLU) + BatchNorm + Dropout (0.4)
    │
Dense (32, ReLU)
    │
Dense (1, Sigmoid)  →  P(Positive)
```

**Loss:** Binary Cross-Entropy  
**Optimiser:** Adam (lr=1e-3, ReduceLROnPlateau)  
**Stopping:** EarlyStopping on val_AUC (patience=4)

---

## Key Design Decisions

| Decision | Reason |
|----------|--------|
| Bidirectional LSTM | Reads sequence both forward & backward — captures context before and after each word |
| Two stacked LSTM layers | Deeper representation; first layer extracts local patterns, second global |
| SpatialDropout1D | Drops entire feature maps rather than individual values — more effective for embeddings |
| Trainable embeddings | Domain-specific — movie vocabulary differs from generic corpora |
| BatchNorm in dense head | Stabilises training, allows higher learning rate |
| SimpleRNN baseline | Demonstrates vanishing gradient limitation; motivates the LSTM choice |

---

## Expected Results

| Metric | SimpleRNN | BiLSTM |
|--------|-----------|--------|
| Test Accuracy | ~50 % | ~87–89 % |
| AUC-ROC | ~0.53 | ~0.95 |
| F1-Score | ~0.45 | ~0.88 |

---
