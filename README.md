# Safe Tweet Detector
### Multi-Class Harm Detection using Machine Learning and Deep Learning

> **MSc Data Science Final Project** · University of Hertfordshire · Module 7PAM2002  
> **Student:** Stephen Ogundero (SRN: 24086166) · **Supervisor:** Dhairya Kataria

---

## 📌 Overview

Social media platforms generate billions of interactions daily, making automated harm detection a critical challenge. The **Safe Tweet Detector** is a multi-class text classification system that automatically assigns social media content to one of four harm severity levels, going beyond simple binary safe/unsafe filtering to provide actionable, fine-grained harm categorisation.

The project implements and compares three models — **Logistic Regression**, **LinearSVC**, and an **LSTM neural network** — trained on the large-scale [BeaverTails](https://huggingface.co/datasets/PKU-Alignment/BeaverTails) dataset.

---

## 🏷️ Harm Categories

| Class | Label | Description |
|:-----:|-------|-------------|
| **0** | Severe Harm | Violence/incitement, terrorism, self-harm, child abuse, sexually explicit |
| **1** | Non-violent Harm | Misinformation, privacy violations, financial crime, drug/weapons references |
| **2** | Social/Contextual Harm | Hate speech, discrimination, offensive language, controversial politics |
| **3** | Safe | No harmful categories present |

---

## 📊 Key Results

| Model | Accuracy (Imbal.) | Weighted F1 (Imbal.) | Accuracy (Bal.) | Macro F1 (Bal.) | ROC-AUC |
|-------|:-----------------:|:--------------------:|:---------------:|:---------------:|:-------:|
| **Logistic Regression** | **0.64** | **0.66** | **0.67** | **0.66** | **0.876** |
| LinearSVC | 0.64 | 0.66 | 0.65 | 0.64 | 0.855 |
| LSTM | 0.60 | 0.63 | 0.63 | 0.63 | 0.853 |

> **Key finding:** Logistic Regression achieves the highest raw accuracy and weighted F1. The corrected LSTM achieves a marginally higher macro-averaged ROC-AUC than LinearSVC, indicating superior probability calibration — a meaningful distinction for real-world deployment where confidence scores matter.

---

## 🗂️ Dataset

**[BeaverTails](https://huggingface.co/datasets/PKU-Alignment/BeaverTails)** (Ji et al., 2024) — released by PKU-Alignment.

| Split | Records |
|-------|---------|
| Training | 330,254 |
| Test | 3,709 |
| **Total** | **333,963** |

Each record contains a `prompt`, a `response`, a 14-category harm label dictionary, and a binary `is_safe` flag. The 14 BeaverTails categories are consolidated into the 4-class scheme above.

---

## 🔧 Pipeline

```
BeaverTails Dataset
        │
        ▼
  Data Preprocessing
  ├── full_text = prompt + " " + response
  ├── Lowercasing, URL removal, punctuation removal
  ├── Stopword filtering (preserving: not, no, how)
  └── WordNet lemmatisation → clean_prompt
        │
        ▼
  Multi-Class Labelling (14 → 4 classes, priority hierarchy)
        │
        ▼
  Class Balancing (RandomUnderSampler → 12,881 per class)
        │
        ▼
  Train/Test Split
        │
   ┌────┴────────────────┐
   ▼                     ▼
TF-IDF Features     Learning Rate Search (0.01 / 0.001 / 0.0001)
   │                     │ best LR = 0.001
   ├── Logistic      LSTM (Keras)
   │   Regression    ├── Embedding (128-dim)
   └── LinearSVC     ├── LSTM (128 units)
                     ├── Dropout (0.5)
                     ├── Dense (64, ReLU)
                     └── Dense (4, Softmax)
        │
        ▼
  Evaluation: Accuracy · F1 · Confusion Matrix · ROC-AUC (OvR)
        │
        ▼
  Real-time Inference Function
```

---

## 🧠 Model Architectures

### Logistic Regression
- **Vectoriser:** TF-IDF, `ngram_range=(1, 3)`, capturing unigrams, bigrams, and trigrams
- **Estimator:** `LogisticRegression(max_iter=10000)`
- **Rationale:** Trigram TF-IDF is well-suited to harm detection where specific phrases are highly discriminative; Davidson et al. (2017) demonstrated ~91% accuracy on similar tasks

### LinearSVC
- **Vectoriser:** TF-IDF, unigrams, `max_features=20,000`
- **Estimator:** `LinearSVC(class_weight='balanced')`
- **Rationale:** Large-margin classifier handles high-dimensional sparse text well (Joachims, 1998); `class_weight='balanced'` provides built-in imbalance handling

### LSTM Neural Network

**Architecture:**
```
Embedding(vocab=20,000, dim=128, input_len=100)
    → LSTM(128 units)
    → Dropout(0.5)
    → Dense(64, ReLU)
    → Dense(4, Softmax)
```

**Hyperparameter tuning — Learning Rate Search:**

| Learning Rate | Best Val Loss | Best Val Accuracy | Epochs Run | Selected |
|:-------------:|:-------------:|:-----------------:|:----------:|:--------:|
| 0.01 | 0.9757 | 0.6247 | 10 | No |
| **0.001** | **0.9264** | **0.6504** | **20** | **✓ Yes** |
| 0.0001 | 0.9429 | 0.6155 | 50 | No |

Each candidate was trained under identical conditions (`batch_size=8192`, `epochs=50`, `EarlyStopping(patience=5)`). Learning rate `0.001` achieved the lowest validation loss and was selected. LR `0.01` overshot the optimum (converged in 10 epochs to a higher loss); LR `0.0001` was too slow, running all 50 epochs without triggering early stopping.

**Final training:** `Adam(learning_rate=0.001)`, `batch_size=8192`, stopped at **epoch 18** via `EarlyStopping(patience=5, restore_best_weights=True)`.

> ⚠️ **Bug fix documented:** An earlier version fitted the tokeniser and label encoder on the full imbalanced corpus rather than the balanced training subset (`X_train_final`). This substantially degraded Class 2 (Social/Contextual Harm) performance. The corrected version is evaluated here.

---

## 📁 Repository Structure

```
safe-tweet-detector/
│
├── 24086166_SafeTweetDetector_BeaverTails_v2.ipynb   # Main notebook (Google Colab)
│
├── figures/                        # Output plots and visualisations
│   ├── cm_lr.png                   # Confusion matrix – Logistic Regression
│   ├── cm_svc.png                  # Confusion matrix – LinearSVC
│   ├── cm_lstm.png                 # Confusion matrix – LSTM
│   ├── roc_curve.png               # ROC curves (macro OvR, all models)
│   ├── train_val_loss_lstm.png     # LSTM training/validation loss
│   ├── lr_search.png               # Learning rate search bar chart
│   └── model_perf_comparison.png   # Side-by-side model comparison
│
├── README.md
└── .gitignore
```

---

## ⚙️ Setup & Usage

### Requirements

```bash
pip install datasets
pip install scikit-learn imbalanced-learn
pip install tensorflow keras
pip install nltk pandas numpy matplotlib seaborn
```

### NLTK Resources

```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
```

### Load Dataset

```python
from datasets import load_dataset
ds = load_dataset("PKU-Alignment/BeaverTails")
train_df = ds["330k_train"].to_pandas()
test_df  = ds["330k_test"].to_pandas()
```

### Run Inference

```python
# Classify new text with all three models
text = "Your input text here"
predict_new_text(text)

# Example output:
# Cleaned Text: input text here
#
# Logistic Regression → Class 1: Non-violent Harm  (confidence: 0.74)
# LinearSVC           → Class 1: Non-violent Harm
# LSTM                → Class 1: Non-violent Harm  (confidence: 0.71)
```

---

## 📈 Evaluation Highlights

- **Confusion matrices** generated for all models on both imbalanced and balanced test sets
- **ROC-AUC** computed using one-vs-rest (OvR) macro averaging — LR: 0.876, SVC: 0.855, LSTM: 0.853
- **Class 2 (Social/Contextual Harm)** is the hardest class: precision 0.16–0.19 on the imbalanced set (support 1,370 vs 14,707 for Class 3); recovers to 0.61–0.65 under balanced evaluation — data-distribution-driven, not a modelling failure
- **EarlyStopping** applied to LSTM (patience=5) — overfitting observed from ~epoch 10; final model stopped at epoch 18
- **Learning rate tuning** across 0.01 / 0.001 / 0.0001 — LR 0.001 selected (lowest val loss: 0.9264)

---

## 🔬 Key Findings

1. **Logistic Regression** is the strongest overall performer on raw accuracy and weighted F1, demonstrating that well-tuned classical methods remain highly competitive on large-scale NLP tasks
2. **The LSTM trades accuracy for calibration** — marginally higher macro ROC-AUC than LinearSVC suggests better probability estimates, relevant for deployment contexts using confidence thresholds
3. **Class imbalance is the dominant challenge**, not model capacity — balanced evaluation closes the gap between models and improves Class 2 precision from ~0.18 to ~0.63
4. **Learning rate 0.001 outperformed alternatives** — confirmed empirically through a three-way search; Adam's default is well-calibrated for this task but the tuning provides evidence rather than assumption
5. **LinearSVC** and Logistic Regression perform near-identically overall, but LR's trigram feature range gives it a slight macro F1 edge

---

## 🚀 Future Work

- Fine-tuned **BERT / RoBERTa** or **HateBERT** for transformer-based comparison
- **SMOTE** or cost-sensitive learning to address Class 2 imbalance more aggressively
- Extended hyperparameter search — LSTM hidden units, dropout rate, embedding dimension
- **Deployment** as a REST API or browser extension for real-time content moderation
- Ethical audit of model outputs for demographic bias across protected characteristics

---

## 📚 References

- Ji, J. et al. (2024). *BeaverTails: Towards Improved Safety Alignment of LLM via a Human-Preference Dataset*. PKU-Alignment.
- Davidson, T. et al. (2017). Automated Hate Speech Detection and the Problem of Offensive Language. *ICWSM*.
- Kingma, D. & Ba, J. (2015). Adam: A Method for Stochastic Optimization. *ICLR*.
- Hochreiter, S. & Schmidhuber, J. (1997). Long Short-Term Memory. *Neural Computation*, 9(8).
- Devlin, J. et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. *arXiv:1810.04805*.
- Lemaître, G. et al. (2017). Imbalanced-learn: A Python Toolbox. *JMLR*, 18(17).
- Joachims, T. (1998). Text Categorization with Support Vector Machines. *ECML*.

---

## 📄 Licence

This project was submitted in partial fulfilment of the MSc Data Science degree at the University of Hertfordshire. Academic use only.