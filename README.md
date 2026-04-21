# 🛡️ Safe Tweet Detector
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

| Model | Accuracy (Imbal.) | Weighted F1 (Imbal.) | Accuracy (Bal.) | Macro F1 (Bal.) |
|-------|:-----------------:|:--------------------:|:---------------:|:---------------:|
| **Logistic Regression** | **0.64** | **0.66** | **0.67** | **0.66** |
| LinearSVC | 0.64 | 0.66 | 0.65 | 0.64 |
| LSTM | 0.60 | 0.63 | 0.63 | 0.63 |

> **Key finding:** Logistic Regression achieves the highest raw accuracy and weighted F1. The corrected LSTM achieves a marginally higher macro-averaged ROC-AUC, indicating superior probability calibration — a meaningful distinction for real-world deployment where confidence scores matter.

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
TF-IDF Features     Tokenisation + Padding
   │                     │
   ├── Logistic      LSTM (Keras)
   │   Regression    ├── Embedding (128-dim)
   └── LinearSVC     ├── LSTM (64 units)
                     ├── Dropout (0.3)
                     └── Dense (softmax, 4 classes)
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
- **Vectoriser:** TF-IDF, n-gram range (1, 3), max 50,000 features
- **Solver:** `lbfgs`, `max_iter=1000`, `C=1.0`
- **Rationale:** Strong TF-IDF baseline; Davidson et al. (2017) demonstrated ~91% accuracy on similar tasks

### LinearSVC
- **Vectoriser:** TF-IDF, unigrams, max 30,000 features
- **Regularisation:** `C=1.0`
- **Rationale:** SVM's large-margin classifier handles high-dimensional sparse text well (Joachims, 1998)

### LSTM Neural Network
- **Vocab size:** 20,000 · **Max sequence length:** 200
- **Architecture:** Embedding(128) → LSTM(64) → Dropout(0.3) → Dense(4, softmax)
- **Training:** Adam optimiser, `EarlyStopping(patience=5, restore_best_weights=True)`
- **Tokeniser & label encoder** fitted on balanced training subset (`X_train_final / y_train_final`)

> ⚠️ **Bug fix documented:** An earlier version incorrectly fitted the tokeniser and label encoder on the full imbalanced corpus rather than the balanced training subset. This substantially degraded Class 2 (Social/Contextual Harm) performance. The corrected version is what is evaluated here — this fix and its impact are discussed as a core finding in the report.

---

## 📁 Repository Structure

```
safe-tweet-detector/
│
├── Safe_Tweet_Detector.ipynb     # Main notebook (Google Colab)
│
├── figures/                      # Output plots and visualisations
│   ├── cm_lr.png                 # Confusion matrix – Logistic Regression
│   ├── cm_svc.png                # Confusion matrix – LinearSVC
│   ├── cm_lstm.png               # Confusion matrix – LSTM
│   ├── roc_curve.png             # ROC curves (macro OvR, all models)
│   ├── train_val_loss_lstm.png   # LSTM training/validation loss
│   └── model_perf_comparison.png # Side-by-side model comparison
│
└── README.md
```

---

## ⚙️ Setup & Usage

### Requirements

```bash
pip install datasets transformers
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
predict_harm(text)

# Example output:
# Logistic Regression → Class 1: Non-violent Harm
# LinearSVC           → Class 1: Non-violent Harm
# LSTM                → Class 1: Non-violent Harm
```

---

## 📈 Evaluation Highlights

- **Confusion matrices** generated for all models on both imbalanced and balanced test sets
- **ROC-AUC** computed using one-vs-rest (OvR) macro averaging
- **Class 2 (Social/Contextual Harm)** is the hardest class: precision as low as 0.16–0.19 on the imbalanced set due to severe under-representation (1,370 samples vs 14,707 for Class 3). Under balanced evaluation, precision recovers to 0.61–0.65, confirming the issue is data-distribution-driven
- **EarlyStopping** applied to LSTM (patience=5) — overfitting confirmed from ~epoch 10 onwards in loss curves

---

## 🔬 Key Findings

1. **Logistic Regression** is the strongest overall performer on raw accuracy and weighted F1, demonstrating that well-tuned classical methods remain highly competitive on large-scale NLP tasks
2. **The LSTM trades accuracy for calibration** — marginally higher macro ROC-AUC suggests better-calibrated probability estimates, which matters in deployment contexts where confidence thresholds are used
3. **Class imbalance is the dominant challenge**, not model capacity — balanced evaluation substantially closes the gap between models and improves Class 2 detection
4. **LinearSVC** and Logistic Regression perform near-identically overall, but LR has a slight macro F1 edge from its trigram feature range

---

## 🚀 Future Work

- Fine-tuned **BERT / RoBERTa** for transformer-based comparison
- **SMOTE** or cost-sensitive learning to address Class 2 imbalance more aggressively
- **Deployment** as a REST API or browser extension for real-time content moderation
- Ethical audit of model outputs for demographic bias across protected characteristics

---

## 📚 References

- Ji, J. et al. (2024). *BeaverTails: Towards Improved Safety Alignment of LLM via a Human-Preference Dataset*. PKU-Alignment.
- Davidson, T. et al. (2017). Automated Hate Speech Detection and the Problem of Offensive Language. *ICWSM*.
- Hochreiter, S. & Schmidhuber, J. (1997). Long Short-Term Memory. *Neural Computation*, 9(8).
- Devlin, J. et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. *arXiv:1810.04805*.
- Lemaître, G. et al. (2017). Imbalanced-learn: A Python Toolbox. *JMLR*, 18(17).

---

## 📄 Licence

This project was submitted in partial fulfilment of the MSc Data Science degree at the University of Hertfordshire. Academic use only.
