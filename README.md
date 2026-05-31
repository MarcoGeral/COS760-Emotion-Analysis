# Multilabel Emotion Classification in Low-Resource African Languages using Data Augmentation

This project investigates whether data augmentation (back-translation and paraphrasing) can improve multilabel emotion classification for low-resource African languages. Two pretrained multilingual models — **XLM-RoBERTa-large** and **AfroXLMR-large** — are fine-tuned across four experimental data conditions on **Afrikaans**, **Hausa**, and **Igbo** using the BRIGHTER benchmark.

> **Full report:** [`COS760_Report.pdf`](report/COS760_Report.pdf)

---

## Authors (Group 6)

| Name |
| :--- |
| Marco Geral |
| Chisom Emekpo |
| Sean Maritz |

---

## Research Questions

1. How can data augmentation strategies (such as back-translation and paraphrasing) improve multilabel emotion classification for African languages?
2. How can existing NLP methods be adapted or applied effectively in low-resource African language contexts for emotion analysis?

---

## Experimental Conditions

| Condition | Augmentation Strategy | Afrikaans | Hausa | Igbo |
| :--- | :--- | :---: | :---: | :---: |
| **A — Baseline** | No augmentation | 1,222 | 2,945 | 3,680 |
| **B — Back-Translation** | High-resource pivot translations | 3,030 | 4,417 | 3,680 |
| **C — Paraphrasing** | Stochastic paraphrasing | 3,033 | 4,417 | 5,516 |
| **D — Combined** | Back-translation + paraphrasing | 4,041 | 5,889 | 7,346 |

Models are evaluated on held-out test sets (Afrikaans: 2,130 / Hausa: 2,160 / Igbo: 2,888 instances) using **macro-averaged F1** as the primary metric.

---

## Results (Macro F1)

**Afrikaans**

| Condition | XLM-RoBERTa-large | AfroXLMR-large |
| :--- | :---: | :---: |
| A — Baseline | 0.5996 | 0.6333 |
| B — Back-Translation | 0.5732 | 0.5955 |
| C — Paraphrasing | 0.6768 | 0.6455 |
| D — Combined | 0.6531 | **0.7302** |

**Hausa**

| Condition | XLM-RoBERTa-large | AfroXLMR-large |
| :--- | :---: | :---: |
| A — Baseline | 0.5944 | 0.6793 |
| B — Back-Translation | 0.6403 | 0.6003 |
| C — Paraphrasing | 0.6318 | **0.7125** |
| D — Combined | 0.6458 | 0.7010 |

**Igbo**

| Condition | XLM-RoBERTa-large | AfroXLMR-large |
| :--- | :---: | :---: |
| A — Baseline | 0.4500 | 0.4858 |
| B — Back-Translation | 0.4347 | 0.4855 |
| C — Paraphrasing | 0.4841 | **0.5045** |
| D — Combined | 0.4925 | 0.4554 |

**Key finding:** Paraphrasing (Condition C) is the most consistent augmentation strategy across all three languages. Back-translation showed high variance and actively degraded performance in several configurations due to semantic drift through pivot languages.

---

## Repository Structure

```text
COS760-Emotion-Analysis/
├── notebooks/
│   ├── Hausa.ipynb        # Hausa: Conditions A, B, C, D
│   ├── Igbo.ipynb         # Igbo: Conditions A, B, C, D
│   └── Afrikaans.ipynb    # Afrikaans: Conditions A, B, C, D
├── report/
│   ├── COS760_Report.pdf  # Full project report
│   └── COS760_Report.zip  # Report source files
├── .gitignore
├── requirements.txt       # Python dependencies
└── README.md
```

---

## How to Run

### Prerequisites

- Python ≥ 3.9
- CUDA-capable GPU strongly recommended (experiments run on Google Colab A100/T4)
- ~15 GB free disk space (models + augmented datasets)

### 1. Clone the repository

```bash
git clone https://github.com/<your-org>/COS760-Emotion-Analysis.git
cd COS760-Emotion-Analysis
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run a notebook

Upload the notebook for your target language to **Google Colab**, or run locally in Jupyter:

| Notebook | Language |
| :--- | :--- |
| [`notebooks/Hausa.ipynb`](notebooks/Hausa.ipynb) | Hausa |
| [`notebooks/Igbo.ipynb`](notebooks/Igbo.ipynb) | Igbo |
| [`notebooks/Afrikaans.ipynb`](notebooks/Afrikaans.ipynb) | Afrikaans |

Each notebook is self-contained and will:

1. Download the BRIGHTER dataset splits from Hugging Face Hub.
2. Generate augmented training sets (Conditions B, C, D).
3. Fine-tune the model for each condition independently.
4. Evaluate on the held-out test set and report macro, micro, and weighted F1.

> **Tip:** Augmentation generation is the most time-intensive step. Look for cells marked `# SAVE / LOAD AUGMENTED DATA` to save intermediate outputs and avoid re-running generation.

---

## Dependencies

See [`requirements.txt`](requirements.txt) for the full list. Core packages:

| Package | Purpose |
| :--- | :--- |
| `transformers` | Model fine-tuning and NLLB-200 inference |
| `datasets` | BRIGHTER dataset loading from Hugging Face Hub |
| `torch` | PyTorch backend |
| `scikit-learn` | F1 score computation |
| `numpy` / `pandas` | Data manipulation |
| `sentencepiece` | NLLB-200 tokenization |
| `accelerate` | Hugging Face Trainer multi-GPU support |

---

