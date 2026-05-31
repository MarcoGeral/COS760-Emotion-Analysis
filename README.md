# Multilabel Emotion Classification in Low-Resource African Languages using Data Augmentation

This project investigates whether data augmentation (back-translation and paraphrasing) can improve multilabel emotion classification for low-resource African languages. Two pretrained multilingual models, **XLM-RoBERTa-large** and **AfroXLMR-large**, are fine-tuned across four experimental data conditions on **Afrikaans**, **Hausa**, and **Igbo** using the BRIGHTER benchmark.

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

## Contents of the Zip File

```text
Group6/
├── notebooks/
│   ├── Hausa.ipynb             # Full augmentation pipeline for Hausa (Conditions A-D)
│   ├── Igbo.ipynb              # Full augmentation pipeline for Igbo (Conditions A-D)
│   └── Afrikaans.ipynb         # Full augmentation pipeline for Afrikaans (Conditions A-D)
├── results/
│   ├── results_hausa.txt       # Raw experiment output for Hausa
│   ├── results_igbo.txt        # Raw experiment output for Igbo
│   └── results_afrikaans.txt   # Raw experiment output for Afrikaans
├── report/
│   ├── COS760_Report.pdf       # Full project report
│   └── COS760_Report.zip       # Report source files
├── .gitignore
├── requirements.txt            # Python dependencies with versions
└── README.md
```

---

## Setup Instructions

### Requirements

- **Python 3.11**
- CUDA-capable GPU strongly recommended (experiments were run on Google Colab A100/T4)
- ~15 GB free disk space (models + augmented datasets)

### Install dependencies

```bash
pip install -r requirements.txt
```

---

## Data Information

This project uses the **BRIGHTER** dataset, publicly available on Hugging Face under a **CC-BY 4.0** licence. The dataset is downloaded automatically at runtime inside each notebook via the Hugging Face `datasets` library — no manual download is required.

The three language subsets used are:

| Language | HuggingFace Split | Train | Dev | Test |
| :--- | :--- | :---: | :---: | :---: |
| Afrikaans | `semeval2025-task11-brighter` | 1,222 | 196 | 2,130 |
| Hausa | `semeval2025-task11-brighter` | 2,145 | 712 | 2,160 |
| Igbo | `semeval2025-task11-brighter` | 2,880 | 958 | 2,888 |

> **Note:** Do not include dataset files in the zip. They are fetched at runtime and are too large for direct inclusion.

---

## Running the Code

### Option 1 — Google Colab (recommended)

1. Upload the relevant notebook from `notebooks/` to [Google Colab](https://colab.research.google.com/).
2. Set the runtime to **GPU** (Runtime → Change runtime type → T4 or A100).
3. Run all cells top to bottom.

### Option 2 — Local Jupyter

```bash
git clone https://github.com/<your-org>/COS760-Emotion-Analysis.git
cd COS760-Emotion-Analysis
pip install -r requirements.txt
jupyter notebook notebooks/Hausa.ipynb
```

### Notebooks

| Notebook | Language | Description |
| :--- | :--- | :--- |
| [`notebooks/Hausa.ipynb`](notebooks/Hausa.ipynb) | Hausa | Full 4-condition augmentation pipeline |
| [`notebooks/Igbo.ipynb`](notebooks/Igbo.ipynb) | Igbo | Full 4-condition augmentation pipeline |
| [`notebooks/Afrikaans.ipynb`](notebooks/Afrikaans.ipynb) | Afrikaans | Full 4-condition augmentation pipeline |

Each notebook is self-contained and will:

1. Download the BRIGHTER dataset splits from Hugging Face Hub.
2. Generate augmented training sets (Conditions B, C, D).
3. Fine-tune the model independently for each condition.
4. Evaluate on the held-out test set and report macro, micro, and weighted F1.

> **Condition C (Paraphrasing) note:** This condition uses a local LLM for paraphrase generation. By default it expects **Ollama** with the **gemma4** model installed locally. If you do not have Ollama set up, configure your API keys and LLM settings in the relevant notebook section before running.

> **Tip:** Augmentation generation is the most time-intensive step. Look for cells marked `# SAVE / LOAD AUGMENTED DATA` to save intermediate outputs and avoid re-running generation.

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

## Dependencies

See [`requirements.txt`](requirements.txt) for the full pinned list. Core packages:

| Package | Version | Purpose |
| :--- | :--- | :--- |
| `torch` | 2.3.0 | PyTorch backend |
| `transformers` | 4.41.0 | Model fine-tuning and NLLB-200 inference |
| `datasets` | 2.19.0 | BRIGHTER dataset loading from Hugging Face Hub |
| `accelerate` | 0.30.0 | Hugging Face Trainer multi-GPU support |
| `scikit-learn` | 1.5.0 | F1 score computation |
| `sentencepiece` | 0.2.0 | NLLB-200 tokenization |
| `numpy` | 1.26.4 | Data manipulation |
| `pandas` | 2.2.2 | Data manipulation |

---

