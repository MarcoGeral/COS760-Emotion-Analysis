# Multilabel Emotion Classification in Low-Resource African Languages using Data Augmentation

This repository contains the official implementation and experimental pipelines for evaluating the impact of data augmentation techniques on multilabel emotion classification. Our research focuses specifically on mitigating the performance degradation caused by data scarcity in low-resource African languages.

Using **AfroXLMR-base** as our core model, we systematically evaluate three diverse typographic languages: **Hausa**, **Igbo**, and **Afrikaans**.

## 👥 Authors (Group 6)
* **Marco Geral**
* **Chisom Emekpo**
* **Sean Maritz**

---

## 📌 Research Overview

Emotion analysis (EA) identifies fine-grained, non-mutually exclusive emotional states (e.g., *joy, sadness, anger, fear, surprise, disgust*). Modern transformer models require vast amounts of annotated data to establish robust decision boundaries for this multi-label task. This project investigates whether expanding lexical diversity and syntactic structure via synthetic text generation can regularize embeddings and minimize overfitting for resource-constrained setups.

### Experimental Conditions
We expose each language independently to four tightly controlled data boundaries to isolate downstream performance gains:
1. **Condition A (Baseline - 1x):** Control environment using only the original human-annotated datasets.
2. **Condition B (Back-Translation - 2x):** Original data appended with English-pivoted (`eng_Latn`) back-translations.
3. **Condition C (Paraphrasing - 2x):** Original data appended with stochastic-sampled paraphrased permutations.
4. **Condition D (Combined Ensemble - 4x):** A dense pooled dataset stacking Original data + English Back-Translation + French Back-Translation + Stochastic Paraphrasing.

---

## 🛠️ Methodology & Technical Architecture

### Model Setup
The model fine-tunes **AfroXLMR-base** (`Davlan/afro-xlmr-base`), which is a language-adapted variant built upon the XLM-RoBERTa architecture, post-pretrained on heavily curated, African-centric corpora. 
* **Classification Head:** Linear projection layer mapping hidden states to $N=6$ target emotion classes.
* **Activation Layer:** Element-wise independent **Sigmoid** functions to allow overlapping predictions.
* **Loss Function:** Binary Cross-Entropy (BCE) with Logits Loss.

### Augmentation Framework
Synthetic generation is powered by the **NLLB-200** (No Language Left Behind) distilled 600M parameter sequence-to-sequence model (`facebook/nllb-200-distilled-600M`).
* **Back-Translation (Lexical Variation):** Maps text sequentially: $L_{\text{src}} \rightarrow L_{\text{pivot}} \rightarrow L_{\text{src}}'$. High-resource pivots include English and French (`fra_Latn`).
* **Paraphrasing (Stochastic Decoding):** Implements **Nucleus (Top-$p$) Sampling** combined with **Top-$k$ Filtering** ($k=50$, $p=0.95$) on the reverse decoding sequence to actively trigger high-probability synonyms instead of deterministic greedy strings.

---

## 📊 Dataset Configurations & Size Metrics

The underlying text corpora is extracted from the **BRIGHTER** dataset benchmark ($100\text{k}$ total instances across $28$ languages). Below is the dataset matrix deployed for each language profile:

| Condition | Augmentation Strategy | Multiplier | Training Size (Samples per Language) |
| :--- | :--- | :---: | :---: |
| **Condition A** | Baseline Control (No Augmentation) | $1\times$ | 2,145 |
| **Condition B** | Back-Translation (`eng_Latn`) | $2\times$ | 4,290 |
| **Condition C** | Stochastic Paraphrasing Only | $2\times$ | 4,290 |
| **Condition D** | Combined Ensemble (BT + Para) | $4\times$ | 8,580 |

---

## ⚙️ Core Hyperparameters

All execution pipelines utilize the following fixed optimization configurations managed through the Hugging Face `Trainer` API:

* **Optimizer:** `AdamW`
* **Epochs:** 10
* **Batch Size:** 16 (per-device)
* **Weight Decay:** 0.01
* **Learning Rate Schedule:** Cosine decay scheduler
  * *Condition A & B:* $5 \times 10^{-5}$
  * *Condition C & D:* $3 \times 10^{-5}$ (conservatively tuned for larger step matrices)
* **Evaluation Metric:** Macro-Averaged F1-Score (guarantees minority and majority emotion states weigh evenly)

---

## 📁 Repository Structure

```text
├── notebooks/
│   ├── Hausa_Augmentation_Pipeline.ipynb     # Complete Hausa execution sheet
│   ├── Igbo_Augmentation_Pipeline.ipynb      # Complete Igbo execution sheet
│   └── Afrikaans_Augmentation_Pipeline.ipynb # Complete Afrikaans execution sheet
├── data/
│   └── .gitkeep                              # Placeholder for downloaded datasets
├── README.md                                 # Project documentation
└── requirements.txt                          # Python dependencies
|__ Report.zip
