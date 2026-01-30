# MutEX: Physics-Aware Protein Stability Prediction

 **Live Demo (Streamlit App):**  
 (https://youtu.be/xzl_gndwBBI?si=fPXEIZZsi1yIOpUD)

---


## Overview

MutEX is a **physics-aware machine learning framework** for predicting protein stability changes (**ΔΔG**) caused by **single amino acid mutations**. Unlike many existing predictors that focus purely on statistical accuracy, MutEX is explicitly designed to produce **thermodynamically consistent predictions** by enforcing the **antisymmetry property of ΔΔG** during training.

The guiding principle of this project is simple but critical:

> A biologically useful model must obey the same physical laws as the system it models.

MutEX demonstrates that incorporating physical constraints into modern ML pipelines leads to predictions that are **not only accurate, but scientifically reliable**.

---

## Problem Statement

Protein stability prediction is a core task in protein engineering, drug design, and disease mutation analysis. While modern machine learning models—especially those based on Protein Language Models (PLMs)—have achieved impressive performance, many suffer from a fundamental flaw:

> They fit experimental data well while violating basic thermodynamic principles.

In particular, many models fail to satisfy **antisymmetry**:

* If mutation **A → B** has ΔΔG = +x,
* then the reverse mutation **B → A** must have ΔΔG = −x.

Violating this rule makes predictions **physically inconsistent** and unreliable for real-world biological or engineering applications. MutEX directly addresses this issue by enforcing antisymmetry *within the training objective itself*, rather than treating it as a post-hoc evaluation metric.

---

## Motivation

Protein stability determines whether a protein folds correctly and performs its biological function. Even a single-point mutation can:

* Destabilize folding
* Reduce or abolish activity
* Cause disease or drug resistance

Although experimental ΔΔG measurements are the gold standard, they are:

* **Slow** (weeks to months per mutation)
* **Expensive**
* **Impossible to scale** to millions of possible mutations

MutEX aims to bridge this gap by combining:

* Large-scale experimental data
* Pretrained protein language models
* Physics-aware learning objectives

---

## What is ΔΔG and Why Antisymmetry Matters

* **ΔG**: Gibbs free energy of protein folding
* **ΔΔG**: Change in folding free energy due to mutation

Interpretation:

* ΔΔG < 0 → Stabilizing mutation
* ΔΔG > 0 → Destabilizing mutation

### Antisymmetry Principle

For any mutation pair:

* ΔΔG(WT → Mutant) = −ΔΔG(Mutant → WT)

Many ML models violate this constraint, producing predictions that contradict thermodynamics. MutEX enforces antisymmetry explicitly during training, ensuring **physical validity by construction**.

---

## Dataset

**Source:** K50 dataset (Tsuboyama et al., 2022, Zenodo)

* Raw scale: **851,552 experimentally measured mutations**

### Cleaning and Validation

The raw dataset contained several inconsistencies:

* Wild-type (WT) entries
* Multiple mutations per sample
* Insertions and deletions
* Inconsistent mutation annotations

Cleaning steps:

* Removed WT entries
* Retained only single-point substitutions
* Excluded insertions and deletions
* Standardized mutation notation using regex
* Verified WT and mutant residue correctness

**Final dataset:**

* **375,560 validated single-point mutations**
* **11 curated columns**
* Used for embedding extraction and model training

---

## Exploratory Data Analysis (EDA)

EDA was performed to assess data quality, bias, and modeling risks:

* No missing values or duplicate samples
* ΔΔG distribution centered near 0 with long tails
* Destabilizing mutations dominate (~79%), reflecting biological reality
* Sequence length shows negligible correlation with ΔΔG (Pearson ≈ 0.05)
* Strong cluster-level imbalance across proteins

### Critical Insight: Data Leakage Risk

Random train–test splits allow homologous proteins to appear in both sets, artificially inflating performance. To prevent this, **cluster-level splitting** was enforced throughout model evaluation.

---

## Protein Language Models (PLMs)

MutEX leverages pretrained **ESM2** protein language models to generate contextual embeddings for both wild-type and mutant sequences. These embeddings capture:

* Local residue environments
* Long-range interactions
* Evolutionary constraints

PLMs enable learning stability-relevant features directly from sequence data, eliminating the need for hand-crafted structural descriptors.

---

## Model Architecture

### Two-Phase Training Pipeline

**Phase 1 – Embedding Extraction**

* Generate embeddings for WT and mutant sequences using pretrained ESM2

**Phase 2 – Prediction Network**

* Lightweight attention layers
* Fully connected regression head
* Output: predicted ΔΔG

### Model Variants

#### Model Version 1 (Baseline)

* Uses WT and mutant embeddings directly
* Antisymmetry evaluated post-training
* High predictive accuracy
* **Physically inconsistent predictions**

#### Model Version 2 (Physics-Aware)

* Uses averaged embeddings from the last 4 PLM layers
* Explicit mutation-aware features:

  * Δ embedding (Mutant − WT)
  * |Δ| embedding
  * Cosine similarity
  * L2 distance
* Antisymmetry enforced via regularization loss during training

---

## Results Summary

* Model-1 variants achieve strong Pearson and Spearman correlations but violate antisymmetry
* Model-2 (MV_2.1.1) achieves:

  * **High antisymmetry correlation (~0.86)**
  * Strong reduction in prediction bias
  * Competitive RMSE and MAE

Although Model-2 sacrifices a small amount of raw error performance, it is the **only model that produces physically meaningful ΔΔG predictions**, making it suitable for real scientific use.

---

## Deployment Architecture

* **Client (CPU):** Input parsing and validation via Streamlit
* **Server (GPU):** ESM2-650M embedding extraction via REST API
* **Client (CPU):** Feature construction and ΔΔG prediction
* **Physics check:** Forward and reverse mutation predictions to verify antisymmetry
* **Output:** Results displayed in UI and downloadable as CSV

---

## Team Contributions

**MS Meenakshi**

* Dataset acquisition and validation
* Data cleaning and preprocessing
* Exploratory data analysis and bias assessment

**Abhiram R**

* PLM embedding pipeline
* Feature engineering
* Model architecture design
* Physics-aware training and evaluation

---

## Future Scope

* Extend to longer and multi-domain proteins
* Support multi-point mutation prediction
* Integrate additional thermodynamic constraints
* Explore fine-tuning of PLMs under physics-aware objectives

---

## References

* Pak et al. (2023). *The new mega dataset combined with a deep neural network makes progress in predicting the impact of single mutations on protein stability.* bioRxiv.
* Tsuboyama et al. (2022). *Mega-scale experimental analysis of protein folding stability.* Zenodo.
* Dieckhaus et al. (2024). *Transfer learning to leverage larger datasets for improved prediction of protein stability changes.* PNAS.

---

## Disclaimer

This project is intended for **research and educational purposes only**. Predictions should not be used directly for clinical or therapeutic decision-making without experimental validation.
