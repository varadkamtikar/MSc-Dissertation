<div align="center">

# 🧠 EEG Reward Detection

### AI-Based Detection of Reward-Related Neural Patterns from EEG Signals

*MSc Data Science Dissertation · Technological University Dublin · 2024–25*

---

[![Live Demo](https://img.shields.io/badge/🚀%20Live%20Demo-Open%20Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://msc-dissertation-varadkamtikar.streamlit.app)

---

![Python](https://img.shields.io/badge/Python-3.9-3776AB?style=flat-square&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML%20Pipeline-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![License](https://img.shields.io/badge/License-Academic-8B5CF6?style=flat-square)
![Dataset](https://img.shields.io/badge/Dataset-DEAP%2032%20Subjects-06B6D4?style=flat-square)

</div>

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Live Dashboard](#-live-dashboard)
- [Key Results](#-key-results)
- [Pipeline](#-pipeline)
- [Project Structure](#-project-structure)
- [Local Setup](#-local-setup)
- [Methodology](#-methodology)
- [Dataset](#-dataset)
- [Tech Stack](#-tech-stack)
- [Author](#-author)

---

## 🔬 Overview

> *"Can a strictly controlled, leakage-free EEG pipeline produce reliable classification performance under realistic multi-subject conditions?"*

This repository contains the full implementation of the MSc dissertation: **AI-Based Detection of Reward-Related Neural Patterns from EEG Signals**. The project designs and evaluates a **strictly leakage-free EEG classification pipeline** on the DEAP dataset, detecting reward vs. no-reward neural states across 32 subjects.

A central contribution of this work is demonstrating that **evaluation methodology has a greater impact on reported accuracy than model choice**. Many published EEG studies report 80–95% accuracy due to inadvertent data leakage. This pipeline prevents that, yielding an honest **~54% cross-subject accuracy** — modest but **methodologically sound and fully reproducible**.

### Core Objectives

| # | Objective |
|---|-----------|
| 1 | Design a fully automated end-to-end EEG preprocessing pipeline |
| 2 | Implement **subject-dependent** (within-subject) and **subject-independent** (cross-subject) evaluation frameworks |
| 3 | Prevent data leakage — scaler and SMOTE fitted strictly inside training folds |
| 4 | Compare 6 classical ML classifiers under identical conditions |
| 5 | Report transparent, reproducible benchmarks using Macro-F1 |

---

## 🚀 Live Dashboard

The interactive dissertation dashboard is deployed and publicly accessible:

<div align="center">

### 👉 [msc-dissertation-varadkamtikar.streamlit.app](https://msc-dissertation-varadkamtikar.streamlit.app)

</div>

The dashboard includes:
- **Overview** — KPI cards, research question, and contribution summary
- **Dataset & Pipeline** — DEAP dataset details, preprocessing stages, feature engineering, and epoch retention
- **Subject Analytics** — Per-subject model performance, fold-level detail, confusion matrices, and EEG signal plots
- **Model Comparison** — Cross-subject heatmaps, violin distributions, and plain vs SMOTE comparison
- **Evaluation & Leakage** — Leakage prevention steps, CV strategy comparison, and performance context
- **Subject-Independent Results** — Final held-out test results and hyperparameter tuning
- **Conclusions** — Key findings, contributions, and future directions

---

## 📊 Key Results

### Subject-Independent (Cross-Subject) — Final Test Set

| Model | Test Accuracy | Test Macro-F1 |
|---|:---:|:---:|
| **Gradient Boosting** | **0.5357** | **0.5263** |
| XGBoost | 0.5179 | 0.5101 |
| Random Forest | 0.5071 | 0.4988 |
| Linear SVM | 0.5268 | 0.5198 |
| SVM-RBF | 0.5089 | 0.4912 |
| Logistic Regression | 0.5214 | 0.5143 |

> **Chance level = 50%.** Results above chance are real signal under strict no-leakage, cross-subject conditions.

### Subject-Dependent (Within-Subject) — 5-Fold CV Average

| Metric | Plain | SMOTE |
|---|:---:|:---:|
| Avg Accuracy | ~0.63 | ~0.61 |
| Avg Macro-F1 | ~0.62 | ~0.60 |

> Best individual subjects reach Macro-F1 of ~0.72; most challenging subjects drop to ~0.42 — reflecting genuine inter-subject EEG variability.

---

## ⚙️ Pipeline

```
Raw DEAP EEG (.bdf/.mat)
        │
        ▼
┌─────────────────────────────┐
│  1. Bandpass Filter 1–40 Hz │
│  2. Average Re-referencing  │
│  3. ICA Artefact Removal    │
│  4. Event Detection         │
│  5. Epoch Extraction (60 s) │
└─────────────────────────────┘
        │
   ┌────┴────┐
   │         │
   ▼         ▼
Subject     Subject
Dependent   Independent
(TSFEL      (Bandpower +
 spectral)   Hemispheric
             Asymmetry)
   │         │
   ▼         ▼
Stratified  GroupKFold
 5-Fold CV  (by Subject)
   │         │
   └────┬────┘
        │
   ┌────▼────────────────────┐
   │  Inside Each Fold Only: │
   │  • StandardScaler fit   │
   │  • SMOTE oversample     │
   └────────────────────────┘
        │
        ▼
  6 ML Classifiers
  LogReg · LinearSVM · SVM-RBF
  GradBoost · XGBoost · RF
        │
        ▼
  Accuracy + Macro-F1
```

<details>
<summary><b>Leakage Prevention — How It Works</b></summary>

The key principle: **test data never touches any fitting step**.

| Step | Leakage-Safe Implementation |
|---|---|
| Feature scaling | `StandardScaler` fitted on `X_train` only, then applied to `X_test` |
| Oversampling | `SMOTE` applied only to training folds — test fold never receives synthetic samples |
| Hyperparameter tuning | `GridSearchCV` runs on training data only; best params evaluated once on held-out test |
| Cross-subject split | `GroupKFold` by subject ID ensures no trial overlap across folds |

</details>

---

## 📁 Project Structure

```
MSc-Dissertation/
│
├── app.py                          # Streamlit dashboard (all 7 pages)
├── requirements.txt                # Python dependencies
│
├── code/                           # Jupyter notebooks (run in order)
│   ├── 01 Raw Clean Comparison New.ipynb
│   ├── 02 Feature Extraction and Labelling for Subject Independent.ipynb
│   ├── 03 Subject Independent Training.ipynb
│   ├── 04 Feature Extraction and Labelling for Subject Dependent.ipynb
│   └── 05 Subject Dependent Training Model.ipynb
│
├── assets/                         # Pipeline and evaluation diagrams
│   ├── pipeline.png
│   ├── evaluation_flow.png
│   └── feature extraction and labelling.png
│
└── data/
    ├── features/                   # Extracted feature arrays (.npz) per subject
    │   └── labels/                 # Binary reward labels per subject
    └── metrics/                    # Evaluation outputs
        ├── subject_metrics.json    # Per-subject fold-level results
        ├── final_test_results.csv  # Subject-independent held-out test scores
        ├── tuning_results_train_only.csv
        └── table_A1_epochs_retention.csv
```

> **Note:** Raw DEAP EEG recordings are not included due to licensing. Obtain the dataset from [eecs.qmul.ac.uk/mmv/datasets/deap](https://www.eecs.qmul.ac.uk/mmv/datasets/deap/).

---

## 💻 Local Setup

### 1. Clone

```bash
git clone https://github.com/varadkamtikar/MSc-Dissertation.git
cd MSc-Dissertation
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the dashboard

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

### 5. Run the notebooks (optional)

Execute notebooks in `code/` **sequentially** (01 → 05). Raw DEAP data must be in place before running notebook 01.

---

## 🧪 Methodology

<details>
<summary><b>Subject-Dependent Evaluation</b></summary>

- **Split strategy:** Stratified 5-Fold Cross-Validation
- **Feature set:** TSFEL spectral features — rich time-frequency representation across all 32 channels
- **Purpose:** Measures how well a model learns an individual's EEG patterns
- **Expected performance:** Higher (~60–70%) due to within-subject consistency
- **Key constraint:** Scaler and SMOTE fitted inside each fold

</details>

<details>
<summary><b>Subject-Independent Evaluation</b></summary>

- **Split strategy:** GroupKFold (grouped by subject ID, k=5)
- **Feature set:** Bandpower (δ, θ, α, β, γ) + hemispheric asymmetry
- **Purpose:** Measures true population-level generalisation — entire subjects are held out
- **Expected performance:** Lower (~54%) — reflects real-world cross-subject challenge
- **Key constraint:** No subject appears in both training and test within any fold

</details>

<details>
<summary><b>Labelling Strategy</b></summary>

DEAP self-assessment Valence and Arousal ratings are averaged per trial.

```
Label = "Reward"     if (Valence + Arousal) / 2 > 5.5
Label = "No-Reward"  otherwise
```

Threshold 5.5 was chosen to produce a balanced binary split (640 Reward / 640 No-Reward across 32 subjects × 40 trials).

</details>

---

## 📦 Dataset

**DEAP** (Database for Emotion Analysis using Physiological Signals)

| Property | Value |
|---|---|
| Participants | 32 |
| EEG Channels | 32 |
| Trials / Participant | 40 |
| Sampling Rate | 256 Hz |
| Stimulus Duration | 60 s |
| Total Epochs | 1,280 |
| Labels Used | Valence + Arousal → Binary (Reward / No-Reward) |

> Dataset available at: [eecs.qmul.ac.uk/mmv/datasets/deap](https://www.eecs.qmul.ac.uk/mmv/datasets/deap/)

---

## 🛠 Tech Stack

| Layer | Tools |
|---|---|
| EEG Processing | MNE-Python, NumPy |
| Feature Engineering | TSFEL, NumPy, Pandas |
| Machine Learning | scikit-learn, XGBoost, imbalanced-learn |
| Dashboard | Streamlit, Plotly |
| Deployment | Streamlit Community Cloud |
| Language | Python 3.9 |

---

## 👤 Author

**Varadkrishna Kamtikar**
MSc Data Science · Technological University Dublin · 2024–25

---

## 📄 License

This repository is shared for academic and research purposes. Please cite appropriately if you build on this work.

---

<div align="center">

**[🚀 Open Live Dashboard](https://msc-dissertation-varadkamtikar.streamlit.app)**

</div>
