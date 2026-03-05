EEG-Reward-Detection

Leakage-Free EEG Classification Pipeline for Reward-Related Neural Pattern Detection

Overview

This repository contains the implementation and experimental pipeline developed for the MSc dissertation:

“AI-Based Detection of Reward-Related Neural Patterns from EEG Signals in Digital Interaction Contexts.”

The project focuses on designing and evaluating a leakage-free EEG classification pipeline that detects reward-related neural patterns using the DEAP dataset.
A key objective of this work is to ensure methodological correctness, reproducibility, and realistic performance evaluation, particularly in cross-subject EEG classification scenarios.

Unlike many EEG studies that report overly optimistic results, this project emphasises:

strict prevention of data leakage

evaluation under subject-independent conditions

transparent and reproducible experimentation.

Research Objectives

The main objectives of this dissertation were:

Design a fully automated end-to-end EEG processing pipeline.

Implement both subject-dependent and subject-independent evaluation frameworks.

Prevent data leakage during preprocessing, feature engineering, and model training.

Evaluate multiple classical machine learning models for EEG classification.

Provide realistic performance estimates for cross-subject EEG generalisation.

Dataset
DEAP Dataset

The experiments were conducted using the DEAP dataset, a widely used dataset in affective computing research.

Dataset characteristics:

32 participants

32 EEG channels

40 trials per participant

Emotional responses elicited through video stimuli

Self-assessment ratings for:

Valence

Arousal

Dominance

Liking

More information about the dataset is available here:

https://www.eecs.qmul.ac.uk/mmv/datasets/deap/

Important Note

The DEAP dataset is not included in this repository due to licensing restrictions.
Users must obtain the dataset from the original authors.

Repository Structure
EEG-Reward-Detection
│
├── notebooks/
│   ├── 01_preprocessing.ipynb
│   ├── 02_feature_extraction_SI.ipynb
│   ├── 03_training_SI.ipynb
│   ├── 04_feature_extraction_SD.ipynb
│   └── 05_training_SD.ipynb
│
├── results/
│   ├── metrics
│   └── evaluation outputs
│
├── requirements.txt
├── README.md
└── .gitignore
Pipeline Overview

The project implements a five-stage EEG processing pipeline.

1. EEG Preprocessing

Raw EEG recordings are cleaned and standardised before analysis.

Steps include:

Band-pass filtering

Notch filtering

Signal re-referencing

ICA-based artefact removal

Event detection and validation

Epoch extraction

Output:

Cleaned EEG epochs for each trial.

Notebook:

01_preprocessing.ipynb
2. Feature Extraction – Subject Independent

Feature vectors are constructed from cleaned EEG epochs for cross-subject evaluation.

Features include:

Bandpower features across frequency bands

Hemispheric asymmetry features

A combined dataset is then created across all subjects.

Notebook:

02_feature_extraction_SI.ipynb
3. Subject-Independent Model Training

Models are trained using GroupKFold cross-validation to ensure subject separation.

Key characteristics:

Training and testing subjects are never mixed

Feature scaling performed inside training folds

SMOTE applied only on training data

Evaluation metrics include accuracy and Macro-F1 score

Notebook:

03_training_SI.ipynb
4. Feature Extraction – Subject Dependent

A second feature engineering approach is implemented for within-subject analysis.

Features:

TSFEL-based spectral features

Channel-wise feature extraction

Each subject is processed independently.

Notebook:

04_feature_extraction_SD.ipynb
5. Subject-Dependent Model Training

Models are trained separately for each subject using stratified cross-validation.

This allows comparison between:

within-subject performance

cross-subject generalisation

Notebook:

05_training_SD.ipynb
Evaluation Strategy

Two evaluation paradigms are used:

Subject-Dependent Evaluation

Training and testing performed on the same subject

Stratified cross-validation ensures class balance

Subject-Independent Evaluation

Training on multiple subjects

Testing on completely unseen subjects

Implemented using GroupKFold cross-validation

This approach provides a more realistic measure of model generalisation.

Results Summary

Key findings from the experiments:

Subject-dependent models achieve higher performance than subject-independent models.

Cross-subject EEG generalisation remains challenging.

Simpler models such as Linear Support Vector Machines perform well under strict evaluation protocols.

Typical subject-independent performance:

Accuracy ≈ 0.54

Macro-F1 ≈ 0.53

These results reflect realistic performance levels when data leakage is prevented.

Key Contributions

This dissertation makes the following contributions:

Development of a fully automated EEG processing pipeline

Implementation of leakage-free evaluation protocols

Comparison of subject-dependent vs subject-independent EEG classification

Transparent reporting of realistic cross-subject performance

Installation

Clone the repository:

git clone https://github.com/<username>/EEG-Reward-Detection.git
cd EEG-Reward-Detection

Install required dependencies:

pip install -r requirements.txt
Running the Pipeline

The notebooks should be executed sequentially:

01_preprocessing.ipynb

02_feature_extraction_SI.ipynb

03_training_SI.ipynb

04_feature_extraction_SD.ipynb

05_training_SD.ipynb

Technologies Used

Python

MNE-Python

NumPy

Pandas

Scikit-learn

TSFEL

Imbalanced-Learn

Matplotlib

Limitations

Several limitations should be acknowledged:

Experiments rely on a single dataset.

Handcrafted features are used instead of deep learning representations.

Cross-subject EEG variability limits classification performance.

Future work may explore:

connectivity-based EEG features

subject-adaptive models

transfer learning approaches.

Author

Varadkrishna Kamtikar
MSc Data Science
Technological University Dublin

License

This repository is intended for academic and research purposes.

If you want, I can also help you add three things that make a dissertation repo look extremely professional:

• a pipeline diagram in the README
• reproducibility instructions for examiners
• GitHub badges (Python version, license, etc.)

These significantly improve how your project looks.