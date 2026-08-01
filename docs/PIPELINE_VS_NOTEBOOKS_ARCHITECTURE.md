# 🏛️ Architecture Breakdown: Production Pipeline vs. Interactive Research Notebooks

## 📌 Executive Summary

This document provides a comprehensive technical breakdown of the **Student Health Risk ML System's** dual-architecture design. In real-world enterprise Machine Learning and academic research (CIS6005), system development is split into two complementary environments:

1. **The Research Laboratory (Interactive Jupyter Notebooks)**
2. **The Production Factory (Automated Headless Pipeline)**

---

## ⚖️ 1. Comparative Analysis: Pipeline vs. Notebooks

| Metric / Dimension | 🔬 Research Laboratory (Notebooks) | 🏭 Production Factory (Pipeline) |
| :--- | :--- | :--- |
| **Primary Goal** | Feature discovery, exploratory analysis, model prototyping, and visual evidence generation. | Automated execution, reproducibility, headless execution, model serialization, and deployment. |
| **Execution Environment** | Interactive Cell-by-Cell execution (Jupyter / VS Code). | CLI-driven non-interactive execution via `Makefile` and `python` scripts. |
| **Primary Output** | High-resolution plots (`figures/`), distribution analysis, markdown explanations. | Serialized `.joblib` model binaries, split CSV datasets, MLflow tracking runs. |
| **State Management** | Stateful (variables retained in memory across cell runs). | Stateless per invocation (executed top-to-bottom clean process). |
| **Target Audience** | Data Scientists, Researchers, Academic Assessors (CIS6005 Report). | MLOps Engineers, Automated CI/CD Workflows, Batch Prediction Services. |
| **Error Handling** | Interactive debugging & immediate trial-and-error. | Exception logging, process exit codes, and automated MLflow metrics capture. |

---

## 🔬 2. Why Notebooks Were Built (`EDA/` & `Model Training/`)

While the production pipeline automates training and inference, **Jupyter Notebooks are indispensable** for the following critical reasons:

### A. Academic & Analytical Evidence (CIS6005 Requirement)
- For university assessment and scientific publication, claiming a model achieves high accuracy is insufficient; **visual and mathematical proof** is mandatory.
- Notebooks generate high-resolution figures (`.png` / `.jpg`) for:
  - Missingness topologies (`missingno` matrices).
  - Outlier detection (Seaborn Violin & Box plots).
  - Target separation proof (KDE distributions of engineered features like `lifestyle_risk_index`).
  - Manifold clustering visualizers (t-SNE 2D projections, UMAP plots, K-Means cluster bounds).

### B. Exploratory Feature Engineering & Hypothesis Testing
- Feature engineering requires rapid iteration. In a notebook, a scientist can test 20 different ratio transformations (e.g., `calories_per_bmi`) in seconds without waiting for a full pipeline retraining loop.
- Once a feature proves its statistical efficacy in a notebook, it is hardcoded into `data_pipeline.py`.

### C. Cell-by-Cell Interactive Debugging
- When models underperform or exhibit class bias, notebooks allow step-by-step inspection of intermediate matrices, OOF probability distributions, and gradient convergence.

---

## 🏭 3. What Happens Inside the Production Pipeline (`Pipeline/`)

The Production Pipeline is an automated, headless MLOps system that operates in 3 core stages:

```text
 ┌─────────────────────────────────────────────────────────────────────────────┐
 │                         STAGES OF THE PIPELINE                              │
 └─────────────────────────────────────────────────────────────────────────────┘
  
  [STAGE 1: DATA INGESTION & PROCESSING]
  `make data` ──> Pipeline/pipelines/data_pipeline.py
       ├── 1. Ingests Raw Data (train.csv, test.csv)
       ├── 2. Applies Feature Engineering (lifestyle_risk_index, sleep_distance_from_8)
       ├── 3. Fits & Serializes StandardScaler ──> artifacts/preprocessor/standard_scaler.joblib
       └── 4. Exports Clean Datasets ──> data/processed/train_processed.csv

  [STAGE 2: MULTI-PARADIGM MODEL TRAINING]
  `make train-[classification|regression|clustering]` ──> Pipeline/pipelines/training_pipeline.py
       ├── 1. Intercepts CLI piping input (classification / regression / clustering)
       ├── 2. Executes Paradigm Logic:
       │      ├── Classification: 5-Fold GBDT Triad (LGBM+XGB+Cat) + Scipy Nelder-Mead Optimization
       │      ├── Regression: XGBRegressor for BMI prediction
       │      └── Clustering: PCA + K-Means (K=3)
       ├── 3. Serializes Models ──> artifacts/models/[classification|regression|clustering]/
       └── 4. Logs Run Parameters, Metrics, & Confusion Matrices to MLflow Database (`mlruns/`)

  [STAGE 3: INFERENCE & SUBMISSION GENERATION]
  `make inference` ──> Pipeline/pipelines/inference_pipeline.py
       ├── 1. Loads Serialized Scalers & Model Binaries
       ├── 2. Runs Batch Predictions on Test Set
       ├── 3. Applies EV Signal Engine / Post-processing Adjustments
       └── 4. Generates Final Submission CSV ──> Pipeline/submission.csv
```

---

## 💡 Summary: How They Work Together

1. **Notebooks** act as the **Laboratory Sandbox**: Experiments are conducted, data hypotheses are verified, and figures are created for reports.
2. **Pipelines** act as the **Automated Factory**: Validated logic is converted into Python modules (`.py`), orchestrated via `Makefile`, tracked via `MLflow`, and executed automatically to produce production models and submission files.
