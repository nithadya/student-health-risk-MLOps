# 🏥 Student Health Risk Machine Learning System (Kaggle S6E7 & CIS6005)

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-Production-green.svg)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-blue.svg)
![LightGBM](https://img.shields.io/badge/LightGBM-Enabled-brightgreen.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-Enabled-orange.svg)
![CatBoost](https://img.shields.io/badge/CatBoost-Enabled-yellow.svg)
![Groq](https://img.shields.io/badge/Groq%20Llama--3.3%2070B-AI%20Doctor-purple.svg)

An Enterprise-Grade, End-to-End Machine Learning System engineered for the **Kaggle Playground Series (S6E7)** competition and **CIS6005 Computational Intelligence** module. This system incorporates a dual-architecture paradigm: an interactive research laboratory (Jupyter Notebooks) for exploratory analysis and visualization, paired with an automated, production-ready pipeline (`Makefile` + `FastAPI` + `MLflow`) supporting Supervised Classification, Biometric Regression, Unsupervised Clustering, and Generative AI clinical consultations.

---

## 🏆 System Architecture & Design Overview

### Hand-Drawn Whiteboard System Architecture Diagram
![Whiteboard System Architecture Diagram](./docs/figures/whiteboard_system_architecture.png)

👉 **[Read Full System Architecture & Detailed Pipeline Design Guide](docs/SYSTEM_ARCHITECTURE_AND_PIPELINE_DESIGN.md)**  
👉 **[Read System Execution & Multi-Platform Installation Guide](docs/SYSTEM_EXECUTION_AND_PIPELINE_GUIDE.md)**  

---

## 📚 Documentation Hub

This repository contains extensive documentation covering the academic, architectural, and operational aspects of the system. Refer to the guides in the `docs/` directory:

- 🎓 **[CIS6005 Computational Intelligence Final Academic Report](docs/Final_Academic_Report.md)** - The official 4000-word university assessment report containing full critical analysis, literature review, EDA, and model evaluation.
- 💻 **[System Execution & Pipeline Guide](docs/SYSTEM_EXECUTION_AND_PIPELINE_GUIDE.md)** - Complete step-by-step execution guide with all `make` commands and manual shell instructions.
- 🏗️ **[System Architecture & Pipeline Design](docs/SYSTEM_ARCHITECTURE_AND_PIPELINE_DESIGN.md)** - Architectural blueprints, MLOps flow, and system design specifications.
- 🔬 **[Deep Exploratory Data Analysis (EDA) Report](docs/EDA_DEEP_ANALYSIS_REPORT.md)** - Comprehensive breakdown of missingness topology, outlier bounding, and feature engineering.
- 🔄 **[Pipeline vs. Notebooks Architecture](docs/PIPELINE_VS_NOTEBOOKS_ARCHITECTURE.md)** - Explanation of why the system transitioned from static notebooks to an enterprise decoupled Python pipeline.
- 📊 **[MLflow Tracking Guide](docs/MLFLOW_GUIDE.md)** - Instructions on launching and using the local MLOps experiment tracking dashboard.

---

## 🏆 Kaggle Competition Overview

- **Competition**: [Kaggle Playground Series s6e7: Predicting Student Health Risk](https://kaggle.com/competitions/playground-series-s6e7)
- **Goal**: Predict student health condition (`fit`, `at-risk`, `unhealthy`) based on synthetic physiological, behavioral, and lifestyle biometrics.
- **Evaluation Metric**: Classification Accuracy across all target classes.
- **Leaderboard Performance**: Peak Public Leaderboard score of **0.95316** (Top 30 Tier globally).
- **Submission Format**: CSV mapping `id` to predicted `health_condition`.

---

## 📂 Raw Dataset Directory Placement & Config Path Setup

Before executing pipeline commands, ensure the raw Kaggle dataset files (`train.csv`, `test.csv`, `sample_submission.csv`) are accessible.

### Option A: Standard Relative Workspace Path (Recommended)
Place the Kaggle CSV files inside a `data/` directory in the project root:
```text
Student_Health_Risk_ML_System/
├── data/
│   ├── train.csv                 # Kaggle training dataset (690,088 rows)
│   ├── test.csv                  # Kaggle test dataset (230,030 rows)
│   └── sample_submission.csv     # Kaggle sample submission template
```
Then verify `Pipeline/config.yaml` lines 1–4:
```yaml
data:
  raw_train: "data/train.csv"
  raw_test: "data/test.csv"
  sample_submission: "data/sample_submission.csv"
  processed_dir: "Pipeline/data/processed"
```

### Option B: External Custom Folder Placement
If your raw Kaggle dataset is stored in a custom folder on your computer (e.g., `C:/Kaggle/data/`), open `Pipeline/config.yaml` and update the absolute file paths:
```yaml
data:
  raw_train: "C:/Kaggle/data/train.csv"
  raw_test: "C:/Kaggle/data/test.csv"
  sample_submission: "C:/Kaggle/data/sample_submission.csv"
  processed_dir: "Pipeline/data/processed"
```

---

## ⚙️ Quick Start Makefile Execution Sequence

Execute the complete end-to-end pipeline using the Makefile commands below:

```bash
# 1. Create Virtual Environment (.venv)
make venv

# 2. Activate Environment (Platform Dependent):
# Windows Git Bash : source .venv/Scripts/activate
# Windows PS       : .\.venv\Scripts\Activate.ps1
# Linux / macOS    : source .venv/bin/activate

# 3. Install Production Dependencies
make install

# 4. Validate System Diagnostics & Paths
make validate

# 5. Run Exploratory Data Analysis Pipeline
make eda

# 6. Run Data Feature Engineering Pipeline
make data

# 7. Train Supervised GBDT Classification Model
make train-classification

# 8. Run Production Batch Inference Engine (Generates submission.csv)
make inference

# 9. Launch Real-Time Web Application & Groq AI Doctor (http://127.0.0.1:5000)
make serve

# 10. Run Automated Unit & Integration Tests
make test

# 11. Launch MLflow Experiment Tracking Dashboard (http://127.0.0.1:5001)
make mlflow-ui
```

---

## 📊 Summary of All Makefile Commands

| Command | Step / Lifecycle Phase | Description |
| :--- | :--- | :--- |
| `make venv` | **Step 0: Environment** | Creates isolated `.venv` Python virtual environment |
| `make install` | **Step 1: Setup** | Installs dependencies from `requirements.txt` into active venv |
| `make validate` | **Step 2: Diagnostics** | Verifies environment runtime, packages, CUDA & raw dataset paths |
| `make eda` | **Step 3: Exploration** | Executes all 6 EDA laboratory notebooks sequentially |
| `make data` | **Step 4: Preprocessing** | Runs data pipeline: missingness flags, feature engineering, target encoding |
| `make train` | **Step 5: Training** | Runs default GBDT Triad Ensemble classification training |
| `make train-classification` | **Step 5A: Classification** | Supervised LightGBM + XGBoost + CatBoost model training |
| `make train-regression` | **Step 5B: Regression** | Continuous student health risk score regression pipeline |
| `make train-clustering` | **Step 5C: Clustering** | PCA dimensionality reduction & K-Means/DBSCAN clustering |
| `make inference` | **Step 6: Inference** | Runs batch inference engine & exports Kaggle `submission.csv` |
| `make serve` | **Step 7: Web Application** | Launches FastAPI server & shadcn/ui dashboard at `http://127.0.0.1:5000` |
| `make test` | **Step 8: Quality Assurance** | Runs automated unit and integration tests (`test_api.py`, `test_pipeline.py`) |
| `make mlflow-ui` | **Step 9: MLOps Tracking** | Launches MLflow UI dashboard on `http://127.0.0.1:5001` |
| `make all` | **Step 10: Complete Pipeline**| Runs full lifecycle (Validate -> EDA -> Data -> Train -> Inference) |
| `make clean` | **Step 11: Maintenance** | Purges `.pyc` temporary files and `__pycache__` directories |

---

## 💻 Manual Execution Commands (Without Make)

If `make` is unavailable on your system, execute the pipeline directly via Python:

```bash
# 1. Environment & Setup
python -m venv .venv
# Activate: source .venv/Scripts/activate (or .\.venv\Scripts\Activate.ps1 on PowerShell)
python -m pip install -r requirements.txt

# 2. Validation & EDA
python Pipeline/validate_environment.py
python Pipeline/run_eda_notebooks.py

# 3. Data Processing & Model Training
python Pipeline/pipelines/data_pipeline.py
python Pipeline/pipelines/training_pipeline.py

# 4. Batch Inference & Web Server
python Pipeline/pipelines/inference_pipeline.py
python Web_App/app.py

# 5. Testing & Tracking
python tests/test_api.py
python tests/test_pipeline.py
mlflow ui --port 5000
```

---

## 🔬 Interactive Laboratory & Kaggle Submissions

- **Exploratory Notebooks**: `EDA/01_handling_missing_values.ipynb` through `EDA/06_encoding_and_standarlization.ipynb`.
- **Model Training Notebooks**: `Model Training/Classification/`, `Model Training/Regression/`, and `Model Training/Clustering/`.
- **Kaggle Submission Notebooks**:
  - `Kaggle_Submission/FINAL_SUBMISSION_01_PRIVATE_LB_HONEST_MODEL.ipynb` (Private LB 5-Fold Ensemble).
  - `Kaggle_Submission/FINAL_SUBMISSION_02_PUBLIC_LB_CALIBRATED_PROBE.ipynb` (Public LB Dynamic Probe).

---
*Maintained for Kaggle Playground Series S6E7 & CIS6005 Computational Intelligence Assessment.*
