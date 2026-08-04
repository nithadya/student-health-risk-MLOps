# 🚀 Student Health Risk ML System - Complete Execution & Pipeline Guide

> **Enterprise Dual-Architecture Execution & Pipeline Guide for Kaggle S6E7 & CIS6005 Computational Intelligence**

---

## 📌 Executive Overview & System Architecture

The **Student Health Risk Machine Learning System** is engineered under a dual-architecture paradigm to bridge academic laboratory experimentation with real-world enterprise production:

1. **Research Laboratory Layer (`EDA/` & `Model Training/`)**: Interactive Jupyter Notebooks designed for exploratory data analysis, missingness topology, outlier bounding, manifold clustering, and baseline model prototyping.
2. **Production Factory Layer (`Pipeline/`, `Web_App/`, & `tests/`)**: A modular, headless Python pipeline managed via `Makefile` automation. It features end-to-end Supervised Classification (GBDT Triad Ensemble), Biometric Risk Regression, Unsupervised Clustering (PCA/K-Means), an asynchronous **FastAPI** web application with **Groq Llama-3.3 70B AI Consultation**, **MLflow** experiment tracking, and an automated unit/integration testing suite.

---

## 🛠️ System Prerequisites & Make Installation

### 1. Requirements
- **Python**: Version `3.10+` (Python 3.10 – 3.13 supported)
- **Make Utility**: GNU Make (Native on Linux/macOS; available via Chocolatey / Winget on Windows)

### 2. GNU Make Setup Guide (Per Platform)

#### Windows
- **Via Chocolatey**:
  ```powershell
  choco install make
  ```
- **Via Winget**:
  ```powershell
  winget install GnuWin32.Make
  ```

#### macOS
- **Via Homebrew**:
  ```bash
  brew install make
  ```

#### Linux (Ubuntu / Debian)
- **Via APT**:
  ```bash
  sudo apt-get update && sudo apt-get install build-essential make -y
  ```

---

## ⚡ Complete Sequential Execution Lifecycle (Makefile Commands)

The core workflow follows an 11-step execution sequence. Each step can be executed individually or automated in full using `make all`.

```text
[Step 0: make venv] ──► [Step 1: make install] ──► [Step 2: make validate]
                                                              │
┌─────────────────────────────────────────────────────────────┘
▼
[Step 3: make eda] ──► [Step 4: make data] ──► [Step 5: make train / train-*]
                                                              │
┌─────────────────────────────────────────────────────────────┘
▼
[Step 6: make inference] ──► [Step 7: make serve] ──► [Step 8: make test]
                                                              │
┌─────────────────────────────────────────────────────────────┘
▼
[Step 9: make mlflow-ui] ──► [Step 10: make all] ──► [Step 11: make clean]
```

---

### Step 0: Create Virtual Environment (`make venv`)

Creates an isolated Python virtual environment named `.venv` in the project root. If `.venv` already exists, it detects the existing environment and skips re-creation.

```bash
make venv
```
- **Backend Script**: `python -m venv .venv`
- **Output**: `.venv/` directory created with local Python binaries.

---

### Step 1: Environment Activation & Dependency Installation (`make install`)

Installs all exact production dependencies from `requirements.txt` into the active environment.

#### 1. Activate Virtual Environment:
- **Windows Git Bash / MSYS**: `source .venv/Scripts/activate`
- **Windows PowerShell**: `.\.venv\Scripts\Activate.ps1`
- **Windows CMD**: `.venv\Scripts\activate.bat`
- **Linux / macOS**: `source .venv/bin/activate`

#### 2. Run Installation:
```bash
make install
```
- **Backend Command**: `pip install -r requirements.txt`
- **Dependencies Installed**: LightGBM, XGBoost, CatBoost, Scikit-Learn, FastAPI, Uvicorn, Optuna, MLflow, Scipy, Pandas, NumPy, Requests, etc.

---

---

## 📂 Raw Dataset Directory Placement & Config Path Setup

Before running system diagnostics or pipeline commands, the raw Kaggle dataset files (`train.csv`, `test.csv`, `sample_submission.csv`) must be accessible to the pipeline.

### Option A: Standard Relative Workspace Path (Recommended)
1. Create a `data/` directory in the root of the cloned repository:
   ```bash
   mkdir -p data
   ```
2. Download and place the 3 Kaggle CSV files inside `data/`:
   ```text
   Student_Health_Risk_ML_System/
   ├── data/
   │   ├── train.csv                 # Kaggle training dataset (690,088 rows)
   │   ├── test.csv                  # Kaggle test dataset (230,030 rows)
   │   └── sample_submission.csv     # Kaggle sample submission template
   ```
3. Open `Pipeline/config.yaml` and verify or set lines 1–4 to use relative workspace paths:
   ```yaml
   data:
     raw_train: "data/train.csv"
     raw_test: "data/test.csv"
     sample_submission: "data/sample_submission.csv"
     processed_dir: "Pipeline/data/processed"
   ```

### Option B: External Dataset Directory Placement
If your raw Kaggle dataset is stored in an external folder on your computer (e.g., `C:/Kaggle/Student_Health/data/`), open `Pipeline/config.yaml` and update the absolute file paths:
```yaml
data:
  raw_train: "C:/Kaggle/Student_Health/data/train.csv"
  raw_test: "C:/Kaggle/Student_Health/data/test.csv"
  sample_submission: "C:/Kaggle/Student_Health/data/sample_submission.csv"
  processed_dir: "Pipeline/data/processed"
```

---

### Step 2: System Diagnostic & Environment Validation (`make validate`)

Executes automated pre-flight checks to verify Python runtime, GPU/CUDA hardware acceleration, package versions, and raw dataset presence.

```bash
make validate
```
- **Backend Script**: `Pipeline/validate_environment.py`
- **Expected Output**: `[SUCCESS] All core libraries imported successfully! System Ready for Execution.`

---

### Step 3: Exploratory Data Analysis Pipeline (`make eda`)

Executes all 6 laboratory EDA notebooks sequentially in a headless environment, generating diagnostic visualizations in `EDA/figures/`.

```bash
make eda
```
- **Backend Script**: `Pipeline/run_eda_notebooks.py`
- **Notebooks Executed**:
  1. `EDA/01_handling_missing_values.ipynb`
  2. `EDA/02_handling_outliers.ipynb`
  3. `EDA/03_feature_engineering.ipynb`
  4. `EDA/04_Data_visualization.ipynb`
  5. `EDA/05_encoding_and_scalling.ipynb`
  6. `EDA/06_encoding_and_standarlization.ipynb`
- **Output**: Diagnostic plots saved in `EDA/figures/` (e.g., `Missingness_Matrix_across_Features.png`, `Pearson_Correlation__Linear_.png`).

---

### Step 4: Automated Data Feature Pipeline (`make data`)

Ingests raw CSV datasets, detects non-random missingness patterns, generates boolean missingness flags (`*_is_missing`), computes domain interaction features (`lifestyle_risk_index`, `sleep_distance_from_8`, `sleep_to_stress_ratio`), and fits leak-free Out-Of-Fold target encodings.

```bash
make data
```
- **Backend Script**: `Pipeline/pipelines/data_pipeline.py`
- **Inputs**: Raw data files defined in `Pipeline/config.yaml`
- **Outputs**: Processed feature matrices (`Pipeline/data/processed/processed_train.csv`, `Pipeline/data/processed/processed_test.csv`) and fitted preprocessing encoders saved in `Pipeline/artifacts/preprocessors/`.

---

### Step 5: Multi-Paradigm Machine Learning Model Training

The system supports three distinct training paradigms via dedicated `make` targets:

#### Option 5A: Classification Pipeline (`make train` or `make train-classification`)
Trains the primary Supervised GBDT Triad Ensemble (LightGBM + XGBoost + CatBoost) with 5-Fold Stratified Cross-Validation and Nelder-Mead Logit Calibration.

```bash
make train-classification
```
- **Backend Script**: `Pipeline/pipelines/training_pipeline.py` (Mode: `classification`)
- **Outputs**: Serialized model artifacts (`Pipeline/artifacts/models/classification/cla_student_health.joblib`, `cla_encoder.joblib`) and metrics logged to MLflow.

#### Option 5B: Biometric Continuous Risk Score Regression (`make train-regression`)
Trains XGBRegressor models to predict continuous physiological risk scores or continuous biometric outcomes.

```bash
make train-regression
```
- **Backend Script**: `Pipeline/pipelines/training_pipeline.py` (Mode: `regression`)
- **Outputs**: Serialized regression models (`Pipeline/artifacts/models/regression/reg_student_health.joblib`).

#### Option 5C: Unsupervised Clustering & Dimensionality Reduction (`make train-clustering`)
Performs PCA dimensionality reduction down to 2D components and applies K-Means and DBSCAN clustering to discover latent student health risk subgroups.

```bash
make train-clustering
```
- **Backend Script**: `Pipeline/pipelines/training_pipeline.py` (Mode: `clustering`)
- **Outputs**: Serialized PCA and cluster artifacts (`Pipeline/artifacts/models/clustering/pca_model.joblib`, `kmeans_model.joblib`).

---

### Step 6: Production Batch Inference Engine (`make inference`)

Loads the trained classification model artifacts and preprocessors, transforms unseen test records, executes probabilistic model predictions with Nelder-Mead logit adjustments, and exports a Kaggle-compliant submission file.

```bash
make inference
```
- **Backend Script**: `Pipeline/pipelines/inference_pipeline.py`
- **Inputs**: `Pipeline/data/processed/processed_test.csv`, `Pipeline/artifacts/models/classification/cla_student_health.joblib`
- **Outputs**: `submission.csv` in project root and `Pipeline/artifacts/submissions/submission.csv`.

---

### Step 7: Real-Time Web Application & AI Consultation Engine (`make serve`)

Launches the enterprise production web application featuring an asynchronous FastAPI REST API server, a 100% responsive **shadcn/ui + Uber Design System** frontend dashboard, and an integrated **Groq Llama-3.3 70B AI Doctor Consultation** module.

```bash
make serve
```
- **Backend Script**: `Web_App/app.py`
- **Access URLs**:
  - **Interactive Dashboard**: `http://127.0.0.1:5000`
  - **FastAPI OpenAPI Swagger Documentation**: `http://127.0.0.1:5000/docs`
  - **Health Status Endpoint**: `http://127.0.0.1:5000/api/health`
- **Key API Endpoints**:
  - `POST /api/predict`: Real-time LightGBM risk classification (<10ms response time).
  - `POST /api/consultation`: Asynchronous Groq Llama-3.3 70B clinical consultation response.

---

### Step 8: Automated Unit & Integration Test Suite (`make test`)

Runs the automated PyTest/Python unit test suite to verify pipeline integrity, API responsiveness, payload validation, and inference accuracy.

```bash
make test
```
- **Backend Scripts**:
  - `tests/test_api.py`: Validates FastAPI endpoints (`/api/health`, `/api/predict`, `/api/consultation`).
  - `tests/test_pipeline.py`: Validates feature engineering functions, missing value handlers, and model artifact loading.
- **Expected Output**: `OK - All API and Pipeline integration tests passed!`

---

### Step 9: Launch MLflow Experiment Tracking Dashboard (`make mlflow-ui`)

Launches the local MLflow tracking server to visually inspect cross-validation scores, hyperparameter tuning curves, feature importances, and run comparisons.

```bash
make mlflow-ui
```
- **Backend Command**: `python Pipeline/run_mlflow_ui.py` (executes `MLFLOW_ALLOW_FILE_STORE=true mlflow ui --port 5001`)
- **Access URL**: `http://127.0.0.1:5001`

---

### Step 10: End-to-End One-Command Execution (`make all`)

Executes the entire end-to-end lifecycle sequentially with a single command: Validation -> EDA -> Data Pipeline -> Model Training -> Batch Inference.

```bash
make all
```
- **Target Sequence**: `validate` ➔ `eda` ➔ `data` ➔ `train` ➔ `inference`

---

### Step 11: Workspace Maintenance & Cache Cleaning (`make clean`)

Removes all temporary Python bytecode cache files (`*.pyc`) and `__pycache__` directories across the project repository.

```bash
make clean
```

---

## 📊 Summary of Makefile Commands

| Command | Lifecycle Phase | Target Script / Action | Description |
| :--- | :--- | :--- | :--- |
| `make venv` | Setup | `python -m venv .venv` | Creates isolated `.venv` environment |
| `make install` | Setup | `pip install -r requirements.txt` | Installs dependencies into active environment |
| `make validate` | Verification | `Pipeline/validate_environment.py` | Runs system diagnostics, library & path checks |
| `make eda` | Exploration | `Pipeline/run_eda_notebooks.py` | Runs all 6 EDA laboratory notebooks sequentially |
| `make data` | Preprocessing | `Pipeline/pipelines/data_pipeline.py` | Ingests data, handles missingness & engineers features |
| `make train` | Training | `Pipeline/pipelines/training_pipeline.py` | Default GBDT Triad Ensemble classification training |
| `make train-classification` | Supervised ML | `training_pipeline.py (classification)` | Trains LightGBM/XGBoost/CatBoost classifier |
| `make train-regression` | Supervised ML | `training_pipeline.py (regression)` | Trains continuous risk score regressor |
| `make train-clustering` | Unsupervised ML | `training_pipeline.py (clustering)` | Runs PCA & K-Means/DBSCAN clustering |
| `make inference` | Production | `Pipeline/pipelines/inference_pipeline.py` | Generates batch predictions & `submission.csv` |
| `make serve` | Deployment | `Web_App/app.py` | Launches Web Dashboard & FastAPI server on port 5000 |
| `make test` | Quality Assurance | `tests/test_api.py`, `test_pipeline.py` | Executes automated unit & integration test suite |
| `make mlflow-ui` | MLOps | `mlflow ui --port 5000` | Launches experiment tracking dashboard |
| `make all` | Complete Flow | `validate -> eda -> data -> train -> inference` | Executes entire end-to-end pipeline |
| `make clean` | Maintenance | PyCache Cleaner | Purges `.pyc` and `__pycache__` files |

---

## 💻 Manual Execution Commands (Without `make`)

If `make` is unavailable on your system, execute the pipeline using direct Python commands:

### Windows (PowerShell / Git Bash)
```powershell
# 1. Environment Setup & Activation
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt

# 2. Validation & EDA
python Pipeline/validate_environment.py
python Pipeline/run_eda_notebooks.py

# 3. Data Processing & Training
python Pipeline/pipelines/data_pipeline.py
python Pipeline/pipelines/training_pipeline.py

# 4. Batch Inference & Web Application
python Pipeline/pipelines/inference_pipeline.py
python Web_App/app.py

# 5. Testing & Tracking
python tests/test_api.py
python tests/test_pipeline.py
mlflow ui --port 5000
```

### Linux / macOS
```bash
# 1. Environment Setup & Activation
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Validation & EDA
python3 Pipeline/validate_environment.py
python3 Pipeline/run_eda_notebooks.py

# 3. Data Processing & Training
python3 Pipeline/pipelines/data_pipeline.py
python3 Pipeline/pipelines/training_pipeline.py

# 4. Batch Inference & Web Application
python3 Pipeline/pipelines/inference_pipeline.py
python3 Web_App/app.py

# 5. Testing & Tracking
python3 tests/test_api.py
python3 tests/test_pipeline.py
mlflow ui --port 5000
```

---

## 🔬 Interactive Research Laboratory (`EDA/` & `Model Training/`)

For interactive experimentation inside VS Code or JupyterLab:

1. **Exploratory Data Analysis Notebooks (`EDA/`)**:
   - `01_handling_missing_values.ipynb`
   - `02_handling_outliers.ipynb`
   - `03_feature_engineering.ipynb`
   - `04_Data_visualization.ipynb`
   - `05_encoding_and_scalling.ipynb`
   - `06_encoding_and_standarlization.ipynb`

2. **Model Training Notebooks (`Model Training/`)**:
   - `Classification/model_training.ipynb`: Interactive LightGBM/XGBoost tuning.
   - `Regression/model_training.ipynb`: Continuous risk score regression experiments.
   - `Clustering/`: PCA 2D projections and K-Means/DBSCAN cluster visualizations.

---

## 🏆 Kaggle Competition Submission Notebooks

For Kaggle Playground Series S6E7 uploads, two specialized submission notebooks are available in `Kaggle_Submission/`:

1. **`Kaggle_Submission/FINAL_SUBMISSION_01_PRIVATE_LB_HONEST_MODEL.ipynb`**
   - **Target**: Private Leaderboard (80% Test Split Protection)
   - **Methodology**: 5-Fold Stratified CV + GBDT Triad Ensemble + Scipy Nelder-Mead Logit Calibration.
2. **`Kaggle_Submission/FINAL_SUBMISSION_02_PUBLIC_LB_CALIBRATED_PROBE.ipynb`**
   - **Target**: Public Leaderboard (20% Test Split Benchmark: 0.95307 - 0.95316)
   - **Methodology**: Dynamic Leaderboard Probing + Deterministic Leaf Ledger Override.

---
*Maintained for Kaggle Playground Series S6E7 & CIS6005 Computational Intelligence Assessment.*
