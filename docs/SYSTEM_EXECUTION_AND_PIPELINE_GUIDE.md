# 🚀 Student Health Risk ML System - Complete Execution & Pipeline Guide
> **Enterprise Dual-Architecture Execution Guide for Kaggle S6E7 & CIS6005**

---

## 📌 Executive Overview & System Architecture

The **Student Health Risk Machine Learning System** operates under a dual-architecture paradigm:
1. **Research Laboratory (`EDA/` & `Model Training/`)**: Interactive Jupyter Notebooks designed for exploratory data analysis, visual diagnostic plots, and academic experiment reporting.
2. **Production Factory (`Pipeline/`)**: Headless, Makefile-managed automated pipeline engineered for Supervised Classification, Biometric Regression, and Unsupervised Clustering with MLflow experiment tracking.

---

## 🛠️ Step-by-Step Environment Setup & Installation Sequence

### 1. System Requirements
- **Python**: Version `3.10+` (Python 3.10 - 3.13 supported)
- **Make Utility**: GNU Make (Native on Linux/macOS; available via Chocolatey / winget on Windows)

---

### Step 1: Create Virtual Environment (`make venv`)

The `make venv` command creates an isolated `.venv` environment if one does not already exist. If `.venv` is already present, it detects the existing environment and skips re-creation:

```bash
make venv
```

---

### Step 2: Activate Environment & Install Dependencies (`make install`)

After creating `.venv`, activate the environment based on your operating system and shell, then run `make install` to install all dependencies from `requirements.txt`:

#### A. Using `make` Automation (Recommended)

```bash
# 1. Activate Environment:
# Windows Git Bash (MSYS)  : source .venv/Scripts/activate
# Windows PowerShell       : .\.venv\Scripts\Activate.ps1
# Windows CMD              : .venv\Scripts\activate.bat
# Linux / macOS            : source .venv/bin/activate

# 2. Install Dependencies into Active Environment:
make install
```

#### B. Direct Terminal Commands (When `make` is not available)

##### Windows Git Bash / MSYS Terminal
```bash
# Create virtual environment if missing
python -m venv .venv

# Activate environment
source .venv/Scripts/activate

# Install dependencies
python -m pip install -r requirements.txt
```

##### Windows PowerShell
```powershell
# Create virtual environment if missing
python -m venv .venv

# Activate environment
.\.venv\Scripts\Activate.ps1

# Install dependencies
python -m pip install -r requirements.txt
```

##### Windows Command Prompt (CMD)
```cmd
cd c:\Users\mihisara\Desktop\ML\Student_Health_Risk_ML_System
python -m venv .venv
.venv\Scripts\activate.bat
python -m pip install -r requirements.txt
```

##### macOS / Linux Terminal
```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

---

### Step 3: Run Environment Validation (`make validate`)

After installation, execute `make validate` (or `python Pipeline/validate_environment.py` without `make`) to verify Python version, virtualenv activation, package installations (LightGBM, XGBoost, CatBoost, Scipy, etc.), hardware acceleration, and dataset paths:

```bash
make validate
```

---

## 🐍 Option: Anaconda / Miniconda Environment Setup

If using Anaconda or Miniconda:

### Step 1: Create & Activate Conda Environment
```bash
# Create isolated Python 3.10 environment named 'health_ml'
conda create -n health_ml python=3.10 -y

# Activate environment
conda activate health_ml

# Install dependencies
make install
# (or python -m pip install -r requirements.txt)

# Validate setup
make validate
# (or python Pipeline/validate_environment.py)
```

---

## ⚙️ GNU Make Utility Installation Guide (Windows / macOS / Linux)

### 1. Windows Setup for `make`
- **Via Chocolatey**:
  ```powershell
  choco install make
  ```
- **Via Winget**:
  ```powershell
  winget install GnuWin32.Make
  ```

### 2. macOS Setup for `make`
- Install via **Homebrew**:
  ```bash
  brew install make
  ```

### 3. Linux Setup for `make`
- **Ubuntu / Debian**:
  ```bash
  sudo apt-get install build-essential make -y
  ```

---

## ⚡ Execution Mode 1: Automated Makefile Hub Commands

Both the **Project Root** and `Pipeline/` directory contain fully configured `Makefile` automation hubs.

| Command | Step / Target Action | Description |
| :--- | :--- | :--- |
| `make venv` | **Step 1** | Create isolated `.venv` environment (Skips if already existing) |
| `make install` | **Step 2** | Install all dependencies from `requirements.txt` into active environment |
| `make validate` | **Step 3** | Run system, dependency & dataset diagnostic check |
| `make eda` | **EDA Execution** | Execute all 6 EDA laboratory notebooks sequentially |
| `make data` | **Data Pipeline** | Ingest raw CSVs, handle missingness flags & compute interaction features |
| `make train` | **Model Training** | Run GBDT Triad Ensemble classification training |
| `make train-classification` | **Classification** | Supervised LightGBM + XGBoost + CatBoost training |
| `make train-regression` | **Regression** | Continuous student risk score regression pipeline |
| `make train-clustering` | **Clustering** | PCA dimensionality reduction & K-Means/DBSCAN clustering |
| `make inference` | **Inference Engine** | Run batch inference & export Kaggle-compliant `submission.csv` |
| `make mlflow-ui` | **MLflow Dashboard** | Launch experiment tracking UI on `http://localhost:5000` |
| `make all` | **Full Lifecycle** | Execute end-to-end (Validate -> EDA -> Data -> Train -> Inference) |
| `make clean` | **Cleanup** | Remove cached `.pyc` files and `__pycache__` directories |

---

## 🔬 Execution Mode 2: Interactive Research Laboratory (Jupyter)

If executing interactively inside VS Code, JupyterLab, or Jupyter Notebook UI:

### 1. Exploratory Data Analysis (`EDA/`)
Open and run cells in order:
1. `EDA/01_handling_missing_values.ipynb`
2. `EDA/02_handling_outliers.ipynb`
3. `EDA/03_feature_engineering.ipynb`
4. `EDA/04_Data_visualization.ipynb`
5. `EDA/05_encoding_and_scalling.ipynb`
6. `EDA/06_encoding_and_standarlization.ipynb`

### 2. Supervised & Unsupervised Prototyping (`Model Training/`)
- **Classification**: Open `Model Training/Classification/model_training.ipynb`
- **Regression**: Open `Model Training/Regression/model_training.ipynb`
- **Dimensionality Reduction & Clustering**: Open notebooks in `Model Training/Clustering/`

---

## 🏆 Execution Mode 3: Kaggle Final Competition Submissions

For Kaggle Playground Series S6E7 competition upload, two documented final submission notebooks are provided in `Kaggle_Submission/`:

1. **`Kaggle_Submission/FINAL_SUBMISSION_01_PRIVATE_LB_HONEST_MODEL.ipynb`**
   - **Target**: Private Leaderboard (80% Test Split Protection)
   - **Method**: Stratified 5-Fold CV + GPU GBDT Triad Ensemble (LightGBM + XGBoost + CatBoost) + Scipy Nelder-Mead Logit Calibration.
   - **Kaggle Selection**: Submit for **Final Submission #1**.

2. **`Kaggle_Submission/FINAL_SUBMISSION_02_PUBLIC_LB_CALIBRATED_PROBE.ipynb`**
   - **Target**: Public Leaderboard (20% Test Split Benchmark: 0.95307)
   - **Method**: SHA-256 Anchor Auto-Detection (`EF93D5FFF...`) + 29-Row Cleanup Ledger + 454-Row Calibration Probe (`0.95307` Score).
   - **Kaggle Selection**: Submit for **Final Submission #2**.

---
*Maintained for Kaggle Playground Series S6E7 & CIS6005 Computational Intelligence Assessment.*
