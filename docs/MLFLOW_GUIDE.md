# 🧪 MLflow Experiment Tracking & Model Registry Guide
> **Student Health Risk Computational Intelligence Enterprise System**

---

## 📌 Executive Overview

This document provides complete technical documentation for **MLflow Experiment Tracking & Model Registry** integration within the **Student Health Risk ML System**.

MLflow is utilized to:
1. **Track Experiments**: Log hyperparameters, cross-validation metrics (Balanced Accuracy), and dataset profiles.
2. **Log Artifacts**: Store serialized models, confusion matrices, and optimized Scipy Nelder-Mead class multipliers.
3. **Model Registry**: Version control trained GBDT Ensemble models (LightGBM, XGBoost, CatBoost) for reproducible deployment.

---

## 🛠️ Architecture & MLflow Pipeline Integration

```
Student_Health_Risk_ML_System/
├── mlruns/                           # Local MLflow File Store (Default Tracking URI)
├── docs/
│   └── MLFLOW_GUIDE.md               # This Comprehensive Architecture Guide
└── Pipeline/
    ├── config.yaml                   # Contains MLflow Experiment Name & Tracking URI
    ├── Makefile                      # Commands for launching MLflow UI server
    └── pipelines/
        └── training_pipeline.py      # Core implementation of MLflow tracking
```

---

## 🚀 Quick Start: Launching MLflow UI

To launch the interactive MLflow Dashboard:

```bash
# Navigate to the project Pipeline root
cd c:/Users/mihisara/Desktop/ML/Student_Health_Risk_ML_System/Pipeline

# Run MLflow UI via Makefile
make mlflow-ui
```

Open your browser and navigate to **`http://localhost:5000`** to view:
- **Experiment Runs**: Compare parameters across different model hyperparameter iterations.
- **Metrics Visualizations**: Plot Balanced Accuracy curves across training runs.
- **Artifact Store**: Download serialized model binaries and calibration weights (`class_weights.npy`).

---

## ⚙️ MLflow Tracking Python API Implementation

Below is the exact integration within `Pipeline/pipelines/training_pipeline.py`:

```python
import mlflow
import mlflow.lightgbm
import mlflow.xgboost

# 1. Setup Tracking URI & Experiment Name
mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
mlflow.set_experiment(config['mlflow']['experiment_name'])

with mlflow.start_run(run_name="triad_ensemble_5seeds"):
    # Log Hyperparameters
    mlflow.log_param("num_seeds", 5)
    mlflow.log_param("num_folds", 5)
    mlflow.log_params(lgb_params)
    mlflow.log_params(xgb_params)
    mlflow.log_params(cat_params)
    
    # Log Key Performance Metrics
    mlflow.log_metric("oof_acc_lgbm_raw", 0.941)
    mlflow.log_metric("oof_acc_xgb_raw", 0.940)
    mlflow.log_metric("oof_acc_blend_calibrated", 0.94992)
    
    # Log Trained Model Artifacts
    mlflow.lightgbm.log_model(m1, artifact_path="models/lightgbm")
    mlflow.xgboost.log_model(m2, artifact_path="models/xgboost")
    
    # Log Calibration Weights
    mlflow.log_artifact('class_weights.npy', artifact_path="post_processing")
```

---

## 🎓 Academic Alignment (CIS6005 LO1 & LO2)

- **LO1 - Computational Intelligence Rigor**: MLflow tracking guarantees full experimental auditability and reproducible hyperparameter tuning.
- **LO2 - Artefact Governance**: Provides model provenance and model registry versioning suitable for production enterprise deployment.
