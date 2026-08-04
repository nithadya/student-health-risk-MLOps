.PHONY: venv validate install eda data train train-regression train-classification train-clustering inference mlflow-ui all clean help

PYTHON = python
PIP = pip
VENV_NAME = .venv

ifeq ($(OS),Windows_NT)
	VENV_PYTHON = ../$(VENV_NAME)/Scripts/python.exe
else
	VENV_PYTHON = ../$(VENV_NAME)/bin/python
endif

venv:
	@if [ -d "../$(VENV_NAME)" ]; then \
		echo "Virtual Environment '$(VENV_NAME)' already exists in project root."; \
	else \
		echo "Creating Python Virtual Environment in project root..."; \
		cd .. && $(PYTHON) -m venv $(VENV_NAME); \
		echo "Virtual Environment '$(VENV_NAME)' created successfully!"; \
	fi

install:
	@echo "Installing Dependencies from requirements.txt into active environment..."
	$(PYTHON) -m pip install -r ../requirements.txt

validate:
	@echo "Running Environment Validation & Dependency Verification..."
	$(PYTHON) validate_environment.py

eda:
	@echo "Executing all 6 EDA Laboratory Notebooks..."
	$(PYTHON) run_eda_notebooks.py

data:
	@echo "Running Production Data Pipeline..."
	$(PYTHON) pipelines/data_pipeline.py

train:
	@echo "Running Multi-Paradigm Training Pipeline (Default Classification)..."
	$(PYTHON) pipelines/training_pipeline.py

train-classification:
	@echo "Running Classification Training (LGBM + XGB + CatBoost)..."
	@echo "classification" | $(PYTHON) pipelines/training_pipeline.py

train-regression:
	@echo "Running Regression Training (Student Risk Score Index)..."
	@echo "regression" | $(PYTHON) pipelines/training_pipeline.py

train-clustering:
	@echo "Running Clustering & Dimensionality Reduction Pipeline (K-Means/DBSCAN/PCA/t-SNE/UMAP)..."
	@echo "clustering\npca" | $(PYTHON) pipelines/training_pipeline.py

inference:
	@echo "Running Production Inference Engine..."
	$(PYTHON) pipelines/inference_pipeline.py

mlflow-ui:
	@echo "Launching MLflow Experiment Tracking UI Dashboard on port 5000..."
	mlflow ui --port 5000

all: validate eda data train inference
	@echo "=== Full System Lifecycle (Validation -> EDA -> Data -> Train -> Inference) Completed Successfully ==="

clean:
	$(PYTHON) -c "import pathlib; [p.unlink() for p in pathlib.Path('.').rglob('*.pyc')]"
	$(PYTHON) -c "import pathlib; [p.rmdir() for p in pathlib.Path('.').rglob('__pycache__')]"

help:
	@echo "Available commands:"
	@echo "  make venv                 - Create virtual environment (.venv) if missing"
	@echo "  make install              - Install requirements.txt dependencies into active environment"
	@echo "  make validate             - Validate environment, Python version & dependencies"
	@echo "  make eda                  - Execute all 6 EDA laboratory notebooks"
	@echo "  make data                 - Run production data pipeline"
	@echo "  make train                - Run training pipeline (Classification)"
	@echo "  make train-classification - Run classification training pipeline"
	@echo "  make train-regression     - Run regression training pipeline"
	@echo "  make train-clustering     - Run clustering/dim-reduction pipeline"
	@echo "  make inference            - Run inference pipeline for batch predictions"
	@echo "  make mlflow-ui            - Launch interactive MLflow UI server"
	@echo "  make all                  - Execute full lifecycle (Validate -> EDA -> Data -> Train -> Inference)"
	@echo "  make clean                - Remove temporary cached files"
