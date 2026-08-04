.PHONY: venv validate install eda data train train-regression train-classification train-clustering inference mlflow-ui serve test all clean help

PYTHON = python
PIP = pip
VENV_NAME = .venv

venv:
	@if [ -d "$(VENV_NAME)" ]; then \
		echo "Virtual Environment '$(VENV_NAME)' already exists."; \
	else \
		echo "Creating Python Virtual Environment ($(VENV_NAME))..."; \
		$(PYTHON) -m venv $(VENV_NAME); \
		echo "Virtual Environment '$(VENV_NAME)' created successfully!"; \
	fi
	@echo ""
	@echo "Activate it using:"
	@echo "  Windows Git Bash / MSYS : source $(VENV_NAME)/Scripts/activate"
	@echo "  Windows PowerShell     : .\\$(VENV_NAME)\\Scripts\\Activate.ps1"
	@echo "  Linux / macOS          : source $(VENV_NAME)/bin/activate"

install:
	@echo "Installing Dependencies from requirements.txt into active environment..."
	$(PIP) install -r requirements.txt

validate:
	@echo "Running Environment Validation & Dependency Verification..."
	$(PYTHON) Pipeline/validate_environment.py

eda:
	@echo "Executing all 6 EDA Laboratory Notebooks..."
	$(PYTHON) Pipeline/run_eda_notebooks.py

data:
	@echo "Running Production Data Pipeline..."
	$(PYTHON) Pipeline/pipelines/data_pipeline.py

train:
	@echo "Running Multi-Paradigm Training Pipeline..."
	$(PYTHON) Pipeline/pipelines/training_pipeline.py

train-classification:
	@echo "Running Classification Training..."
	echo classification | $(PYTHON) Pipeline/pipelines/training_pipeline.py

train-regression:
	@echo "Running Regression Training..."
	echo regression | $(PYTHON) Pipeline/pipelines/training_pipeline.py

train-clustering:
	@echo "Running Clustering & Dimensionality Reduction Pipeline..."
	echo clustering\npca | $(PYTHON) Pipeline/pipelines/training_pipeline.py

inference:
	@echo "Running Production Inference Engine..."
	$(PYTHON) Pipeline/pipelines/inference_pipeline.py

mlflow-ui:
	$(PYTHON) Pipeline/run_mlflow_ui.py

serve:
	@echo "🚀 Launching Enterprise Student Health ML Web Application Server on http://127.0.0.1:5000..."
	$(PYTHON) Web_App/app.py

test:
	@echo "🧪 Running Production Automated Unit & Integration Test Suite..."
	$(PYTHON) tests/test_api.py
	$(PYTHON) tests/test_pipeline.py

all: validate eda data train inference
	@echo "=== Full System Lifecycle (Validation -> EDA -> Data -> Train -> Inference) Completed Successfully ==="

clean:
	$(PYTHON) -c "import pathlib; [p.unlink() for p in pathlib.Path('.').rglob('*.pyc')]"
	$(PYTHON) -c "import pathlib; [p.rmdir() for p in pathlib.Path('.').rglob('__pycache__')]"

help:
	@echo "Available commands:"
	@echo "  make venv                 - Create virtual environment (.venv) if it doesn't exist"
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
