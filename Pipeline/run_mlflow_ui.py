import os
import sys

# Ensure local file store is allowed in MLflow 2.x+
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

import mlflow.cli

if __name__ == "__main__":
    port = "5001"
    print(f"Launching MLflow Experiment Tracking UI Dashboard on http://127.0.0.1:{port}...")
    sys.argv = ["mlflow", "ui", "--port", port]
    mlflow.cli.cli()
