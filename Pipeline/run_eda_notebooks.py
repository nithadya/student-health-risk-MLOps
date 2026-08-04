import os
import sys
import time
import json
import subprocess
from datetime import datetime
from pathlib import Path

def setup_jupyter_kernel():
    """Ensure python3 ipykernel is registered quietly to prevent nbconvert warnings."""
    try:
        subprocess.run(
            [sys.executable, "-m", "ipykernel", "install", "--user", "--name", "python3"],
            capture_output=True, text=True, timeout=15
        )
    except Exception:
        pass

def run_all_eda():
    start_time = time.time()
    root_dir = Path(__file__).resolve().parent.parent
    eda_dir = root_dir / "EDA"
    figures_dir = eda_dir / "figures"
    artifacts_dir = eda_dir / "artifacts"
    
    figures_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    setup_jupyter_kernel()
    
    print("=================================================================================")
    print("  STUDENT HEALTH RISK ML SYSTEM -- AUTOMATED EDA EXECUTION ENGINE")
    print("=================================================================================")
    print(f"  Target Directory : {eda_dir.resolve()}")
    print(f"  Figures Directory: {figures_dir.resolve()}")
    print(f"  Artifacts Store  : {artifacts_dir.resolve()}")
    print(f"  Start Time       : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=================================================================================\n")
    
    notebooks = sorted(list(eda_dir.glob("*.ipynb")))
    if not notebooks:
        print("  [FAIL] No EDA notebooks found in EDA/ directory.")
        return
        
    descriptions = {
        "01_handling_missing_values.ipynb": "MNAR/MCAR Missingness Topology & Indicator Flag Generation",
        "02_handling_outliers.ipynb": "IQR Statistical Outlier Bounding & Seaborn Violin Density Plots",
        "03_feature_engineering.ipynb": "Lifestyle Risk Index & Biometric Interaction Ratio Generation",
        "04_Data_visualization.ipynb": "Multivariate Correlation Pairplots & Pearson/Spearman Heatmaps",
        "05_encoding_and_scalling.ipynb": "Categorical Ordinal Encoding & MinMaxScaler Transformation",
        "06_encoding_and_standarlization.ipynb": "StandardScaler Normalization & Z-Score Feature Scaling"
    }
    
    success_count = 0
    results_log = []
    
    for idx, nb_path in enumerate(notebooks, 1):
        nb_name = nb_path.name
        desc = descriptions.get(nb_name, "Exploratory Data Analysis Notebook")
        
        print(f"[{idx}/{len(notebooks)}] Processing: {nb_name}")
        print(f"  --> Focus Topic : {desc}")
        if "outliers" in nb_name.lower():
            print("  --> Note        : Computing KDE density over 690,088 rows (Estimated time: ~25-35s)...")
        else:
            print("  --> Note        : Ingesting dataset and generating figures...")
            
        nb_start = time.time()
        
        # We exclusively use the custom Python execution engine to ensure
        # display() is mocked and plt.show() explicitly exports to EDA/figures/
        try:
            with open(nb_path, "r", encoding="utf-8") as f:
                nb = json.load(f)
            py_code = []
            for cell in nb.get("cells", []):
                if cell.get("cell_type") == "code":
                    lines = cell.get("source", [])
                    clean_lines = [l for l in lines if not l.strip().startswith(("%", "!"))]
                    py_code.append("".join(clean_lines))
            
            # Injection to mock display() and override plt.show() to save images
            injection_code = """
import os
import pandas as pd
import matplotlib.pyplot as plt

def display(obj):
    if isinstance(obj, pd.DataFrame) or isinstance(obj, pd.Series):
        print("\\n" + obj.to_string() + "\\n")
    else:
        print(obj)

_original_show = plt.show
def custom_show(*args, **kwargs):
    fig = plt.gcf()
    if fig.axes:
        title = fig.axes[0].get_title()
        if not title:
            title = f"plot_{id(fig)}"
        safe_title = "".join([c if c.isalnum() else "_" for c in title])
        filepath = os.path.join("figures", f"{safe_title}.png")
        fig.savefig(filepath, bbox_inches="tight", dpi=300)
        print(f"  [+] Plot Matrix Generated & Saved -> {filepath}")
    _original_show(*args, **kwargs)

plt.show = custom_show
"""
            exec_code = injection_code + "\n\n" + "\n\n".join(py_code)
            
            orig_cwd = os.getcwd()
            try:
                os.chdir(eda_dir)
                global_env = {"__name__": "__main__"}
                exec(exec_code, global_env)
                elapsed = time.time() - nb_start
                print(f"  [SUCCESS] {nb_name} executed cleanly in {elapsed:.1f}s.\n")
                success_count += 1
                results_log.append((nb_name, "SUCCESS", f"{elapsed:.1f}s"))
            finally:
                os.chdir(orig_cwd)
        except Exception as e:
            elapsed = time.time() - nb_start
            print(f"  [FAILED] {nb_name} - Error: {str(e)}\n")
            results_log.append((nb_name, "FAILED", f"{elapsed:.1f}s"))
                
    total_time = time.time() - start_time
    print("=================================================================================")
    print("  AUTOMATED EDA EXECUTION SUMMARY REPORT")
    print("=================================================================================")
    for name, status, duration in results_log:
        status_str = "[OK] SUCCESS" if status == "SUCCESS" else "[X] FAILED "
        print(f"  {status_str}  |  {name:<42}  | Duration: {duration}")
    print("---------------------------------------------------------------------------------")
    print(f"  Total Status     : {success_count}/{len(notebooks)} Notebooks Executed Successfully")
    print(f"  Total Duration   : {total_time:.1f} seconds ({total_time/60:.2f} minutes)")
    print(f"  Output Figures   : Saved to {figures_dir.resolve()}")
    print("=================================================================================\n")

if __name__ == "__main__":
    run_all_eda()
