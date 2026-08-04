import sys
import os
import platform
import subprocess
from pathlib import Path

def validate():
    print("=========================================================")
    print("  Student Health Risk ML System Environment Validation")
    print("=========================================================")
    
    # 1. OS & Python Check
    print("\n[1/5] OS & System Information:")
    print(f"  - Operating System : {platform.system()} {platform.release()} ({platform.architecture()[0]})")
    print(f"  - Python Executable: {sys.executable}")
    print(f"  - Python Version   : {sys.version.split()[0]}")
    
    if sys.version_info < (3, 10):
        print("  [WARN] Python 3.10+ is recommended!")
    else:
        print("  [OK] Python version requirement satisfied (>= 3.10).")
        
    # 2. Virtual Environment Check
    print("\n[2/5] Virtual Environment Status:")
    in_venv = (sys.prefix != sys.base_prefix) or ("CONDA_PREFIX" in os.environ)
    if in_venv:
        env_name = os.environ.get("CONDA_DEFAULT_ENV", Path(sys.prefix).name)
        print(f"  [OK] Active Virtual Environment Detected: '{env_name}' ({sys.prefix})")
    else:
        print("  [WARN] Running outside a virtual environment! Recommended to use venv or conda.")
        
    # 3. Key Machine Learning Packages Verification
    print("\n[3/5] Package Dependencies Check:")
    required_packages = [
        "numpy", "pandas", "scipy", "sklearn", "lightgbm",
        "xgboost", "catboost", "joblib", "yaml", "jupyter"
    ]
    
    missing = []
    for pkg in required_packages:
        try:
            if pkg == "sklearn":
                import sklearn
                ver = sklearn.__version__
            elif pkg == "yaml":
                import yaml
                ver = yaml.__version__
            else:
                mod = __import__(pkg)
                ver = getattr(mod, "__version__", "installed")
            print(f"  [OK]   {pkg:<12}: Version {ver}")
        except ImportError:
            print(f"  [FAIL] {pkg:<12}: NOT INSTALLED")
            missing.append(pkg)
            
    if missing:
        print(f"\n  [FAIL] Missing dependencies detected: {', '.join(missing)}")
        print("  --> Run 'make install' or 'pip install -r requirements.txt' to install missing packages.")
    else:
        print("\n  [OK] All core Machine Learning & Data Science packages are installed.")
        
    # 4. Hardware Acceleration (GPU / CUDA) Check
    print("\n[4/5] Hardware Acceleration (GPU / CUDA) Check:")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  [OK] PyTorch CUDA Available: {torch.cuda.get_device_name(0)}")
        else:
            print("  [INFO] PyTorch CUDA Not Available (CPU Execution Mode).")
    except ImportError:
        print("  [INFO] PyTorch not installed (CPU GBDT mode active).")
        
    try:
        import xgboost as xgb
        print("  [OK] XGBoost GPU / CPU Engine Installed.")
    except Exception:
        pass
        
    # 5. Project Directory & Data Path Verification
    print("\n[5/5] Project Directory Structure Check:")
    root_dir = Path(__file__).resolve().parent.parent
    
    data_path = root_dir / "data" if (root_dir / "data").exists() else root_dir / "Pipeline" / "data"
    paths_to_check = [
        ("Data Directory", data_path),
        ("EDA Directory", root_dir / "EDA"),
        ("Pipeline Directory", root_dir / "Pipeline"),
        ("Model Training Directory", root_dir / "Model Training"),
        ("Kaggle Submission Directory", root_dir / "Kaggle_Submission"),
        ("Requirements File", root_dir / "requirements.txt"),
    ]
    
    for name, path in paths_to_check:
        if path.exists():
            print(f"  [OK]   {name:<30}: {path.relative_to(root_dir)}")
        else:
            print(f"  [FAIL] {name:<30}: MISSING ({path.resolve()})")
            
    print("\n=========================================================")
    print("  Validation Finished Successfully!")
    print("=========================================================\n")

if __name__ == "__main__":
    validate()
