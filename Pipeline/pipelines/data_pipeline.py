import os
import sys
import yaml
import logging
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# Ensure working directory is normalized to project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(PROJECT_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# Setup Production Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger('DataPipeline')

def load_config(config_path='Pipeline/config.yaml'):
    if not os.path.exists(config_path):
        if os.path.exists('config.yaml'):
            config_path = 'config.yaml'
        else:
            config_path = os.path.join(PROJECT_ROOT, 'Pipeline', 'config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def validate_schema(df, dataset_type='train'):
    """Verify data schema completeness and column presence"""
    required_numerical = ['sleep_duration', 'exercise_duration', 'step_count', 'calorie_expenditure', 'heart_rate', 'bmi']
    required_categorical = ['gender', 'stress_level', 'physical_activity_level', 'sleep_quality', 'smoking_alcohol', 'diet_type']
    
    missing_cols = [c for c in required_numerical + required_categorical if c not in df.columns]
    if missing_cols:
        logger.warning(f"[WARNING] Schema validation: Missing columns in {dataset_type}: {missing_cols}")
    else:
        logger.info(f"[OK] Schema validation PASSED for {dataset_type} ({len(df):,} rows, {len(df.columns)} columns)")

def run_data_pipeline():
    logger.info("=" * 60)
    logger.info("Running Enterprise Production Data Pipeline")
    logger.info("=" * 60)
    
    config = load_config()
    processed_dir = config['data']['processed_dir']
    os.makedirs(processed_dir, exist_ok=True)
    
    preprocessor_dir = config.get('artifacts', {}).get('encoder_dir', 'Pipeline/artifacts/preprocessor')
    preprocessor_dir = os.path.join('Pipeline', 'artifacts', 'preprocessor')
    os.makedirs(preprocessor_dir, exist_ok=True)
    
    train_path = config['data']['raw_train']
    test_path = config['data']['raw_test']
    
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Raw training data file not found at: {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Raw test data file not found at: {test_path}")
        
    logger.info(f"Ingesting raw datasets from {train_path} & {test_path}")
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    
    validate_schema(train, 'train')
    validate_schema(test, 'test')
    
    # Feature Engineering & Missingness Topology Engine
    def apply_feature_engineering(df):
        df = df.copy()
        
        # 1. MNAR Missingness Indicators
        num_features = ['sleep_duration', 'exercise_duration', 'step_count', 'calorie_expenditure', 'heart_rate', 'bmi', 'age']
        for col in num_features:
            if col in df:
                df[f'{col}_is_missing'] = df[col].isna().astype('int8')
        
        # 2. Sleep Distance from 8 Hours
        if 'sleep_duration' in df:
            df['sleep_distance_from_8'] = (df['sleep_duration'] - 8.0).abs()
            
        # 3. Lifestyle Risk Index Accumulator
        risk = pd.Series(0, index=df.index, dtype='int8')
        if 'sleep_duration' in df: risk += (df['sleep_duration'] < 6.0).astype('int8')
        if 'stress_level' in df: risk += (df['stress_level'].astype(str) == 'high').astype('int8')
        if 'sleep_quality' in df: risk += (df['sleep_quality'].astype(str) == 'poor').astype('int8')
        if 'physical_activity_level' in df: risk += (df['physical_activity_level'].astype(str) == 'sedentary').astype('int8')
        if 'smoking_alcohol' in df: risk += (df['smoking_alcohol'].astype(str) == 'yes').astype('int8')
        if 'bmi' in df: risk += (df['bmi'] >= 25.0).astype('int8')
        df['lifestyle_risk_index'] = risk
        
        # 4. Biometric Interaction Combinations
        if 'stress_level' in df and 'physical_activity_level' in df:
            df['stress_activity_combo'] = df['stress_level'].astype(str) + "_" + df['physical_activity_level'].astype(str)
        if 'sleep_quality' in df and 'stress_level' in df:
            df['sleep_stress_combo'] = df['sleep_quality'].astype(str) + "_" + df['stress_level'].astype(str)
        if 'lifestyle_risk_index' in df and 'stress_level' in df:
            df['lifestyle_triad'] = df['lifestyle_risk_index'].astype(str) + "_" + df['stress_level'].astype(str)
            
        return df

    logger.info("Computing MNAR missingness flags and domain biometric features...")
    train_processed = apply_feature_engineering(train)
    test_processed = apply_feature_engineering(test)
    
    # Fit & Serialize Production StandardScaler
    num_cols = train_processed.select_dtypes(include=['float64', 'int64']).columns.tolist()
    if 'id' in num_cols: num_cols.remove('id')
    
    scaler = StandardScaler()
    scaler.fit(train_processed[num_cols].fillna(train_processed[num_cols].mean()))
    scaler_path = os.path.join(preprocessor_dir, 'standard_scaler.joblib')
    joblib.dump(scaler, scaler_path)
    logger.info(f"Fitted & exported StandardScaler artifact to {scaler_path}")
    
    # Export Processed Data Binaries
    out_train = os.path.join(processed_dir, 'train_processed.csv')
    out_test = os.path.join(processed_dir, 'test_processed.csv')
    train_processed.to_csv(out_train, index=False)
    test_processed.to_csv(out_test, index=False)
    
    logger.info(f"Exported processed training dataset ({len(train_processed):,} rows) to {out_train}")
    logger.info(f"Exported processed testing dataset ({len(test_processed):,} rows) to {out_test}")
    logger.info("[SUCCESS] Data Pipeline Execution Finished Successfully.\n")

if __name__ == "__main__":
    run_data_pipeline()
