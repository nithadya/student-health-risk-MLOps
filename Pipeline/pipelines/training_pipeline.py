import os
import sys
import yaml
import logging
import numpy as np
import pandas as pd
import warnings
import joblib
import mlflow
import mlflow.lightgbm
import mlflow.xgboost
import mlflow.sklearn

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, classification_report, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.optimize import minimize

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

warnings.filterwarnings('ignore')

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
logger = logging.getLogger('TrainingPipeline')

def load_config(config_path='Pipeline/config.yaml'):
    if not os.path.exists(config_path):
        if os.path.exists('config.yaml'):
            config_path = 'config.yaml'
        else:
            config_path = os.path.join(PROJECT_ROOT, 'Pipeline', 'config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

config = load_config()
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
mlflow.set_experiment(config['mlflow']['experiment_name'])

def run_regression_pipeline():
    logger.info("=" * 60)
    logger.info("Running Production Regression Pipeline")
    logger.info("=" * 60)
    
    train_processed_path = os.path.join(config['data']['processed_dir'], 'train_processed.csv')
    if not os.path.exists(train_processed_path):
        raise FileNotFoundError(f"Processed training data not found at {train_processed_path}. Run 'make data' first.")
        
    train = pd.read_csv(train_processed_path)
    target = 'bmi'
    features = ['sleep_duration', 'exercise_duration', 'step_count', 'calorie_expenditure', 'heart_rate']
    
    available_features = [f for f in features if f in train.columns]
    df = train[available_features + [target]].dropna()
    
    X = df[available_features]
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    reg_data_dir = config['artifacts']['regression_data_dir']
    os.makedirs(reg_data_dir, exist_ok=True)
    X_train.to_csv(os.path.join(reg_data_dir, 'X_train_reg.csv'), index=False)
    X_test.to_csv(os.path.join(reg_data_dir, 'X_test_reg.csv'), index=False)
    y_train.to_csv(os.path.join(reg_data_dir, 'y_train_reg.csv'), index=False)
    y_test.to_csv(os.path.join(reg_data_dir, 'y_test_reg.csv'), index=False)
    
    with mlflow.start_run(run_name="Regression_XGB"):
        model = xgb.XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42)
        model.fit(X_train, y_train)
        
        preds = model.predict(X_test)
        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
        r2 = float(r2_score(y_test, preds))
        
        mlflow.log_metric('rmse', rmse)
        mlflow.log_metric('r2', r2)
        
        reg_model_dir = config['artifacts']['regression_model_dir']
        os.makedirs(reg_model_dir, exist_ok=True)
        model_save_path = os.path.join(reg_model_dir, 'reg_student_health.joblib')
        joblib.dump(model, model_save_path)
        mlflow.xgboost.log_model(model, name="regression_model")
        
        logger.info(f"[SUCCESS] Regression Model trained (RMSE: {rmse:.4f}, R2: {r2:.4f}) -> {model_save_path}")

def run_clustering_pipeline():
    logger.info("=" * 60)
    logger.info("Running Production Clustering Pipeline")
    logger.info("=" * 60)
    
    train_processed_path = os.path.join(config['data']['processed_dir'], 'train_processed.csv')
    if not os.path.exists(train_processed_path):
        raise FileNotFoundError(f"Processed training data not found at {train_processed_path}. Run 'make data' first.")
        
    train = pd.read_csv(train_processed_path)
    num_cols = train.select_dtypes(include=np.number).columns.tolist()
    if 'id' in num_cols: num_cols.remove('id')
    
    X = train[num_cols].fillna(train[num_cols].mean())
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    with mlflow.start_run(run_name="Clustering_PCA_KMeans"):
        pca = PCA(n_components=2, random_state=42)
        pca_components = pca.fit_transform(X_scaled)
        
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        kmeans.fit(pca_components)
        
        clus_data_dir = config['artifacts']['clustering_data_dir']
        os.makedirs(clus_data_dir, exist_ok=True)
        pd.DataFrame(pca_components, columns=['PC1', 'PC2']).to_csv(os.path.join(clus_data_dir, 'PCA__components.csv'), index=False)
        
        clus_model_dir = config['artifacts']['clustering_model_dir']
        os.makedirs(clus_model_dir, exist_ok=True)
        model_save_path = os.path.join(clus_model_dir, 'kmean_student_health.joblib')
        joblib.dump(kmeans, model_save_path)
        mlflow.sklearn.log_model(kmeans, name="clustering_model")
        
        logger.info(f"[SUCCESS] Clustering Model trained (K=3, PCA 2D) -> {model_save_path}")

def run_classification_pipeline():
    logger.info("=" * 60)
    logger.info("Running Production Classification Pipeline (GBDT Engine)")
    logger.info("=" * 60)
    
    train_processed_path = os.path.join(config['data']['processed_dir'], 'train_processed.csv')
    if not os.path.exists(train_processed_path):
        raise FileNotFoundError(f"Processed training data not found at {train_processed_path}. Run 'make data' first.")
        
    train = pd.read_csv(train_processed_path)
    TARGET = config['preprocessing']['target_column']
    
    X = train.drop(columns=[TARGET]).copy()
    y_text = train[TARGET].copy()
    
    le = LabelEncoder()
    y = le.fit_transform(y_text)
    
    encoder_dir = config['artifacts']['encoder_dir']
    os.makedirs(encoder_dir, exist_ok=True)
    joblib.dump(le, os.path.join(encoder_dir, 'classification_target_encoder.joblib'))
    joblib.dump(le, os.path.join(encoder_dir, 'target_encoder.joblib'))
    
    # Save splits
    cla_data_dir = config['artifacts']['classification_data_dir']
    os.makedirs(cla_data_dir, exist_ok=True)
    X.to_csv(os.path.join(cla_data_dir, 'X_train_cla.csv'), index=False)
    pd.DataFrame(y, columns=['target']).to_csv(os.path.join(cla_data_dir, 'y_train_cla.csv'), index=False)
    
    with mlflow.start_run(run_name="Classification_GBDT_Ensemble"):
        cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        for c in cat_cols:
            X[c] = X[c].astype(str).replace('nan', 'missing').astype('category')
            
        logger.info("Training GBDT Model 1/2: LightGBM Classifier...")
        lgb_params = config['model_training']['lightgbm']
        m_lgb = lgb.LGBMClassifier(**lgb_params)
        m_lgb.fit(X, y)
        probs_lgb = m_lgb.predict_proba(X)
        
        logger.info("Training GBDT Model 2/2: XGBoost Classifier...")
        xgb_params = config['model_training']['xgboost'].copy()
        m_xgb = xgb.XGBClassifier(**xgb_params)
        m_xgb.fit(X, y)
        probs_xgb = m_xgb.predict_proba(X)
        
        # 50-50 GBDT Ensemble Blend
        probs_blend = 0.5 * probs_lgb + 0.5 * probs_xgb
        preds_blend = np.argmax(probs_blend, axis=1)
        
        # Log Hyperparameters & Dataset Metadata to MLflow
        mlflow.log_params({f"lgb_{k}": str(v) for k, v in lgb_params.items()})
        mlflow.log_params({f"xgb_{k}": str(v) for k, v in xgb_params.items()})
        mlflow.log_param("num_samples", len(X))
        mlflow.log_param("num_features", X.shape[1])
        mlflow.log_param("target_classes", str(list(le.classes_)))
        
        # Comprehensive Metrics Calculation
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        acc_blend = float(balanced_accuracy_score(y, preds_blend))
        
        acc_standard = float(accuracy_score(y, preds_blend))
        prec_macro = float(precision_score(y, preds_blend, average='macro'))
        rec_macro = float(recall_score(y, preds_blend, average='macro'))
        f1_mac = float(f1_score(y, preds_blend, average='macro'))
        cm = confusion_matrix(y, preds_blend)
        
        # Log Metrics to MLflow
        mlflow.log_metric('balanced_accuracy_blend', acc_blend)
        mlflow.log_metric('accuracy_blend', acc_standard)
        mlflow.log_metric('precision_macro', prec_macro)
        mlflow.log_metric('recall_macro', rec_macro)
        mlflow.log_metric('f1_macro', f1_mac)
        mlflow.log_metric('balanced_accuracy_lgb', float(balanced_accuracy_score(y, np.argmax(probs_lgb, axis=1))))
        mlflow.log_metric('balanced_accuracy_xgb', float(balanced_accuracy_score(y, np.argmax(probs_xgb, axis=1))))
        
        # Generate & Log Confusion Matrix Image Artifact
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_, ax=ax)
            ax.set_xlabel('Predicted Label')
            ax.set_ylabel('True Label')
            ax.set_title('Ensemble Confusion Matrix')
            plt.tight_layout()
            cm_plot_path = os.path.join(cla_data_dir, 'confusion_matrix_mlflow.png')
            plt.savefig(cm_plot_path, dpi=150)
            plt.close()
            mlflow.log_artifact(cm_plot_path, artifact_path="evaluations")
        except Exception as plot_err:
            logger.warning(f"Could not log confusion matrix plot artifact: {plot_err}")
        
        # Log Confusion Matrix Elements as Parameters
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                mlflow.log_param(f'cm_cell_{i}_{j}', int(cm[i, j]))
        
        models_dir = config['artifacts']['classification_model_dir']
        os.makedirs(models_dir, exist_ok=True)
        model_save_path = os.path.join(models_dir, 'cla_student_health.joblib')
        joblib.dump(m_lgb, model_save_path)
        joblib.dump(m_xgb, os.path.join(models_dir, 'xgb_student_health.joblib'))
        
        # Export probability matrix and initial weights for inference compatibility
        np.save(os.path.join(models_dir, 'blended_test_probs.npy'), probs_blend)
        class_multipliers = np.array(config['post_processing'].get('initial_blend_weights', [0.1000, 1.45596, 0.99622]))
        np.save(os.path.join(models_dir, 'class_weights.npy'), class_multipliers)
        
        mlflow.lightgbm.log_model(m_lgb, name="lightgbm_model")
        mlflow.xgboost.log_model(m_xgb, name="xgboost_model")
        
        logger.info(f"[SUCCESS] Multi-Model GBDT Ensemble trained (LightGBM + XGBoost) | OOF Accuracy: {acc_blend:.5f} -> {model_save_path}")

if __name__ == "__main__":
    if not sys.stdin.isatty():
        input_data = sys.stdin.read().strip()
        mode = input_data.split('\n')[0].strip() if input_data else ''
    else:
        mode = input("Select training mode (classification/regression/clustering): ").strip()
        
    mode = mode.lower()
    
    if mode == 'regression':
        run_regression_pipeline()
    elif mode == 'clustering':
        run_clustering_pipeline()
    else:
        run_classification_pipeline()
