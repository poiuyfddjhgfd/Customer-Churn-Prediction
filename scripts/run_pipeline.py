"""
Run Sequential : load--> validate-->preprocess-->feature engineering
"""

import os
import sys
import time
import argparse
import pandas as pd
import mlflow
import mlflow.sklearn
from posthog import project_root
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
from xgboost import XGBClassifier

# === Fix Import path for local modules==
# Essential : Allows import from src/directory structure
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"..")))

# Local modules -Core Pipeline components
from src.data.load_data import load_data                   # Data Loading with error handling
from src.data.preprocess import preprocess_data            # Basic Data Cleaning
from src.features.build_features import build_features     # Feature engineering (CRITICAL for model performance)
from src.utils.validate_data import validate_telco_data     # Data Quality validation

def main(args):
    """
    Main Training pipelines function that orchestrates the complete ML workflow  
        """
    ### === Mlflow setup-Essential for experiment tracking===
    # Configure MlFlow to use local file based tracking (not a tracking server)
    project_root=os.path.abspath(os.path.join(os.path.dirname(__file__),".."))
    mlrun_path=args.mlflow_url or f"file://{project_root}/mlruns"
    mlflow.set_tracking_uri(mlrun_path)
    mlflow.set_experiment(args.experiment)  # Create experiment if doen not exist
    
    # Start Mlflow run-- all subsequent logging will be tracked under this run
    with mlflow.start_run():
        # === Log Hyperparameters and configuration===
        # Required : These parameters are essential for model reproductivity
        mlflow.log_param("model","xgboost")   # Model type for comparison
        mlflow.log_param("threshold",args.threshold)    # Classification threshold (default : 0.35)
        mlflow.log_param("test_size",args.test_size)    # train/test split ratio

        #===Stage 1: Data Loading & Validation ===
        print("Loading Data")
        df=load_data(args,input)  # Load raw CSV data with error handling
        print(f"Data loaded:{df.shape[0]} rows,{df.shape[1]} columns")

        # === Critical : Data Quality Validation===
        # This step is Essential for production ML - Validates data quality before training
        print("Validating Data Quality with Great Expectations")
        is_valid,failed=validate_telco_data(df)
        mlflow.log_metric("data_quality_pass",int(is_valid)) # track data quality over time

        if not is_valid:
            # Log Validation failures for debugging
            import json
            mlflow.log_text(json.dumps(failed,indent=2),artifact_file="failed_expectations.json")
            raise ValueError(f"Data quality check failed.Issues :{failed}")
        else:
            print("Data validation passed. Logged to Mlflow")

        # === STAGE 2: DATA PREPROCESSING===
        print("Preprocessing Data")
        df=preprocess_data(df)        # Basic cleaning (handle missing values,fix data types)

        # Save preprocessed datset for reproductivity and debugging
        processed_path=os.path.join(project_root,"data","processed","telco_churn_processed.csv")
        os.makedirs(os.path.dirname(processed_path),exist_ok=True)
        df.to_csv(processed_path,index=False)
        print(f"Preprocessed dataset saved to{processed_path}| Shape:{df.shape}")

        # === STAGE 3: Featur Engineering -
        print("Building Features")
        target=args.target
        if target not in df.columns:
            raise ValueError(f"Target column'{target}' not found in data")
        
        # Apply feature engineering transformations
        df_enc=build_features(df,target_col=target)    # Binary encoding + one hot encoding

        # IMPORTANT: Convert boolean columns to integers for XGBoost compatibility
        for c in df_enc.select_dtypes(include=["bool"]).columns:
            df_enc[c]=df_enc[c].astype(int)
        print(f"Features engineering completed :{df_enc.shape[1]} features")

        #=== Critical : Save Features MetaDAta for Serving Consistency===
        # This ensures serving pipelines uses exact same features in exact same order
        import json,joblib
        artifacts_dir=os.path.join(project_root,"artifacts")
        os.makedirs(artifacts_dir,exist_ok=True)

        # Get Feature columns (exclude target)
        feature_cols=list(df_enc.drop(columns=[target]).columns)

        # Save locally for development serving
        with open(os.path.join(artifacts_dir,"feature_columns.json"),"w") as f:
            json.dump(feature_cols,f) 

        #Log to MLFlow for Production serving
        mlflow.log_text("\n".join(feature_cols),artifact_file="feature_columns.txt") 
          
        # Essential: Save preprocessing artifacts for serving pipelines
        # These artifacts ensure training and serving use identival transformation 
        preprocessing_artifacts={
            "feature_columns": feature_cols,  # Exact features order
            "target":target                   # Target columns name

        }  

        joblib.dump(preprocessing_artifacts,os.path.join(artifacts_dir,"preprocessing.pkl"))
        mlflow.log_artifact(os.path.join(artifact_dir,"preprocessing.pkl"))
        print(f"Saved {len(features_col)} features columns for serving consistency")

        # === Stage 4: Train/Test Split===
        print("Spliting data")
        X=df_enc.drop(columns=[target])    #Feature matrix
        y=df_enc[target]                    # Target vector

        # Stratified split to maintain class distribution in both sets
        X_train,X_test,y_train,y_test=train_test_split(
            X,y,
            test_size=args.test_size,     #Default : 20% for testing
            stratify=y,                    #Maintain class balance
            random_state=42
        )

        print(f"train:{X_train.shape[0]} samples| Test:{X_test.shape[0]} samples")

        #=== CRITICAL : Handle Class Imbalance===
        # Calculate scale_pos_weight to handle imbalanced dataset
        # This tells XGBoost to give more weight to the minority class(churners)
        scale_pos_weight=(y_train==0).sum()/(y_train==1).sum()
        print(f"Class imbalance ratio:{scale_pos_weight:.2f}(applied to positive class)")


        # === STAGE 5: Model Training with Optimized Hyperparameters===
        print("Training XGBoost model.")

        # IMPORTANT: These hyperparameter were optimized through hyperparameter tuning
        # In Production, consider using hyperparameter optimization tools like Optuna
        model=XGBClassifier(
            # trees structure parameters
            n_estimators=310,              # Number of trees
            learning_rate=0.034,            # Step size shrinkage
            max_depth=7,                   # Maximum tree depth

        # Regularization parameters
        subsample=0.95,                       # Sample ratio of training instances
        colsample_bytree=0.98,                  # Sample ratio of features for each tree

        # Performance parameter
        n_jobs=-1,    # use all CPU
        random_state=42,   #Reproducible results
        eval_metric="logloss",    #Evaluation metric

        #ESSENTIAL: Handle class imbalance
        scale_pos_weight=scale_pos_weight  # Weight for positive class (churners)

        )
        
        
        # === Train Model and Track Training Time ===
        t0 = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - t0
        mlflow.log_metric("train_time", train_time)  # Track training performance
        print(f"  Model trained in {train_time:.2f} seconds")

        # === STAGE 6: Model Evaluation ===
        print(" Evaluating model performance...")
        
        # Generate predictions and track inference time
        t1 = time.time()
        proba = model.predict_proba(X_test)[:, 1]  # Get probability of churn (class 1)
        
        # Apply classification threshold (default: 0.35, optimized for churn detection)
        # Lower threshold = more sensitive to churn (higher recall, lower precision)
        y_pred = (proba >= args.threshold).astype(int)
        pred_time = time.time() - t1
        mlflow.log_metric("pred_time", pred_time)  # Track inference performance

        # === CRITICAL: Log Evaluation Metrics to MLflow ===
        # These metrics are essential for model comparison and monitoring
        precision = precision_score(y_test, y_pred)    # Of predicted churners, how many actually churned?
        recall = recall_score(y_test, y_pred)          # Of actual churners, how many did we catch?
        f1 = f1_score(y_test, y_pred)                  # Harmonic mean of precision and recall
        roc_auc = roc_auc_score(y_test, proba)         # Area under ROC curve (threshold-independent)
        
        # Log all metrics for experiment tracking
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall) 
        mlflow.log_metric("f1", f1)
        mlflow.log_metric("roc_auc", roc_auc)
        
        print(f"   Model Performance:")
        print(f"   Precision: {precision:.3f} | Recall: {recall:.3f}")
        print(f"   F1 Score: {f1:.3f} | ROC AUC: {roc_auc:.3f}")

        # === STAGE 7: Model Serialization and Logging ===
        print(" Saving model to MLflow...")
        # ESSENTIAL: Log model in MLflow's standard format for serving
        mlflow.sklearn.log_model(
            model, 
            artifact_path="model"  # This creates a 'model/' folder in MLflow run artifacts
        )
        print(" Model saved to MLflow for serving pipeline")

        # === Final Performance Summary ===
        print(f"\n  Performance Summary:")
        print(f"   Training time: {train_time:.2f}s")
        print(f"   Inference time: {pred_time:.4f}s")
        print(f"   Samples per second: {len(X_test)/pred_time:.0f}")
        
        print(f"\n Detailed Classification Report:")
        print(classification_report(y_test, y_pred, digits=3))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Run churn pipeline with XGBoost + MLflow")
    p.add_argument("--input", type=str, required=True,
                   help="path to CSV (e.g., data/raw/Telco-Customer-Churn.csv)")
    p.add_argument("--target", type=str, default="Churn")
    p.add_argument("--threshold", type=float, default=0.35)
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--experiment", type=str, default="Telco Churn")
    p.add_argument("--mlflow_uri", type=str, default=None,
                    help="override MLflow tracking URI, else uses project_root/mlruns")

    args = p.parse_args()
    main(args)

"""
# Use this below to run the pipeline:

python scripts/run_pipeline.py \                                            
    --input data/raw/Telco-Customer-Churn.csv \
    --target Churn

"""