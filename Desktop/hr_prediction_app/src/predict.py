import pandas as pd
import numpy as np
import os
import mlflow
from sklearn.metrics import recall_score, accuracy_score, f1_score, roc_auc_score, classification_report
from ml_model import _simple_eng

THRESHOLD = 0.5  # можно изменить на 0.66, если хотите

def predict(model, df_test: pd.DataFrame):
    X_test = df_test.drop('Attrition', axis=1)
    y_test = df_test['Attrition']
    
    # Применяем ту же feature engineering
    X_test = _simple_eng(X_test)
    
    probs = model.predict_proba(X_test)[:, 1]
    y_pred = (probs >= THRESHOLD).astype(int)
    
    recall = recall_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, probs)
    report = classification_report(y_test, y_pred)
    
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    artifacts_dir = os.path.join(root_dir, 'artifacts')
    os.makedirs(artifacts_dir, exist_ok=True)
    
    with open(os.path.join(artifacts_dir, 'recall_score.txt'), 'w') as f:
        f.write(f"Recall: {recall}")
    with open(os.path.join(artifacts_dir, 'classification_report.txt'), 'w') as f:
        f.write(report)
    
    mlflow.set_experiment("hr-attrition-exp")
    with mlflow.start_run(run_name="Evaluation", nested=True):
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1", f1)
        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.log_text(report, "classification_report.txt")
    
    print(f"Test Recall: {recall:.4f}, Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
    return recall