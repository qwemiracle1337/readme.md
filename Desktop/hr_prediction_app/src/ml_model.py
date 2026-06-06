import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from dotenv import load_dotenv
import os

load_dotenv()

def train_model(df_train: pd.DataFrame, df_val: pd.DataFrame = None):
    """
    Train LogisticRegression with SMOTE and return model.
    """
    X = df_train.drop('Attrition', axis=1)
    y = df_train['Attrition']
    
    # Feature engineering (полностью повторяет последний успешный ноутбук)
    X = _simple_eng(X)
    
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False), categorical_cols)
    ])
    
    # Pipeline с SMOTE и логистической регрессией
    pipeline = ImbPipeline([
        ('prep', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('clf', LogisticRegression(C=0.1, class_weight='balanced', max_iter=1000, random_state=42))
    ])
    
    start = time.time()
    pipeline.fit(X, y)
    train_time = time.time() - start
    
    mlflow.set_experiment("hr-attrition-exp")
    with mlflow.start_run(run_name=os.getenv("MODEL_NAME", "LogisticRegression")):
        mlflow.log_params(pipeline.named_steps['clf'].get_params())
        mlflow.log_metric("train_time_sec", train_time)
        mlflow.sklearn.log_model(pipeline, "model")
    
    return pipeline

def _simple_eng(df):
    df = df.copy()
    drop_cols = ['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours',
                 'DailyRate', 'MonthlyRate', 'StockOptionLevel', 'PercentSalaryHike']
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)
    if 'OverTime' in df.columns:
        df['OverTime'] = (df['OverTime'] == 'Yes').astype(int)
    if 'MaritalStatus' in df.columns:
        df['IsSingle'] = (df['MaritalStatus'] == 'Single').astype(int)
    if 'BusinessTravel' in df.columns:
        df['Travel_Rarely'] = (df['BusinessTravel'] == 'Travel_Rarely').astype(int)
        df['Travel_Frequently'] = (df['BusinessTravel'] == 'Travel_Frequently').astype(int)
    if 'Gender' in df.columns:
        df['Male'] = (df['Gender'] == 'Male').astype(int)
    df['tenure_ratio'] = df['YearsAtCompany'] / (df['TotalWorkingYears'] + 1e-5)
    df['satisfaction'] = (df['JobSatisfaction'] + df['EnvironmentSatisfaction'] + df['RelationshipSatisfaction']) / 3
    df.drop(columns=['MaritalStatus', 'BusinessTravel', 'Gender'], inplace=True, errors='ignore')
    return df