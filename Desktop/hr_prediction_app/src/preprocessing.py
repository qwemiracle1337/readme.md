import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline


def drop_columns(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    cols_to_drop = [
        'EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours',
        'DailyRate', 'MonthlyRate', 'StockOptionLevel', 'PercentSalaryHike'
    ]
    cols_to_drop = [c for c in cols_to_drop if c in X.columns]
    if cols_to_drop:
        X = X.drop(columns=cols_to_drop, axis=1)
    return X


def fill_na(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    object_cols = X.select_dtypes(include=['object', 'category']).columns
    for col in object_cols:
        X[col] = X[col].fillna('0')
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if X[col].isna().any():
            X[col] = X[col].fillna(X[col].mean())
    return X


def to_bin(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    if 'OverTime' in X.columns:
        X['OverTime'] = (X['OverTime'] == 'Yes').astype(int)
    if 'MaritalStatus' in X.columns:
        X['IsSingle'] = (X['MaritalStatus'] == 'Single').astype(int)
    if 'BusinessTravel' in X.columns:
        X['Travel_Rarely'] = (X['BusinessTravel'] == 'Travel_Rarely').astype(int)
        X['Travel_Frequently'] = (X['BusinessTravel'] == 'Travel_Frequently').astype(int)
    if 'Gender' in X.columns:
        X['Male'] = (X['Gender'] == 'Male').astype(int)
    X['tenure_ratio'] = X['YearsAtCompany'] / (X['TotalWorkingYears'] + 1e-5)
    X['satisfaction'] = (X['JobSatisfaction'] + X['EnvironmentSatisfaction'] + X['RelationshipSatisfaction']) / 3
    X.drop(columns=['MaritalStatus', 'BusinessTravel', 'Gender'], inplace=True, errors='ignore')
    return X


def preprocessing() -> Pipeline:
    drop_transformer = FunctionTransformer(drop_columns)
    fill_na_transformer = FunctionTransformer(fill_na)
    to_bin_transformer = FunctionTransformer(to_bin)
    return Pipeline([
        ('drop_columns', drop_transformer),
        ('fill_na', fill_na_transformer),
        ('to_bin', to_bin_transformer),
    ])


def process_all_files(raw_folder: str = 'raw', processed_folder: str = 'processed'):
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    load_path = os.path.join(root_dir, 'data', raw_folder)
    save_path = os.path.join(root_dir, 'data', processed_folder)
    os.makedirs(save_path, exist_ok=True)

    all_files = [f for f in os.listdir(load_path) if f.endswith('.csv')]
    print(f"Found files: {all_files}")

    pipeline_data = preprocessing()
    train_df = val_df = test_df = None

    for filename in all_files:
        file_path = os.path.join(load_path, filename)
        data = pd.read_csv(file_path)
        if 'Attrition' in data.columns:
            data['Attrition'] = (data['Attrition'] == 'Yes').astype(int)

        if 'train' in filename:
            train_df = pipeline_data.fit_transform(data)
            train_df.to_csv(os.path.join(save_path, 'hr_train_processed.csv'), index=False)
        elif 'validation' in filename:
            val_df = pipeline_data.transform(data)
            val_df.to_csv(os.path.join(save_path, 'hr_val_processed.csv'), index=False)
        elif 'test' in filename:
            test_df = pipeline_data.transform(data)
            test_df.to_csv(os.path.join(save_path, 'hr_test_processed.csv'), index=False)

    return train_df, val_df, test_df