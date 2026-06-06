import pandas as pd
import mlflow
from ml_model import train_model
from predict import predict
from save_model import save_model
from dotenv import load_dotenv
import os

load_dotenv()

def setup_mlflow():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("hr-attrition-exp")

def main():
    setup_mlflow()
    model_name = os.getenv("MODEL_NAME", "LogisticRegression")
    print(f"Selected model: {model_name}")
    
    # Загрузка train/test из raw (уже должны быть созданы load_data.ipynb)
    train = pd.read_csv('data/raw/raw_eda_hr_train.csv')
    test = pd.read_csv('data/raw/raw_eda_hr_test.csv')
    
    # Убедимся, что Attrition в нужном формате
    train['Attrition'] = train['Attrition'].astype(int)
    test['Attrition'] = test['Attrition'].astype(int)
    
    print(f"Train positive: {train['Attrition'].sum()}, Test positive: {test['Attrition'].sum()}")
    
    model = train_model(train)
    predict(model, test)
    save_model(model, name=model_name)
    print("Pipeline finished successfully.")

if __name__ == "__main__":
    main()