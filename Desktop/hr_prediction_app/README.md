# HR Attrition Prediction

## Business Problem
Predict employee attrition to enable proactive retention measures.

## Model
- Logistic Regression with SMOTE and class_weight='balanced'
- Threshold = 0.66
- Final metrics: Recall=0.53, Precision=0.50, F1=0.5155

## Pipeline
1. Load raw data
2. Feature engineering (drop constant columns, create ratios, binary indicators)
3. Train/Test split (80/20)
4. Preprocessing: StandardScaler + OneHotEncoder
5. SMOTE oversampling
6. Train LogisticRegression (C=0.1)
7. Evaluate on test set with threshold 0.66
8. Save model and metrics

## Run
```bash
python src/main_pipeline.py