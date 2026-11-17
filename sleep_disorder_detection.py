# Sleep Health and Sleep Disorder Detection
# Generated from Google Colab

import os
import glob
import kagglehub
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# 1. LOAD DATASET
path = kagglehub.dataset_download(
    "siamaktahmasbi/insights-into-sleep-patterns-and-daily-habits"
)
csv_files = glob.glob(os.path.join(path, "*.csv"))
df = pd.read_csv(csv_files[0])
print("Dataset Shape:", df.shape)

# 2. DATA CLEANING
df.columns = df.columns.str.strip()
if "Person ID" in df.columns:
    df = df.drop(columns=["Person ID"])

bp_col = next((c for c in df.columns if "blood pressure" in c.lower()), None)
if bp_col:
    bp = df[bp_col].astype(str).str.split("/", expand=True)
    df["Systolic_BP"] = pd.to_numeric(bp[0], errors="coerce")
    df["Diastolic_BP"] = pd.to_numeric(bp[1], errors="coerce")
    df = df.drop(columns=[bp_col])

target_col = next((c for c in df.columns if "disorder" in c.lower()), None)
df[target_col] = df[target_col].fillna("None")

for col in df.columns:
    if "bmi" in col.lower():
        df[col] = df[col].replace("Normal Weight", "Normal")

numeric_cols = df.select_dtypes(include=np.number).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# 3. PREPARE FEATURES AND TARGET
X = df.drop(columns=[target_col])
y = df[target_col]
X = pd.get_dummies(X, drop_first=True).astype(float)
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# 4. SPLIT AND SCALE
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. MODELS AND EVALUATION
models = {
    "Logistic Regression": (LogisticRegression(max_iter=1000, random_state=42), X_train_scaled, X_test_scaled),
    "SVM": (SVC(kernel="rbf", probability=True, random_state=42), X_train_scaled, X_test_scaled),
    "Random Forest": (RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42), X_train, X_test),
    "XGBoost": (XGBClassifier(eval_metric="mlogloss", random_state=42), X_train, X_test)
}

for name, (model, train_data, test_data) in models.items():
    model.fit(train_data, y_train)
    y_pred = model.predict(test_data)
    print(f"\n{name} Results:")
    print(classification_report(y_test, y_pred, target_names=encoder.classes_, zero_division=0))
