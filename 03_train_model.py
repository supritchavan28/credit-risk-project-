"""
03_train_model.py
------------------
Trains two models on the loan dataset:
  1. Logistic Regression  - the industry-standard, interpretable baseline
     that credit risk teams favor (regulators like explainability).
  2. Random Forest        - a stronger, non-linear model for comparison.

Saves both models + the fitted preprocessing pipeline to disk (joblib)
so 04_evaluate.py and the scoring app can reuse them without retraining.
"""

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report

df = pd.read_csv("/home/claude/loan_data.csv")

TARGET = "default"
NUMERIC_FEATURES = [
    "age", "annual_income", "employment_length_years", "credit_history_years",
    "open_credit_lines", "past_delinquencies_2yr", "credit_utilization",
    "loan_amount", "loan_term_months", "interest_rate",
    "debt_to_income", "payment_to_income",
]
CATEGORICAL_FEATURES = ["home_ownership", "purpose"]

X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), NUMERIC_FEATURES),
    ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
])

# ---- Model 1: Logistic Regression ----
logreg_pipe = Pipeline([
    ("prep", preprocessor),
    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
])
logreg_pipe.fit(X_train, y_train)
logreg_probs = logreg_pipe.predict_proba(X_test)[:, 1]
logreg_auc = roc_auc_score(y_test, logreg_probs)

# ---- Model 2: Random Forest ----
rf_pipe = Pipeline([
    ("prep", preprocessor),
    ("clf", RandomForestClassifier(
        n_estimators=300, max_depth=6, min_samples_leaf=20,
        class_weight="balanced", random_state=42, n_jobs=-1
    )),
])
rf_pipe.fit(X_train, y_train)
rf_probs = rf_pipe.predict_proba(X_test)[:, 1]
rf_auc = roc_auc_score(y_test, rf_probs)

print(f"Logistic Regression ROC-AUC: {logreg_auc:.4f}")
print(f"Random Forest ROC-AUC:       {rf_auc:.4f}")

print("\n--- Logistic Regression classification report (threshold=0.5) ---")
print(classification_report(y_test, (logreg_probs >= 0.5).astype(int)))

print("\n--- Random Forest classification report (threshold=0.5) ---")
print(classification_report(y_test, (rf_probs >= 0.5).astype(int)))

# Save everything needed downstream
joblib.dump(logreg_pipe, "/home/claude/model_logreg.joblib")
joblib.dump(rf_pipe, "/home/claude/model_rf.joblib")
X_test.assign(default=y_test, logreg_prob=logreg_probs, rf_prob=rf_probs).to_csv(
    "/home/claude/test_predictions.csv", index=False
)

print("\nSaved models and test predictions.")
