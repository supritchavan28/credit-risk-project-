"""
04_evaluate.py
--------------
Generates evaluation visuals: ROC curves for both models, a confusion
matrix, and a feature importance chart (from the Random Forest and the
Logistic Regression coefficients).
"""

import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay

test_df = pd.read_csv("/home/claude/test_predictions.csv")
y_test = test_df["default"]

logreg_pipe = joblib.load("/home/claude/model_logreg.joblib")
rf_pipe = joblib.load("/home/claude/model_rf.joblib")

# ---- ROC Curve ----
fpr_lr, tpr_lr, _ = roc_curve(y_test, test_df["logreg_prob"])
fpr_rf, tpr_rf, _ = roc_curve(y_test, test_df["rf_prob"])
auc_lr = auc(fpr_lr, tpr_lr)
auc_rf = auc(fpr_rf, tpr_rf)

fig, ax = plt.subplots(figsize=(6.5, 6))
ax.plot(fpr_lr, tpr_lr, label=f"Logistic Regression (AUC={auc_lr:.3f})", color="#2c5f8a", lw=2)
ax.plot(fpr_rf, tpr_rf, label=f"Random Forest (AUC={auc_rf:.3f})", color="#a83232", lw=2)
ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random guess")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve: Default Prediction Models")
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig("/home/claude/chart_roc_curve.png", dpi=140)
plt.close()

# ---- Confusion Matrix (Random Forest, threshold=0.5) ----
rf_preds = (test_df["rf_prob"] >= 0.5).astype(int)
cm = confusion_matrix(y_test, rf_preds)
fig, ax = plt.subplots(figsize=(5.5, 5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Default", "Default"])
disp.plot(ax=ax, cmap="Blues", colorbar=False)
ax.set_title("Random Forest Confusion Matrix (threshold=0.5)")
plt.tight_layout()
plt.savefig("/home/claude/chart_confusion_matrix.png", dpi=140)
plt.close()

# ---- Feature Importance (Random Forest) ----
rf_clf = rf_pipe.named_steps["clf"]
preprocessor = rf_pipe.named_steps["prep"]
feature_names = preprocessor.get_feature_names_out()
importances = rf_clf.feature_importances_

imp_df = pd.DataFrame({"feature": feature_names, "importance": importances})
imp_df = imp_df.sort_values("importance", ascending=True).tail(12)

fig, ax = plt.subplots(figsize=(7.5, 6))
ax.barh(imp_df["feature"], imp_df["importance"], color="#3a8a5f")
ax.set_title("Top Feature Importances (Random Forest)")
ax.set_xlabel("Importance")
plt.tight_layout()
plt.savefig("/home/claude/chart_feature_importance.png", dpi=140)
plt.close()

print("Saved: chart_roc_curve.png, chart_confusion_matrix.png, chart_feature_importance.png")
print(f"\nFinal AUC comparison -> Logistic Regression: {auc_lr:.4f} | Random Forest: {auc_rf:.4f}")
print("\nTop 5 most important features (Random Forest):")
print(imp_df.tail(5)[::-1].to_string(index=False))
