"""
02_eda.py
---------
Exploratory Data Analysis on the loan dataset. Produces summary stats
and saves charts as PNGs for the README / report.
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

df = pd.read_csv("/home/claude/loan_data.csv")

print("=== Shape ===")
print(df.shape)

print("\n=== Missing values ===")
print(df.isnull().sum().sum(), "total missing values")

print("\n=== Default rate ===")
print(df["default"].value_counts(normalize=True))

print("\n=== Numeric summary ===")
print(df.describe().T)

# --- Chart 1: Default rate by credit utilization bucket ---
df["utilization_bucket"] = pd.cut(
    df["credit_utilization"], bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0]
)
util_default = df.groupby("utilization_bucket", observed=True)["default"].mean()

fig, ax = plt.subplots(figsize=(7, 4.5))
util_default.plot(kind="bar", ax=ax, color="#2c5f8a")
ax.set_title("Default Rate by Credit Utilization")
ax.set_xlabel("Credit Utilization Bucket")
ax.set_ylabel("Default Rate")
ax.set_ylim(0, max(util_default) * 1.3)
for i, v in enumerate(util_default):
    ax.text(i, v + 0.003, f"{v:.1%}", ha="center", fontsize=9)
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("/home/claude/chart_utilization_vs_default.png", dpi=140)
plt.close()

# --- Chart 2: Default rate by past delinquencies ---
delinq_default = df.groupby("past_delinquencies_2yr")["default"].mean()
delinq_counts = df.groupby("past_delinquencies_2yr")["default"].count()
delinq_default = delinq_default[delinq_counts >= 30]  # filter noisy small buckets

fig, ax = plt.subplots(figsize=(7, 4.5))
delinq_default.plot(kind="bar", ax=ax, color="#a83232")
ax.set_title("Default Rate by Past Delinquencies (last 2 years)")
ax.set_xlabel("Number of Past Delinquencies")
ax.set_ylabel("Default Rate")
for i, v in enumerate(delinq_default):
    ax.text(i, v + 0.003, f"{v:.1%}", ha="center", fontsize=9)
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("/home/claude/chart_delinquencies_vs_default.png", dpi=140)
plt.close()

# --- Chart 3: Payment-to-income vs default ---
df["pti_bucket"] = pd.cut(df["payment_to_income"], bins=[0, 0.1, 0.2, 0.3, 0.5, 3])
pti_default = df.groupby("pti_bucket", observed=True)["default"].mean()

fig, ax = plt.subplots(figsize=(7, 4.5))
pti_default.plot(kind="bar", ax=ax, color="#3a8a5f")
ax.set_title("Default Rate by Payment-to-Income Ratio")
ax.set_xlabel("Payment-to-Income Bucket")
ax.set_ylabel("Default Rate")
for i, v in enumerate(pti_default):
    ax.text(i, v + 0.005, f"{v:.1%}", ha="center", fontsize=9)
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("/home/claude/chart_pti_vs_default.png", dpi=140)
plt.close()

print("\nSaved 3 charts to /home/claude/")
