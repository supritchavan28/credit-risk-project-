"""
01_generate_data.py
--------------------
Generates a synthetic but realistic consumer loan dataset for a credit
risk / probability-of-default modeling project.

Why synthetic data?
- Runs anywhere with zero setup / no downloads / no API keys.
- We control the underlying relationships, so we KNOW what a good model
  should discover (useful for sanity-checking your pipeline).

To use REAL data instead, see the note at the bottom of this file for
how to swap in the Kaggle "Give Me Some Credit" or "Lending Club" datasets
with the same column names, so the rest of the pipeline just works.
"""

import numpy as np
import pandas as pd

np.random.seed(42)

N = 10000  # number of loan applicants

# ---- Base demographic / financial features -----------------------------
age = np.random.normal(40, 12, N).clip(18, 75).round(0)

annual_income = np.random.lognormal(mean=10.9, sigma=0.5, size=N).clip(15000, 400000).round(0)

employment_length = np.random.exponential(scale=6, size=N).clip(0, 40).round(1)

# Credit history length (years)
credit_history_years = (age - 18 - np.random.exponential(3, N)).clip(0, None).round(1)

# Number of open credit lines
open_credit_lines = np.random.poisson(5, N).clip(0, 20)

# Number of past delinquencies (30+ days late) in last 2 years
past_delinquencies = np.random.poisson(0.4, N).clip(0, 10)

# Credit utilization ratio (balance / total credit limit)
credit_utilization = np.random.beta(2, 5, N).clip(0, 1)

# Loan amount requested
loan_amount = np.random.lognormal(mean=9.3, sigma=0.6, size=N).clip(1000, 100000).round(0)

# Loan term in months
loan_term = np.random.choice([12, 24, 36, 48, 60], N, p=[0.1, 0.2, 0.35, 0.2, 0.15])

# Interest rate offered (correlated with risk factors, added after)
base_rate = np.random.normal(9, 2, N).clip(4, 25)

# Home ownership status
home_ownership = np.random.choice(
    ["RENT", "MORTGAGE", "OWN"], N, p=[0.4, 0.45, 0.15]
)

# Purpose of loan
purpose = np.random.choice(
    ["debt_consolidation", "credit_card", "home_improvement",
     "major_purchase", "small_business", "medical", "other"],
    N, p=[0.35, 0.2, 0.12, 0.1, 0.08, 0.07, 0.08]
)

# ---- Derived financial ratios (features a real credit analyst would use) ---
debt_to_income = (loan_amount / loan_term * 12) / annual_income  # rough proxy
debt_to_income = debt_to_income.clip(0, 2)

monthly_income = annual_income / 12
installment = loan_amount * (base_rate / 1200) / (1 - (1 + base_rate / 1200) ** (-loan_term))
payment_to_income = (installment / monthly_income).clip(0, 3)

# ---- Build a "true" default probability from a logistic combination -------
# This is the ground truth relationship the model will try to learn.
z = (
    -3.0
    + 2.2 * credit_utilization
    + 0.55 * past_delinquencies
    + 2.0 * payment_to_income
    + 1.3 * debt_to_income
    - 0.03 * credit_history_years
    - 0.015 * employment_length
    - 0.00002 * annual_income / 1000 * 10  # slight negative effect, income in $10ks
    + 0.10 * (home_ownership == "RENT")
    + 0.15 * (purpose == "small_business")
    - 0.10 * (open_credit_lines.clip(0, 8))  # having some credit lines is fine, extreme not modeled here
    + np.random.normal(0, 0.6, N)  # noise
)
prob_default = 1 / (1 + np.exp(-z))
default = np.random.binomial(1, prob_default)

df = pd.DataFrame({
    "age": age.astype(int),
    "annual_income": annual_income,
    "employment_length_years": employment_length,
    "home_ownership": home_ownership,
    "credit_history_years": credit_history_years,
    "open_credit_lines": open_credit_lines,
    "past_delinquencies_2yr": past_delinquencies,
    "credit_utilization": credit_utilization.round(3),
    "loan_amount": loan_amount,
    "loan_term_months": loan_term,
    "interest_rate": base_rate.round(2),
    "purpose": purpose,
    "debt_to_income": debt_to_income.round(3),
    "payment_to_income": payment_to_income.round(3),
    "default": default,  # target: 1 = defaulted, 0 = repaid
})

df.to_csv("/home/claude/loan_data.csv", index=False)

print(f"Generated {len(df)} loan records.")
print(f"Default rate: {df['default'].mean():.2%}")
print(df.head())

# ---------------------------------------------------------------------------
# TO USE REAL DATA INSTEAD:
# Download "Give Me Some Credit" (Kaggle) or the Lending Club dataset,
# then rename/derive columns to match the schema above (same column names),
# save as loan_data.csv in this folder, and skip this script. Everything
# downstream (EDA, feature engineering, modeling) will work unchanged as
# long as a binary "default" column exists.
# ---------------------------------------------------------------------------
