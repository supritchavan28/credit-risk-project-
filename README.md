# Consumer Loan Default Prediction (Credit Risk Model)

A end-to-end credit risk project: given a loan applicant's financial profile,
predict their probability of default — the same core problem underwriting
and risk teams at banks (JPMorgan, Morgan Stanley, etc.) solve to price and
approve consumer loans.

**[Try the live scoring tool →](#)** *(paste your published artifact link here, or open `credit_risk_terminal.html` locally)*

## Why this project

Credit risk modeling sits at the intersection of finance and data science:
you need to understand *which financial signals actually predict default*
(debt-to-income, utilization, payment burden) and *why interpretability
matters* (regulators and risk committees need to explain declines — you
can't just ship a black box).

## Project structure

```
credit-risk-project/
├── 01_generate_data.py      # builds the synthetic loan dataset
├── 02_eda.py                 # exploratory analysis + charts
├── 03_train_model.py         # trains Logistic Regression + Random Forest
├── 04_evaluate.py            # ROC curves, confusion matrix, feature importance
├── credit_risk_terminal.html # interactive scoring tool (runs the model client-side)
├── data/
│   ├── loan_data.csv
│   └── test_predictions.csv
├── charts/                   # all generated PNGs
└── models/                   # trained model pipelines (.joblib)
```

## Dataset

10,000 synthetic loan records with realistic financial relationships
(built in `01_generate_data.py`), including:

| Feature | Description |
|---|---|
| `age`, `annual_income`, `employment_length_years` | Applicant profile |
| `credit_history_years`, `open_credit_lines`, `past_delinquencies_2yr` | Credit bureau signals |
| `credit_utilization` | % of available credit currently used |
| `loan_amount`, `loan_term_months`, `interest_rate`, `purpose` | Loan terms |
| `debt_to_income`, `payment_to_income` | Derived affordability ratios |
| `default` | **Target** — 1 if the loan defaulted |

Default rate: **~8%**, in line with real consumer loan portfolios.

> The data is synthetic so the project runs anywhere with no downloads or
> API keys. To use real data, swap in Kaggle's "Give Me Some Credit" or
> the Lending Club dataset — see the note at the bottom of
> `01_generate_data.py` for how to map columns.

## Approach

1. **EDA** (`02_eda.py`) — checked class balance, missingness, and how
   default rate moves across utilization, delinquency, and payment burden
   buckets. Confirmed the data behaves the way real credit data does before
   modeling anything.
2. **Feature engineering** — added `debt_to_income` and `payment_to_income`,
   the two ratios credit analysts actually underwrite against, rather than
   relying on raw loan amount and income separately.
3. **Modeling** (`03_train_model.py`) — trained two models on purpose:
   - **Logistic Regression** (with balanced class weights) as the
     interpretable baseline. Banks favor models like this for credit
     decisions because coefficients are auditable and regulators
     (e.g. under fair lending laws) require explainability.
   - **Random Forest** as a stronger non-linear comparison, to see how much
     performance is left on the table by choosing interpretability.
4. **Evaluation** (`04_evaluate.py`) — compared models on ROC-AUC rather
   than raw accuracy, since defaults are rare (~8%) and accuracy alone is
   misleading on imbalanced data.

## Results

| Model | ROC-AUC |
|---|---|
| Logistic Regression | **0.707** |
| Random Forest | 0.690 |

The simpler, interpretable model held its own against the more complex one
— a realistic and common finding in credit risk, which is part of why
logistic regression remains an industry standard for credit scoring
despite newer techniques being available.

**Top predictive features** (Random Forest importance): payment-to-income
ratio, credit utilization, and debt-to-income — all consistent with how
real underwriting models weight risk.

See `/charts` for the full ROC curve, confusion matrix, and feature
importance plots, plus EDA charts showing default rate by utilization,
delinquency count, and payment burden.

## Interactive scoring tool

`credit_risk_terminal.html` runs the trained Logistic Regression model
entirely client-side (the fitted coefficients and scaler are embedded in
the page) — enter an applicant's profile and get a live default
probability plus the top risk drivers for that specific applicant. No
backend, no data leaves the browser.

## How to talk about this project in an interview

- **The interpretability trade-off**: be ready to explain *why* you
  compared a linear model to a non-linear one, and why the linear model's
  competitiveness is actually a meaningful finding, not a disappointment.
- **Feature engineering choices**: debt-to-income and payment-to-income
  aren't in the raw data — you derived them, the way a real credit analyst
  would, because raw income/loan-amount don't capture affordability well
  on their own.
- **Why ROC-AUC, not accuracy**: with an 8% default rate, a model that
  predicts "never defaults" gets 92% accuracy while being useless. AUC
  measures ranking ability regardless of class imbalance.
- **What you'd do with real data / more time**: calibrate probabilities
  (Platt scaling / isotonic regression), test for disparate impact across
  protected classes (fair lending compliance), and add macroeconomic
  features (unemployment rate, regional housing prices).

## Running it yourself

```bash
pip install pandas numpy scikit-learn matplotlib joblib
python 01_generate_data.py
python 02_eda.py
python 03_train_model.py
python 04_evaluate.py
```

Then open `credit_risk_terminal.html` in any browser.
