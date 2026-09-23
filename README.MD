# FinTech Innovations — Automated Loan-Approval Risk Model

## Project Overview
FinTech Innovations evaluates consumer credit applications using manual underwriter review. This process is slow, inconsistent across reviewers, and opaque for compliance. This project builds a reproducible binary classification pipeline to predict loan default risk using only data available at application time across 20,000 historical records.

## Key Findings and Financial Impact
- **Asymmetric Costs:** False Negative (rejecting good applicant) = $8,000 lost profit. False Positive (approving defaulter) = $50,000 loss. Cost ratio 1:6.25, so accuracy alone is misleading.
- **Drivers:** After removing leakage features, MonthlyIncome and TotalDebtToIncomeRatio are the top predictors, aligning with affordability fundamentals.
- **Baseline vs Model:** Blind approval on validation set would incur significant default exposure. Optimized model reduces false approvals and minimizes total portfolio drag, balancing risk and throughput.

## Dataset Profile
- **Dimensions:** 20,000 rows, 35 initial attributes
- **Target:** LoanApproved ( ~65% Denied, ~35% Approved)
- **Missing Values:** MaritalStatus (1331), EducationLevel (901), SavingsAccountBalance (572)
- **Cleaning:** AnnualIncome stripped of $ and commas and cast to float. Leakage features removed prior to modeling: RiskScore, RiskScore2, InterestRate, InterestRate2.

## Pipeline Architecture
Unified `ColumnTransformer` inside 80/20 split to prevent leakage:
- **Numerical:** Median imputation + StandardScaler
- **Categorical:** Constant "Unknown" imputation + OneHotEncoder
- **Ordinal:** Most frequent imputation + Ordered encoding (High School < Associate < Bachelor < Master < Doctorate)

## Model Evaluation
Five algorithms evaluated on isolated 20% test set using cross-validated pipelines.

| Model | Accuracy | F1-Score | ROC-AUC |
| :--- | :--- | :--- | :--- |
| Logistic Regression | 0.93 | 0.85 | 0.97 |
| Decision Tree | 0.91 | 0.83 | 0.90 |
| Random Forest | 0.94 | 0.87 | 0.97 |
| Gradient Boosting | 0.93 | 0.86 | 0.97 |
| XGBoost | 0.93 | 0.86 | 0.97 |

## Hyperparameter Tuning
5-fold GridSearchCV on XGBoost targeting ROC-AUC.

**Best Params:**
- `prep__num__imputer__strategy`: median
- `model__learning_rate`: 0.1
- `model__max_depth`: 3
- `model__n_estimators`: 300

Shallow depth (3) with 300 estimators provided regularization, preventing overfit while maintaining performance. Best CV ROC-AUC: 0.9784.

## Compliance & Fairness
Segmentation audit across EducationLevel showed consistent False Positive Rates across groups, indicating no systematic over-approval bias.

## Future Work
- Transition to XGBoost native missing value handling
- Implement PSI drift monitoring for production data