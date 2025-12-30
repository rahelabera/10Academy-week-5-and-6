# 10 Academy Week 5 & 6 Challenge: Improved Detection of Fraud Cases for E-commerce and Bank Transactions

## Project Overview

This project develops robust machine learning models to detect fraudulent transactions in two highly imbalanced datasets:

- **E-commerce transactions** (`Fraud_Data.csv` + `IpAddress_to_Country.csv`)
- **Credit card transactions** (`creditcard.csv`)

The work focuses on:
- Advanced feature engineering (geolocation mapping, time-based features, transaction velocity)
- Handling extreme class imbalance
- Model training and evaluation with business-relevant metrics (AUC-PR, F1-score, confusion matrix)
- Model interpretability using SHAP for actionable insights

The ultimate goal is to reduce financial losses from missed fraud (false negatives) while minimising customer friction from incorrect flags (false positives).

### Repository Structure

```plaintext
fraud-detection/
├── .vscode/
│   └── settings.json
├── .github/
│   └── workflows/
│       └── unittests.yml
├── data/                           # Add this folder to .gitignore
│   ├── raw/                        # Original datasets
│   └── processed/                  # Cleaned and feature-engineered data
├── notebooks/
│   ├── __init__.py
│   ├── eda-fraud-data.ipynb
│   ├── eda-creditcard.ipynb
│   ├── feature-engineering.ipynb
│   ├── modeling.ipynb
│   ├── shap-explainability.ipynb
│   └── README.md
├── src/
│   ├── __init__.py
├── tests/
│   ├── __init__.py
├── models/                         # Saved model artifacts
├── scripts/
│   ├── __init__.py
│   └── README.md
├── requirements.txt
├── README.md
└── .gitignore
## Setup & Installation

1. Clone the repository:
   git clone https://github.com/rahelabera/fraud-detection.git
   cd fraud-detection

2. Create and activate a virtual environment (recommended):Bashpython -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
3. Install dependencies:Bashpip install -r requirements.txt
4. Place the raw datasets in data/raw/:
Fraud_Data.csv
IpAddress_to_Country.csv
creditcard.csv


## Project Workflow
The analysis follows a systematic, reproducible pipeline documented in Jupyter notebooks:

- Exploratory Data Analysis
eda-fraud-data.ipynb: Univariate/bivariate analysis, class imbalance, initial insights for e-commerce data
eda-creditcard.ipynb: Analysis of PCA features, transaction amounts, extreme imbalance

- Feature Engineering (feature-engineering.ipynb)
IP address → country mapping (range-based merge)
Time features: hour_of_day, day_of_week, time_since_signup
Velocity & frequency: transactions per user/device in time windows
Scaling (StandardScaler) and one-hot encoding
Strategy planning for class imbalance (SMOTE on training data only)

- Modeling (modeling.ipynb)
Stratified train-test split
Logistic Regression baseline (interpretable)
Ensemble model: Random Forest (with basic hyperparameter tuning)
Evaluation: AUC-PR, F1-score, confusion matrices
Cross-validation (Stratified K-Fold)
Model comparison and final selection

- Model Explainability (shap-explainability.ipynb)
Built-in feature importance vs. SHAP summary plot
SHAP force plots for individual predictions (true positive, false positive, false negative)
Top 5 drivers of fraud identified
3+ actionable business recommendations grounded in SHAP insights


All notebooks are fully executed, well-commented, and include key visualizations.
Key Results & Insights

Best performing model: Random Forest (superior AUC-PR and F1 on both datasets)
Top fraud indicators (from SHAP):
Very short time_since_signup
High transaction velocity
Transactions from high-risk countries
Unusual purchase values
Specific hours/days of activity

Business recommendations include additional verification for rapid post-signup purchases and country-based risk scoring.

### Requirements
Full list in requirements.txt. Core packages:
- textpandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- imbalanced-learn
- shap
- joblib
