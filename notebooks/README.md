# notebooks/README.md

## Notebooks Overview

This folder contains the Jupyter notebooks that document the complete end-to-end workflow for the 10 Academy Week 5 & 6 Fraud Detection Challenge.

### Recommended Execution Order

1. **`eda-fraud-data.ipynb`**  
   Exploratory Data Analysis for the e-commerce dataset (`Fraud_Data.csv`). Covers data cleaning, distributions, relationships with the target, and class imbalance insights.

2. **`eda-creditcard.ipynb`**  
   Exploratory Data Analysis for the credit card dataset (`creditcard.csv`). Focuses on PCA features, transaction amounts, time patterns, and extreme imbalance.

3. **`feature-engineering.ipynb`**  
   Feature creation and preprocessing:  
   - IP address to country mapping  
   - Time-based features (`hour_of_day`, `day_of_week`, `time_since_signup`)  
   - Transaction velocity and frequency  
   - Numerical scaling and categorical encoding  
   - Class imbalance strategy (SMOTE example)

4. **`modeling.ipynb`**  
   Model building and evaluation:  
   - Stratified train-test split  
   - Logistic Regression baseline  
   - Ensemble model (Random Forest)  
   - Metrics: AUC-PR, F1-score, confusion matrices  
   - Cross-validation and model comparison

5. **`shap-explainability.ipynb`**  
   Model interpretability using SHAP:  
   - Global and local feature importance  
   - SHAP summary and force plots  
   - Analysis of true/false positives and negatives  
   - Business recommendations derived from insights

All notebooks are fully reproducible when the raw datasets are placed in `../data/raw/`. Visualizations and key findings are included in each notebook for clarity.