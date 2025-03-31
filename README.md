## Project Overview
This project implements a **fraud detection system** using **XGBoost** and **SHAP (SHapley Additive exPlanations)** to improve model interpretability. The dataset consists of anonymized credit card transactions and presents a highly imbalanced classification problem, where fraudulent transactions account for only **0.172%** of the total data.

## Objectives
- **Detect fraudulent transactions** with high precision and recall.
- **Handle class imbalance** using **SMOTE (Synthetic Minority Over-sampling Technique)**.
- **Improve model interpretability** with SHAP to understand feature impact.
- **Optimize model performance** using hyperparameter tuning.

## Dataset
- Transactions made by credit cards in September 2013 by European cardholders.
- Contains **284,807 transactions**, including **492 fraudulent cases**.
- **Features**:
  - `V1` to `V28`: Principal Component Analysis (PCA) transformed features.
  - `Time`: Seconds elapsed between transactions.
  - `Amount`: Transaction amount.
  - `Class`: **Target variable** (1 = Fraud, 0 = Legitimate).

**Dataset Source**: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)

## Technologies & Libraries
- **Python** (Jupyter Notebook)
- **XGBoost** for classification
- **SMOTE** for handling class imbalance
- **SHAP** for explainability
- **Scikit-Learn**, **Pandas**, **NumPy**

## Exploratory Data Analysis (EDA)
- Checked for missing values and data imbalances.
- Plotted fraud vs. non-fraud transaction distributions.
- Analyzed feature correlations to identify important predictors.

## ⚡ Model Training Pipeline
1. **Data Preprocessing**:
   - Removed NaN values.
   - Applied **SMOTE** to balance fraudulent cases.
2. **Feature Engineering**:
   - Created interaction features between top SHAP features.
   - Normalized `Amount` and `Time` features.
3. **Model Selection & Training**:
   - Used **XGBoost Classifier** with optimized hyperparameters.
   - Performed **GridSearchCV** for tuning.
4. **Model Evaluation**:
   - Used **Precision-Recall AUC** due to class imbalance.
   - Evaluated using **F1-score, Precision, Recall, and Confusion Matrix**.

## Results & Insights
### **Model Performance**
| Metric                 | Score |
|------------------------|-------|
| Precision (Fraud)      | 0.74  |
| Recall (Fraud)         | 0.93  |
| F1-Score (Fraud)       | 0.82  |
| Precision-Recall AUC   | 0.965 |


## Key Takeaways
- **Class imbalance handling is crucial**: Using **SMOTE** significantly improved recall.
- **Explainability matters**: SHAP helped identify the most influential features.
- **Feature 14 & Feature 4 were critical**: These should be considered for business-level fraud detection policies.


