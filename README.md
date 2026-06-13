# Loan Approval Prediction - ANN

## Overview
This project implements an Artificial Neural Network (ANN) using TensorFlow/Keras to predict whether a loan application will be **Approved** or **Rejected**, based on applicant financial and personal details. This is a **binary classification** problem.

---

## Dataset
- **Source:** `loan_approval_dataset.csv`
- **Rows:** 4,269
- **Original Columns:** 13
- **Dropped Column:** `loan_id` (not predictive)
- **Final Input Features:** 11
- **Target Variable:** `loan_status`
  - `0` = Approved
  - `1` = Rejected

  *(Encoding determined automatically by `LabelEncoder`, which assigns labels alphabetically: Approved → 0, Rejected → 1)*

### Features Used

| Feature | Description |
|---|---|
| no_of_dependents | Number of dependents |
| education | Graduate / Not Graduate |
| self_employed | Yes / No |
| income_annum | Annual income |
| loan_amount | Loan amount requested |
| loan_term | Loan duration (years) |
| cibil_score | Applicant's credit score |
| residential_assets_value | Value of residential assets |
| commercial_assets_value | Value of commercial assets |
| luxury_assets_value | Value of luxury assets |
| bank_asset_value | Value of bank assets |

---

## Data Preprocessing
1. **Inspection:** Checked for missing values (`isnull().sum()` → none found) and duplicate rows (`duplicated().sum()` → none found).
2. **Cleaned column names:** Removed leading/trailing whitespace from column headers using `df.columns.str.strip()`.
3. **Cleaned categorical values:** Stripped whitespace from `education`, `self_employed`, and `loan_status` columns.
4. **Dropped** `loan_id`, as it carries no predictive information.
5. **Encoded Target Variable:** Used `LabelEncoder` on `loan_status` (Approved → 0, Rejected → 1).
6. **Defined Features and Target:**
   - `X` = all columns except `loan_status`
   - `y` = `loan_status`
7. **Identified Column Types:**
   - Numerical: `no_of_dependents`, `income_annum`, `loan_amount`, `loan_term`, `cibil_score`, `residential_assets_value`, `commercial_assets_value`, `luxury_assets_value`, `bank_asset_value`
   - Categorical: `education`, `self_employed`
8. **Built Preprocessing Pipelines** using `ColumnTransformer`:
   - **Numerical Pipeline:** `SimpleImputer(strategy='mean')` → `StandardScaler()`
   - **Categorical Pipeline:** `SimpleImputer(strategy='most_frequent')` → `OneHotEncoder(drop='first')`
9. **Train-Test Split:** 80% train / 20% test (`random_state=42`)
10. Applied `preprocessor.fit_transform()` on training data and `preprocessor.transform()` on test data.

---

## Handling Class Imbalance
The dataset has a moderate class imbalance — approximately 62% Approved vs 38% Rejected.

To ensure the model treats both classes fairly, class weights were computed using `compute_class_weight(class_weight='balanced')` and passed into `model.fit()`.
Class Weights: {0: 0.805, 1: 1.319}
---

## Model Architecture (ANN)
Input Layer (11 features after one-hot encoding)

↓

Dense(64, activation='relu')

↓

Dropout(0.3)

↓

Dense(32, activation='relu')

↓

Dense(1, activation='sigmoid')   →  Output probability (0–1)
- **ReLU** activations in the hidden layers allow the network to learn non-linear patterns.
- **Dropout(0.3)** randomly disables 30% of neurons during training to reduce overfitting.
- **Sigmoid** output activation produces a probability between 0 and 1, thresholded at 0.5 to obtain the final binary prediction.

---

## Training Configuration

| Setting | Value |
|---|---|
| Optimizer | Adam |
| Loss Function | Binary Crossentropy |
| Metric | Accuracy |
| Epochs | 20 |
| Batch Size | 32 |
| Validation Split | 10% |
| Class Weights | Applied (balanced) |

---

## Evaluation Results
precision    recall  f1-score   support

       0       0.97      0.94      0.96       536
       1       0.90      0.96      0.93       318

macro avg       0.94      0.95      0.94       854

weighted avg       0.95      0.94      0.95       854

accuracy                           0.94       854
| Metric | Class 0 (Approved) | Class 1 (Rejected) |
|---|---|---|
| Precision | 0.97 | 0.90 |
| Recall | 0.94 | 0.96 |
| F1-Score | 0.96 | 0.93 |

**Overall Accuracy: 94%**

---

## Observations
- The model performs strongly on both classes, with an overall accuracy of 94%.
- `cibil_score` is a strong driver of loan approval, consistent with real-world lending practices.
- Recall for the Rejected class (0.96) is slightly higher than for the Approved class (0.94), meaning the model is highly effective at correctly identifying applications that should be rejected.
- The relatively balanced class distribution (62/38) combined with class weighting resulted in stable performance across both classes without major trade-offs.
- **Potential Future Improvements:**
  - Feature importance analysis (e.g., SHAP) to better understand which factors drive loan decisions
  - Hyperparameter tuning (layer sizes, learning rate, dropout rate)
  - Threshold tuning for use cases where false approvals/rejections carry different costs
  - k-fold cross-validation for more robust performance estimates

---

## Tech Stack
- Python 3.13
- NumPy, Pandas
- Scikit-learn (preprocessing, pipelines, metrics)
- TensorFlow / Keras (ANN model)

---

## How to Run
1. Install dependencies:
```bash
   pip install numpy pandas scikit-learn tensorflow
```
2. Place `loan_approval_dataset.csv` in the project directory.
3. Open and run the notebook (`Notebook.ipynb`) cell by cell from top to bottom.

---

## Repository Structure
├── loan_approval_dataset.csv

├── Notebook.ipynb

└── README.md
## Author
**Muhammad Taha Sattar Arain**
AI & Data Science Student — SMIT (Batch 10)

- GitHub: [funterpie](https://github.com/funterpie)
- Portfolio: [tahatradz.online](https://tahatradz.online)
