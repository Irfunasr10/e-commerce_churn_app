# E-Commerce Platform Churn Prediction

## Overview

This project uses machine learning to predict whether an e-commerce customer is likely to churn. The notebook performs data cleaning, exploratory data analysis, feature encoding, feature scaling, model training, hyperparameter tuning, and class-imbalance handling using SMOTE.

The main classifier used is Random Forest.

## Dataset

The notebook uses:

E Commerce Dataset.xlsx

The dataset contains 5,630 customer records and 20 original columns, including:

- CustomerID
- Churn
- Tenure
- PreferredLoginDevice
- CityTier
- WarehouseToHome
- PreferredPaymentMode
- Gender
- HourSpendOnApp
- NumberOfDeviceRegistered
- PreferedOrderCat
- SatisfactionScore
- MaritalStatus
- NumberOfAddress
- Complain
- OrderAmountHikeFromlastYear
- CouponUsed
- OrderCount
- DaySinceLastOrder
- CashbackAmount

The target variable is Churn:

- 1 = Customer churned
- 0 = Customer did not churn

## Project Workflow

1. Data Loading
2. Data Inspection
3. Data Cleaning
4. Exploratory Data Analysis
5. Feature Engineering
6. Feature Scaling
7. Train-Test Split
8. Random Forest Classification
9. Class Imbalance Handling using SMOTE
10. Hyperparameter Tuning using GridSearchCV
11. Final Model Evaluation

## Data Cleaning

The preprocessing includes:

- Removing duplicate rows
- Dropping CustomerID
- Standardizing login-device values
- Standardizing payment modes
- Renaming selected columns
- Standardizing order-category values

No missing values were found in the original dataset.

## Exploratory Data Analysis

The notebook uses:

- Histograms with KDE curves
- Count plots
- Pearson correlation
- Correlation heatmap

The analysis examines variables such as:

- Tenure
- Warehouse-to-home distance
- App usage
- Order count
- Cashback amount
- Satisfaction score
- Complaints
- Device usage

Tenure has a notable negative correlation with churn (-0.345).

## Feature Engineering

Binary categorical variables are converted into numerical form.

Examples:

- Mobile Phone → 1
- Computer → 0
- Male → 1
- Female → 0

These are represented as:

- Is_mobile_phone
- Is_male

Other categorical variables are converted using One-Hot Encoding.

These include:

- Payment mode
- Order category
- Marital status
- City tier

## Feature Scaling

Numerical features are standardized using StandardScaler.

Features include:

- Tenure
- WarehouseToHome
- HourSpendOnApp
- OrderAmountHikeFromlastYear
- CouponUsed
- OrderCount
- DaySinceLastOrder
- CashbackAmount
- NumberOfDeviceRegistered
- SatisfactionScore
- NumberOfAddress

The scaler is fitted on the training data and then applied to the test data.

## Train-Test Split

The dataset is divided into:

- 80% training data
- 20% testing data

random_state = 42

The test set contains 1,126 records.

## Machine Learning Model

### Random Forest Classifier

The initial Random Forest model uses:

- n_estimators = 100
- random_state = 42

The model is evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

### Initial Model Results

Training Accuracy: 98.83%

Test Accuracy: 96.89%

Churn Precision: 0.99

Churn Recall: 0.82

Churn F1-score: 0.90

Confusion Matrix:

[[940   1]
 [ 34 151]]

## Handling Class Imbalance

The dataset contains significantly fewer churned customers than non-churned customers.

Before SMOTE:

Class 0: 3741
Class 1: 763

SMOTE (Synthetic Minority Over-sampling Technique) is applied to the training data.

After SMOTE:

Class 0: 3741
Class 1: 3741

SMOTE is applied only to the training data. The original test set is retained for final evaluation.

## Hyperparameter Tuning

GridSearchCV with 5-fold cross-validation is used to tune the Random Forest model.

Parameters searched:

- n_estimators: 100, 200
- max_depth: 10, 20, None
- min_samples_split: 2, 5
- min_samples_leaf: 1, 2

Recall is used as the scoring metric because correctly identifying customers who may churn is an important objective.

Best parameters:

n_estimators = 200
max_depth = None
min_samples_split = 2
min_samples_leaf = 1

Best cross-validation recall:

0.9701

## Final Model Results

The tuned Random Forest trained using the SMOTE-balanced training data achieved:

Accuracy: 96.36%

Churn Precision: 0.94

Churn Recall: 0.83

Churn F1-score: 0.88

Confusion Matrix:

[[931  10]
 [ 31 154]]

Classification Report:

              precision    recall  f1-score   support

0                 0.97      0.99      0.98       941
1                 0.94      0.83      0.88       185

accuracy                              0.96      1126
macro avg          0.95      0.91      0.93      1126
weighted avg       0.96      0.96      0.96      1126

## Key Takeaway

The project demonstrates an end-to-end customer churn prediction workflow using machine learning.

The Random Forest model provides strong overall predictive performance, while SMOTE is used to address the imbalance between churned and non-churned customers.

The final model correctly identifies 154 of the 185 churn cases in the test set, giving a churn recall of approximately 83%.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- imbalanced-learn
- Jupyter Notebook
- Excel

## Required Libraries

Install the required packages using:

pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn openpyxl jupyter

## How to Run

1. Place E Commerce Dataset.xlsx in the same directory as the notebook.
2. Open E_Commerce_platform_churn_prediction.ipynb in Jupyter Notebook or JupyterLab.
3. Install the required Python libraries.
4. Run the notebook cells from top to bottom.

## Project Structure

E_Commerce_platform_churn_prediction/
│
├── E_Commerce_platform_churn_prediction.ipynb
├── E Commerce Dataset.xlsx
└── README.md

## Notes

- The notebook expects the Excel dataset to be named exactly E Commerce Dataset.xlsx.
- random_state = 42 is used for reproducibility.
- SMOTE is applied only to the training data.
- Final evaluation is performed on the original test set.
- The main machine learning model used is Random Forest.
