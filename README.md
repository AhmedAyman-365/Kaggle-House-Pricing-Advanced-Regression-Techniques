# Kaggle-House-Pricing-Advanced-Regression-Techniques

## Ames Housing Price Prediction
# Overview
This project uses machine learning to predict the final sale price of residential homes in Ames, Iowa. The dataset, provided by Kaggle, contains 79 explanatory variables describing various aspects of residential homes.

The goal is to build a robust regression model that achieves a low Root Mean Squared Error (RMSE) between the logarithm of the predicted value and the logarithm of the observed sale price.

## Workflow & Methodology
This notebook follows a rigorous data science workflow to ensure reproducible and accurate results. The process is broken down into four key stages.

# 1. Data Loading and Initial Setup
Loaded training and test datasets.

Saved the 'Id' column from the test set for the final submission.

Separated the target variable ('SalePrice') from the training features.

Target Transformation: Applied a log-transformation (np.log1p) to the target variable ('SalePrice'). This is crucial because house prices are highly right-skewed, and the evaluation metric is based on log-error.

# 2. Data Cleaning & Imputation
A critical step is handling missing values correctly based on domain knowledge.

Domain-Specific Imputation: For features like PoolQC, GarageType, and BsmtQual, a missing value (NaN) does not mean "unknown," but rather that the feature does not exist (e.g., "No Pool"). These were explicitly filled with the string "None".

Data Splitting: The training data was split into a training set (X_trn) and a validation set (X_val) to evaluate model performance before proceeding to the test set.

Statistical Imputation: To avoid data leakage, imputation statistics (modes for categorical data, medians for numerical data) were calculated solely on the training split (X_trn). These statistics were then used to fill remaining missing values in the training, validation, and final test sets.

# 3. Feature Engineering & Scaling
Raw data was transformed into a format suitable for a linear machine learning model.

One-Hot Encoding: Categorical features were converted into numerical format using pd.get_dummies.

Column Alignment: The validation and test sets were aligned to the training set to ensure all datasets contain the exact same columns (features) after encoding.

Feature Scaling: A StandardScaler was fitted on the encoded training data (X_trn_encoded) and then used to transform the validation and test sets. This ensures all features are on the same scale (mean=0, variance=1), which is essential for linear models.

# 4. Model Training & Prediction
Model Selection: A Linear Regression model was chosen as a baseline.

Training: The model was trained on the processed and scaled training data (X_trn_scaled) and the log-transformed target (y_trn).

Validation: The model's performance was evaluated on the validation set (X_val_scaled) using Root Mean Squared Error (RMSE).

Final Prediction: The trained model generated predictions for the processed test set (test_scaled).

Inverse Transformation: These predictions (which were in log-scale) were converted back to actual dollar values using np.expm1.

Submission Generation: A final CSV file was created containing the test IDs and the predicted sale prices, ready for Kaggle submission.
