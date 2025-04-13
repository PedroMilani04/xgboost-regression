# Time Series Imputation & Forecast with XGBoost + Optuna

A machine learning pipeline for imputing missing values and forecasting a target variable (`Y`) in a time series dataset using **XGBoost** and **Optuna** for hyperparameter optimization.

---

## Project Overview

This project consists of two main phases:

1. **Missing Value Imputation**  
   Missing columns (`X06` to `X23`) are filled using independent XGBoost regression models. Each model is trained using only the **complete columns** and the **available values** of the target column it is meant to impute.

2. **Target Variable Forecasting (`Y`)**  
   After filling in all missing `Xn` columns, a separate model is trained to predict `Y`.  
   Features used include:
   - All filled `Xn` columns
   - `Month` and `Year` extracted from the date column  
   Hyperparameters are tuned using **Optuna**, and evaluation is performed using **K-Fold Cross-Validation**.

---

## Key Features

1. **Column-wise Intelligent Imputation**  
   Each column with missing data is filled individually using a supervised learning model. Only complete columns are used as predictors during this step.

2. **Hyperparameter Tuning with Optuna**  
   Efficient automated search for the best hyperparameters to minimize Mean Absolute Error (MAE) during the training of the `Y` model.

3. **Robust Evaluation with K-Fold**  
   The final model is evaluated using a 5-Fold cross-validation strategy with metrics such as MAE and MAPE.

4. **Time-based Visualization**  
   The model generates a line plot showing the predicted `Y` values over time.

---

## Technical Steps

- **Preprocessing**
  - Parse date and extract `Month` and `Year`
- **Missing Data Imputation**
  - Loop through target columns with missing values (`X06` to `X23`)
  - Train a regressor using only non-missing entries of the current target column
- **Y Prediction**
  - Train a model using `train_test_split` and tune with `Optuna`
  - Evaluate final performance with K-Fold CV
  - Predict any remaining missing `Y` values
- **Save Results**
  - Export processed datasets as `.csv` files
  - Plot predicted values over time

---

## File Outputs

- `projecao.csv`: Original dataset
- `projecao_preenchido_corrigido.csv`: Dataset after `Xn` imputation
- `projecao_final.csv`: Final dataset with `Y` predictions
- Matplotlib plot: Time series of predicted `Y` values

---

## Libraries Used

- `pandas`, `numpy`
- `xgboost`
- `optuna`
- `sklearn` (metrics, model_selection)
- `matplotlib`

---
## Conclusion

This project demonstrates how to combine **missing value imputation** and **time series forecasting** using powerful machine learning tools.  
The modular approach allows easy adaptation to other datasets or models (e.g., LightGBM, CatBoost, etc.).

---
