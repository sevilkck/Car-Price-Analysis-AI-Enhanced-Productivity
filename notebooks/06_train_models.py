# Databricks notebook source
# MAGIC %md
# MAGIC # Notebook 06: Train Models
# MAGIC
# MAGIC **Objective:** Train three models and log them with MLflow:
# MAGIC 1. **Linear Regression** — interpretable baseline
# MAGIC 2. **Decision Tree** — captures non-linear splits
# MAGIC 3. **XGBoost** — strong gradient-boosted performance
# MAGIC
# MAGIC **MLflow** is built into Databricks — every run is automatically tracked under the `Experiments` tab on the left sidebar. You'll see metrics, parameters, and saved models without lifting a finger.

# COMMAND ----------

# MAGIC %pip install xgboost -q

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import warnings
warnings.filterwarnings('ignore')

import mlflow
import mlflow.sklearn
import mlflow.xgboost

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# COMMAND ----------

# Load data
X_train_scaled = pd.read_csv(X_TRAIN_SCALED_PATH)
X_test_scaled  = pd.read_csv(X_TEST_SCALED_PATH)
X_train        = pd.read_csv(X_TRAIN_PATH)
X_test         = pd.read_csv(X_TEST_PATH)
y_train        = pd.read_csv(Y_TRAIN_PATH).squeeze()
y_test         = pd.read_csv(Y_TEST_PATH).squeeze()
print(f"Train: {X_train.shape[0]} samples, {X_train.shape[1]} features")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6.1 Helper to train, evaluate, and log to MLflow

# COMMAND ----------

def train_and_log(name, model, X_tr, X_te, y_tr, y_te, log_fn):
    """Train, evaluate, and log to MLflow in one shot."""
    with mlflow.start_run(run_name=name):
        model.fit(X_tr, y_tr)
        y_train_pred = model.predict(X_tr)
        y_test_pred  = model.predict(X_te)

        results = {
            'Model': name,
            'Train R²':   r2_score(y_tr, y_train_pred),
            'Test R²':    r2_score(y_te, y_test_pred),
            'Train RMSE': np.sqrt(mean_squared_error(y_tr, y_train_pred)),
            'Test RMSE':  np.sqrt(mean_squared_error(y_te, y_test_pred)),
            'Train MAE':  mean_absolute_error(y_tr, y_train_pred),
            'Test MAE':   mean_absolute_error(y_te, y_test_pred),
        }

        # Log to MLflow
        mlflow.log_metric("train_r2", results['Train R²'])
        mlflow.log_metric("test_r2",  results['Test R²'])
        mlflow.log_metric("train_rmse", results['Train RMSE'])
        mlflow.log_metric("test_rmse",  results['Test RMSE'])
        mlflow.log_metric("train_mae",  results['Train MAE'])
        mlflow.log_metric("test_mae",   results['Test MAE'])
        log_fn(model, "model")

        print(f"\n{'='*60}\n  {name}\n{'='*60}")
        print(f"  Train R²:  {results['Train R²']:.4f}    |  Test R²:  {results['Test R²']:.4f}")
        print(f"  Train RMSE: ${results['Train RMSE']:,.0f}  |  Test RMSE: ${results['Test RMSE']:,.0f}")
        print(f"  Train MAE:  ${results['Train MAE']:,.0f}  |  Test MAE:  ${results['Test MAE']:,.0f}")

        return results, y_test_pred

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6.2 Linear Regression (uses scaled data)

# COMMAND ----------

lr_model = LinearRegression()
lr_results, lr_preds = train_and_log(
    'Linear Regression', lr_model,
    X_train_scaled, X_test_scaled, y_train, y_test, mlflow.sklearn.log_model
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6.3 Decision Tree (uses unscaled data)
# MAGIC
# MAGIC Trees split on thresholds — scaling doesn't matter. `max_depth=5` keeps the tree readable and prevents overfitting.

# COMMAND ----------

dt_model = DecisionTreeRegressor(max_depth=5, random_state=RANDOM_STATE)
dt_results, dt_preds = train_and_log(
    'Decision Tree', dt_model,
    X_train, X_test, y_train, y_test, mlflow.sklearn.log_model
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6.4 XGBoost

# COMMAND ----------

xgb_model = XGBRegressor(
    n_estimators=200, max_depth=5, learning_rate=0.1,
    random_state=RANDOM_STATE, verbosity=0
)
xgb_results, xgb_preds = train_and_log(
    'XGBoost', xgb_model,
    X_train, X_test, y_train, y_test, mlflow.xgboost.log_model
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6.5 Comparison

# COMMAND ----------

results_df = pd.DataFrame([lr_results, dt_results, xgb_results]).set_index('Model')
print("\n" + "="*70)
print("  MODEL COMPARISON")
print("="*70)
results_df

# COMMAND ----------

models = ['Linear Regression', 'Decision Tree', 'XGBoost']
colors = ['#e74c3c', '#2ecc71', '#3498db']
preds_list = [lr_preds, dt_preds, xgb_preds]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, name, preds, color in zip(axes, models, preds_list, colors):
    ax.scatter(y_test, preds, alpha=0.6, color=color, s=40)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
            'k--', linewidth=1.5, label='Perfect prediction')
    ax.set_xlabel('Actual Price ($)'); ax.set_ylabel('Predicted Price ($)')
    ax.set_title(name); ax.legend()
plt.suptitle('Actual vs Predicted', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6.6 Save models and results to DBFS
# MAGIC
# MAGIC We save pickle files too so notebooks 07 and 08 can load without re-querying MLflow.

# COMMAND ----------

with open(LR_MODEL_PATH, 'wb') as f:
    pickle.dump(lr_model, f)
with open(DT_MODEL_PATH, 'wb') as f:
    pickle.dump(dt_model, f)
with open(XGB_MODEL_PATH, 'wb') as f:
    pickle.dump(xgb_model, f)

results_df.to_csv(RESULTS_PATH)
print(f"✅ Models and results saved to {BASE_PATH}/models and {RESULTS_PATH}")
print("✅ All runs also tracked in MLflow — see Experiments tab in Databricks sidebar")
