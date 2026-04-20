# Databricks notebook source
# MAGIC %md
# MAGIC # Notebook 07: Model Evaluation
# MAGIC
# MAGIC **Objective:** Deep evaluation — overfitting check, cross-validation, residual analysis, business interpretation.

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

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

# COMMAND ----------

# Load data and models
X_train_scaled = pd.read_csv(X_TRAIN_SCALED_PATH)
X_test_scaled  = pd.read_csv(X_TEST_SCALED_PATH)
X_train        = pd.read_csv(X_TRAIN_PATH)
X_test         = pd.read_csv(X_TEST_PATH)
y_train        = pd.read_csv(Y_TRAIN_PATH).squeeze()
y_test         = pd.read_csv(Y_TEST_PATH).squeeze()

with open(LR_MODEL_PATH, 'rb') as f: lr_model = pickle.load(f)
with open(DT_MODEL_PATH, 'rb') as f: dt_model = pickle.load(f)
with open(XGB_MODEL_PATH, 'rb') as f: xgb_model = pickle.load(f)

# Generate predictions
lr_train_pred  = lr_model.predict(X_train_scaled);  lr_test_pred  = lr_model.predict(X_test_scaled)
dt_train_pred  = dt_model.predict(X_train);         dt_test_pred  = dt_model.predict(X_test)
xgb_train_pred = xgb_model.predict(X_train);        xgb_test_pred = xgb_model.predict(X_test)

print("Data, models, and predictions ready.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7.1 Comprehensive metrics

# COMMAND ----------

def full_metrics(name, y_tr, y_tr_pred, y_te, y_te_pred):
    return {
        'Model': name,
        'Train R²': r2_score(y_tr, y_tr_pred),
        'Test R²':  r2_score(y_te, y_te_pred),
        'R² Gap':   r2_score(y_tr, y_tr_pred) - r2_score(y_te, y_te_pred),
        'Train RMSE': np.sqrt(mean_squared_error(y_tr, y_tr_pred)),
        'Test RMSE':  np.sqrt(mean_squared_error(y_te, y_te_pred)),
        'Train MAE':  mean_absolute_error(y_tr, y_tr_pred),
        'Test MAE':   mean_absolute_error(y_te, y_te_pred),
        'RMSE/Mean Price %': (np.sqrt(mean_squared_error(y_te, y_te_pred)) / y_te.mean()) * 100
    }

metrics = [
    full_metrics('Linear Regression', y_train, lr_train_pred,  y_test, lr_test_pred),
    full_metrics('Decision Tree',     y_train, dt_train_pred,  y_test, dt_test_pred),
    full_metrics('XGBoost',           y_train, xgb_train_pred, y_test, xgb_test_pred),
]
metrics_df = pd.DataFrame(metrics).set_index('Model')
metrics_df

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7.2 Overfitting check
# MAGIC
# MAGIC **Logic**: if Train R² is much higher than Test R², the model memorized training data instead of learning the pattern. Like a student who aces practice tests but fails the real exam.

# COMMAND ----------

models_names = ['Linear Regression', 'Decision Tree', 'XGBoost']
print("R² Gap (Train - Test):")
for name in models_names:
    gap = metrics_df.loc[name, 'R² Gap']
    status = 'OK' if gap < 0.1 else 'POSSIBLE OVERFIT' if gap < 0.2 else 'OVERFITTING'
    print(f"  {name}: {gap:.4f} → {status}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7.3 5-fold cross-validation
# MAGIC
# MAGIC One train/test split could just be lucky. CV repeats the test 5 times on different slices — much more honest.

# COMMAND ----------

df_full = pd.read_csv(ENCODED_PATH)
X_full = df_full.drop(columns=['price'])
y_full = df_full['price']

cv_models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree':     DecisionTreeRegressor(max_depth=5, random_state=RANDOM_STATE),
    'XGBoost':           XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.1,
                                       random_state=RANDOM_STATE, verbosity=0)
}

print("5-Fold Cross-Validation (RMSE):")
for name, model in cv_models.items():
    scores = cross_val_score(model, X_full, y_full, cv=5, scoring='neg_mean_squared_error')
    rmse_scores = np.sqrt(-scores)
    print(f"  {name}: ${rmse_scores.mean():,.0f} ± ${rmse_scores.std():,.0f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7.4 Residual analysis
# MAGIC
# MAGIC Residual = actual - predicted. We want them clustered around 0 with no pattern. A funnel shape means the model errors more on expensive cars.

# COMMAND ----------

residuals = {
    'Linear Regression': y_test.values - lr_test_pred,
    'Decision Tree':     y_test.values - dt_test_pred,
    'XGBoost':           y_test.values - xgb_test_pred,
}
preds = [lr_test_pred, dt_test_pred, xgb_test_pred]
colors = ['#e74c3c', '#2ecc71', '#3498db']

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
for i, (name, res) in enumerate(residuals.items()):
    axes[0, i].scatter(preds[i], res, alpha=0.6, color=colors[i], s=40)
    axes[0, i].axhline(y=0, color='black', linestyle='--', linewidth=1)
    axes[0, i].set_xlabel('Predicted Price'); axes[0, i].set_ylabel('Residual')
    axes[0, i].set_title(f'{name} — Residuals vs Predicted')

    axes[1, i].hist(res, bins=20, color=colors[i], edgecolor='white')
    axes[1, i].set_xlabel('Residual'); axes[1, i].set_ylabel('Frequency')
    axes[1, i].set_title(f'{name} — Residual Distribution')

plt.suptitle('Residual Analysis', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7.5 Percentage error analysis
# MAGIC
# MAGIC How often is each model within 10% / 20% of the true price? That's what stakeholders actually want to know.

# COMMAND ----------

for name, pred in zip(models_names, preds):
    pct_error = np.abs((y_test.values - pred) / y_test.values) * 100
    print(f"\n{name}:")
    print(f"  Mean error:   {pct_error.mean():.1f}%")
    print(f"  Median error: {np.median(pct_error):.1f}%")
    print(f"  <10% off: {(pct_error < 10).sum()}/{len(pct_error)} ({(pct_error < 10).mean()*100:.0f}%)")
    print(f"  <20% off: {(pct_error < 20).sum()}/{len(pct_error)} ({(pct_error < 20).mean()*100:.0f}%)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7.6 Best model

# COMMAND ----------

best_model_name = metrics_df['Test RMSE'].idxmin()
best_rmse = metrics_df.loc[best_model_name, 'Test RMSE']
best_r2 = metrics_df.loc[best_model_name, 'Test R²']

print(f"\n{'='*60}")
print(f"  BEST MODEL: {best_model_name}")
print(f"{'='*60}")
print(f"  Test R²:   {best_r2:.4f}")
print(f"  Test RMSE: ${best_rmse:,.0f}")
print(f"  RMSE/Mean: {metrics_df.loc[best_model_name, 'RMSE/Mean Price %']:.1f}%")
print(f"\n  Explains {best_r2*100:.1f}% of price variance.")
print(f"  Predictions are off by ~${best_rmse:,.0f} on average.")
