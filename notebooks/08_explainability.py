# Databricks notebook source
# MAGIC %md
# MAGIC # Notebook 08: Model Explainability
# MAGIC
# MAGIC **Objective:** Understand *why* each model makes the predictions it does.
# MAGIC - Linear Regression → coefficients
# MAGIC - Decision Tree → top-level splits
# MAGIC - XGBoost → feature importance

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

X_train_scaled = pd.read_csv(X_TRAIN_SCALED_PATH)
X_train        = pd.read_csv(X_TRAIN_PATH)

with open(LR_MODEL_PATH, 'rb') as f: lr_model = pickle.load(f)
with open(DT_MODEL_PATH, 'rb') as f: dt_model = pickle.load(f)
with open(XGB_MODEL_PATH, 'rb') as f: xgb_model = pickle.load(f)

print("Loaded.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8.1 Linear Regression coefficients
# MAGIC
# MAGIC Each coefficient = "holding everything else constant, how much does price change when this feature goes up by 1?"

# COMMAND ----------

coef_df = pd.DataFrame({
    'Feature': X_train_scaled.columns,
    'Coefficient': lr_model.coef_
}).sort_values('Coefficient', key=abs, ascending=False)

print(f"Intercept (baseline price): ${lr_model.intercept_:,.2f}")
coef_df.head(20)

# COMMAND ----------

top = coef_df.head(20)
colors = ['green' if c > 0 else 'red' for c in top['Coefficient']]
plt.figure(figsize=(10, 10))
plt.barh(range(len(top)), top['Coefficient'], color=colors)
plt.yticks(range(len(top)), top['Feature'])
plt.xlabel('Coefficient ($)')
plt.title('Linear Regression — Top 20 Coefficients\n(Green = increases price, Red = decreases price)')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8.2 Decision Tree — top-level splits
# MAGIC
# MAGIC The root node is the most important split — the single question that best separates expensive from cheap cars.

# COMMAND ----------

from sklearn.tree import plot_tree, export_text

plt.figure(figsize=(28, 14))
plot_tree(dt_model, feature_names=X_train.columns, filled=True,
          rounded=True, fontsize=9, max_depth=3, proportion=True)
plt.title('Decision Tree — Top 3 Levels', fontsize=16)
plt.tight_layout()
plt.show()

# COMMAND ----------

print(export_text(dt_model, feature_names=list(X_train.columns), max_depth=3))

# COMMAND ----------

tree = dt_model.tree_
root_feature = X_train.columns[tree.feature[0]]
root_threshold = tree.threshold[0]

print("="*60)
print("DECISION TREE INTERPRETATION")
print("="*60)
print(f"\nRoot Node (most important feature):")
print(f"  Feature:   {root_feature}")
print(f"  Threshold: {root_threshold:.2f}")
print(f"  → First split: {root_feature} <= {root_threshold:.2f} or > {root_threshold:.2f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8.3 Decision Tree feature importance

# COMMAND ----------

dt_imp = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': dt_model.feature_importances_
}).sort_values('Importance', ascending=False)

top_dt = dt_imp[dt_imp['Importance'] > 0].head(15)
plt.figure(figsize=(10, 8))
plt.barh(range(len(top_dt)), top_dt['Importance'], color='#2ecc71')
plt.yticks(range(len(top_dt)), top_dt['Feature'])
plt.xlabel('Importance')
plt.title('Decision Tree — Feature Importance')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8.4 XGBoost feature importance

# COMMAND ----------

xgb_imp = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': xgb_model.feature_importances_
}).sort_values('Importance', ascending=False)

top_xgb = xgb_imp.head(15)
plt.figure(figsize=(10, 8))
plt.barh(range(len(top_xgb)), top_xgb['Importance'], color='#3498db')
plt.yticks(range(len(top_xgb)), top_xgb['Feature'])
plt.xlabel('Importance')
plt.title('XGBoost — Top 15 Feature Importances')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

xgb_imp.head(10)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8.5 Comparison across models

# COMMAND ----------

comp = pd.DataFrame({'Feature': X_train.columns})
comp = comp.merge(dt_imp.rename(columns={'Importance': 'DT Importance'}), on='Feature')
comp = comp.merge(xgb_imp.rename(columns={'Importance': 'XGB Importance'}), on='Feature')

lr_coef = pd.DataFrame({
    'Feature': X_train_scaled.columns,
    'LR |Coefficient|': np.abs(lr_model.coef_)
})
lr_coef['LR |Coefficient|'] = lr_coef['LR |Coefficient|'] / lr_coef['LR |Coefficient|'].max()
comp = comp.merge(lr_coef, on='Feature', how='left').fillna(0)

comp['Avg Importance'] = (comp['DT Importance'] + comp['XGB Importance'] + comp['LR |Coefficient|']) / 3
comp = comp.sort_values('Avg Importance', ascending=False)
comp.head(15)

# COMMAND ----------

print("✅ Pipeline complete. Models trained, evaluated, and explained.")
