# Databricks notebook source
# MAGIC %md
# MAGIC # Notebook 05: Preprocessing
# MAGIC
# MAGIC **Objective:** Prepare data for modeling — drop helper columns, one-hot encode categoricals, split train/test, scale numerics.

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv(ENGINEERED_PATH)
print(f"Loaded: {df.shape}")
df.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5.1 Drop helper / intermediate columns
# MAGIC
# MAGIC - `brand` → already replaced by `brand_tier` and `brand_avg_price`
# MAGIC - `log_price` → alternative target, not a feature
# MAGIC - `*_bin` columns → were for EDA, redundant with originals

# COMMAND ----------

drop_cols = ['brand', 'log_price', 'hp_bin', 'enginesize_bin', 'weight_bin']
drop_cols = [c for c in drop_cols if c in df.columns]
df.drop(columns=drop_cols, inplace=True)
print(f"Shape after drop: {df.shape}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5.2 One-hot encode categoricals
# MAGIC
# MAGIC `drop_first=True` avoids the **dummy variable trap** — if you have N categories, you only need N-1 columns. The dropped one is the "baseline" everything else is compared against.

# COMMAND ----------

cat_cols = df.select_dtypes(include='object').columns.tolist()
print(f"Encoding: {cat_cols}")

df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True, dtype=int)
print(f"Shape after encoding: {df_encoded.shape}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5.3 Split features (X) and target (y)

# COMMAND ----------

X = df_encoded.drop(columns=['price'])
y = df_encoded['price']
print(f"X: {X.shape}, y: {y.shape}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5.4 Train-test split
# MAGIC
# MAGIC 80% to train on, 20% held back to test. `random_state=42` makes the split reproducible — same seed → same split every time.

# COMMAND ----------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)
print(f"Train: {X_train.shape[0]} samples")
print(f"Test:  {X_test.shape[0]} samples")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5.5 Scale features
# MAGIC
# MAGIC **Critical pattern**: `fit` on train only, then `transform` both. If we fit on the full dataset, info from test bleeds into train (data leakage) and our test scores become a lie.
# MAGIC
# MAGIC **Real-life analogy**: studying for an exam using the actual exam questions. You'd ace it — but it tells you nothing about real understanding.

# COMMAND ----------

scaler = StandardScaler()
X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train),
    columns=X_train.columns, index=X_train.index
)
X_test_scaled = pd.DataFrame(
    scaler.transform(X_test),
    columns=X_test.columns, index=X_test.index
)
print("Scaling applied (StandardScaler).")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5.6 Leakage check

# COMMAND ----------

overlap = set(X_train.index) & set(X_test.index)
print(f"Index overlap: {len(overlap)} (must be 0)")
assert 'price' not in X_train.columns, "LEAK: price in features!"
print("✅ No data leakage detected.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5.7 Save all splits

# COMMAND ----------

X_train_scaled.to_csv(X_TRAIN_SCALED_PATH, index=False)
X_test_scaled.to_csv(X_TEST_SCALED_PATH, index=False)
X_train.to_csv(X_TRAIN_PATH, index=False)
X_test.to_csv(X_TEST_PATH, index=False)
y_train.to_csv(Y_TRAIN_PATH, index=False)
y_test.to_csv(Y_TEST_PATH, index=False)
df_encoded.to_csv(ENCODED_PATH, index=False)

print("✅ Files saved:")
print(f"  Scaled (for Linear Regression): {X_TRAIN_SCALED_PATH}")
print(f"  Unscaled (for tree models):     {X_TRAIN_PATH}")
print(f"  Targets:                        {Y_TRAIN_PATH}")
print(f"  Full encoded (for PyCaret/CV):  {ENCODED_PATH}")
