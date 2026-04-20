# Databricks notebook source
# MAGIC %md
# MAGIC # Notebook 00: Pipeline Configuration
# MAGIC
# MAGIC Single source of truth for all paths and settings.
# MAGIC Every other notebook does `%run ./00_config` at the top to inherit these values.
# MAGIC
# MAGIC **Why this exists:** if you ever move from DBFS to Unity Catalog, you only change paths here — not in 8 notebooks.

# COMMAND ----------

# Base storage location (DBFS — works on every Databricks workspace)
BASE_PATH = "/dbfs/FileStore/car_price"

# Raw input data (you upload CarPrice_Assignment.csv here once)
RAW_DATA_PATH = f"{BASE_PATH}/raw/CarPrice_Assignment.csv"

# Intermediate outputs (each notebook writes here, the next reads from here)
CLEANED_PATH      = f"{BASE_PATH}/cleaned/car_price_cleaned.csv"
ENGINEERED_PATH   = f"{BASE_PATH}/engineered/car_price_engineered.csv"
ENCODED_PATH      = f"{BASE_PATH}/encoded/car_price_encoded.csv"

# Train/test splits
X_TRAIN_PATH        = f"{BASE_PATH}/splits/X_train.csv"
X_TEST_PATH         = f"{BASE_PATH}/splits/X_test.csv"
X_TRAIN_SCALED_PATH = f"{BASE_PATH}/splits/X_train_scaled.csv"
X_TEST_SCALED_PATH  = f"{BASE_PATH}/splits/X_test_scaled.csv"
Y_TRAIN_PATH        = f"{BASE_PATH}/splits/y_train.csv"
Y_TEST_PATH         = f"{BASE_PATH}/splits/y_test.csv"

# Trained model artifacts (saved as pickle files)
LR_MODEL_PATH       = f"{BASE_PATH}/models/linear_regression_model.pkl"
DT_MODEL_PATH       = f"{BASE_PATH}/models/decision_tree_model.pkl"
XGB_MODEL_PATH      = f"{BASE_PATH}/models/xgboost_model.pkl"

# Results
RESULTS_PATH        = f"{BASE_PATH}/results/model_results.csv"

# Reproducibility
RANDOM_STATE = 42
TEST_SIZE = 0.2

# COMMAND ----------

# Make sure all subdirectories exist before any notebook tries to write to them
import os
for path in [BASE_PATH, f"{BASE_PATH}/raw", f"{BASE_PATH}/cleaned",
             f"{BASE_PATH}/engineered", f"{BASE_PATH}/encoded",
             f"{BASE_PATH}/splits", f"{BASE_PATH}/models", f"{BASE_PATH}/results"]:
    os.makedirs(path, exist_ok=True)

print(f"Base path: {BASE_PATH}")
print("All subdirectories ready.")
