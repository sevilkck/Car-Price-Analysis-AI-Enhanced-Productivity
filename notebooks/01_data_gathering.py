# Databricks notebook source
# MAGIC %md
# MAGIC # Notebook 01: Data Gathering
# MAGIC ## Car Price Prediction Pipeline
# MAGIC
# MAGIC **Objective:** Load `CarPrice_Assignment.csv` from DBFS and perform an initial inspection.
# MAGIC
# MAGIC **Prerequisite:** Upload `CarPrice_Assignment.csv` to `/dbfs/FileStore/car_price/raw/` once before the first run.
# MAGIC You can do this via the Databricks UI: **Catalog → DBFS → FileStore → car_price → raw → Upload**.

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1.1 Load the Dataset

# COMMAND ----------

df = pd.read_csv(RAW_DATA_PATH)
print(f"Dataset loaded successfully!")
print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1.2 First Look

# COMMAND ----------

df.head()

# COMMAND ----------

df.tail()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1.3 Dataset Info

# COMMAND ----------

df.info()

# COMMAND ----------

print("Columns:")
for i, col in enumerate(df.columns, 1):
    print(f"  {i}. {col} ({df[col].dtype})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1.4 Descriptive Statistics

# COMMAND ----------

df.describe().T

# COMMAND ----------

df.describe(include='object').T

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1.5 Missing Values & Duplicates

# COMMAND ----------

missing = df.isnull().sum()
missing_pct = (df.isnull().sum() / len(df)) * 100
missing_df = pd.DataFrame({'Missing Count': missing, 'Missing %': missing_pct})
missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing %', ascending=False)

if len(missing_df) == 0:
    print("No missing values found in the dataset!")
else:
    print("Missing values:")
    print(missing_df)

# COMMAND ----------

dup_count = df.duplicated().sum()
print(f"Number of duplicate rows: {dup_count}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1.6 Target Variable Overview

# COMMAND ----------

print("Target Variable: price")
print(f"  Mean:   ${df['price'].mean():,.2f}")
print(f"  Median: ${df['price'].median():,.2f}")
print(f"  Min:    ${df['price'].min():,.2f}")
print(f"  Max:    ${df['price'].max():,.2f}")
print(f"  Std:    ${df['price'].std():,.2f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1.7 Hand off to next notebook
# MAGIC
# MAGIC We don't need to save anything here — Notebook 02 reads the same raw CSV directly.
# MAGIC This notebook's job is **inspection only**.

# COMMAND ----------

print("✅ Data gathering complete. Next: 02_data_cleaning")
