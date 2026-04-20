# Databricks notebook source
# MAGIC %md
# MAGIC # Notebook 02: Data Cleaning
# MAGIC
# MAGIC **Objective:** Fix typos in brand names, drop unhelpful columns, convert text-based numbers to numeric, remove near-zero-variance columns.

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv(RAW_DATA_PATH)
print(f"Loaded: {df.shape}")
df.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.1 Extract Brand from CarName
# MAGIC
# MAGIC `CarName` mixes brand and model (e.g., `"toyota corolla"`). We split on the first space, lowercase, and fix typos.

# COMMAND ----------

df['brand'] = df['CarName'].str.split(' ').str[0].str.lower().str.strip()

print(f"Unique brands before cleaning: {df['brand'].nunique()}")
print(sorted(df['brand'].unique()))

# COMMAND ----------

brand_fix = {
    'alfa-romero': 'alfa-romeo',
    'maxda': 'mazda',
    'porcshce': 'porsche',
    'toyouta': 'toyota',
    'vokswagen': 'volkswagen',
    'vw': 'volkswagen'
}
df['brand'] = df['brand'].replace(brand_fix)

print(f"Unique brands after cleaning: {df['brand'].nunique()}")
print(sorted(df['brand'].unique()))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.2 Drop columns we don't need
# MAGIC
# MAGIC - `car_ID`: arbitrary index, no predictive value.
# MAGIC - `CarName`: replaced by `brand`.

# COMMAND ----------

df.drop(columns=['car_ID', 'CarName'], inplace=True)
print(f"Shape after dropping: {df.shape}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.3 Convert text-based numbers to numeric
# MAGIC
# MAGIC `cylindernumber` and `doornumber` are stored as words (`"four"`, `"six"`). Models need numbers.

# COMMAND ----------

cylinder_map = {'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'eight': 8, 'twelve': 12}
df['cylindernumber'] = df['cylindernumber'].map(cylinder_map)
print(df['cylindernumber'].value_counts().sort_index())

# COMMAND ----------

door_map = {'two': 2, 'four': 4}
df['doornumber'] = df['doornumber'].map(door_map)
print(df['doornumber'].value_counts().sort_index())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.4 Drop near-zero variance columns
# MAGIC
# MAGIC `enginelocation` is ~98.5% `'front'` — it carries almost no information. Keeping it would just add noise and make models slower.

# COMMAND ----------

print(df['enginelocation'].value_counts(normalize=True) * 100)

# COMMAND ----------

df.drop(columns=['enginelocation'], inplace=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.5 Final checks

# COMMAND ----------

print(f"Final shape: {df.shape}")
print(f"Missing values: {df.isnull().sum().sum()}")
print(f"Duplicates: {df.duplicated().sum()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.6 Save cleaned data for the next notebook

# COMMAND ----------

df.to_csv(CLEANED_PATH, index=False)
print(f"✅ Cleaned data saved to {CLEANED_PATH}")
