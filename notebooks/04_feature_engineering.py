# Databricks notebook source
# MAGIC %md
# MAGIC # Notebook 04: Feature Engineering
# MAGIC
# MAGIC **Objective:** Create new features that help models learn faster and better:
# MAGIC - **Brand tier**: group 22 brands into 3 tiers (luxury / mid / economy) based on average price
# MAGIC - **Brand average price**: numeric encoding of brand value
# MAGIC - **Power-to-weight ratio**: a known mechanical predictor of car value
# MAGIC - **Binned versions** of skewed features (for EDA / interpretation only)
# MAGIC - **Log price**: a less-skewed target that some models prefer
# MAGIC
# MAGIC **Real-life analogy:** raw ingredients (flour, eggs, sugar) vs. prepared mix (pancake batter). Models cook faster and better with the right mix.
# MAGIC
# MAGIC > ⚠️ **Note**: This notebook was reconstructed from references in `05_preprocessing.ipynb`. If you have your original `04_Feature_Engineering.ipynb`, paste its logic in here.

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv(CLEANED_PATH)
print(f"Loaded: {df.shape}")
df.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4.1 Brand average price
# MAGIC
# MAGIC We replace each brand with its **average price**. This turns a high-cardinality categorical (22 brands) into a meaningful number — without creating 22 dummy columns.

# COMMAND ----------

brand_avg_map = df.groupby('brand')['price'].mean().to_dict()
df['brand_avg_price'] = df['brand'].map(brand_avg_map)
print(df[['brand', 'brand_avg_price']].drop_duplicates().sort_values('brand_avg_price', ascending=False).head(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4.2 Brand tier
# MAGIC
# MAGIC Group brands into three buckets — much easier for models (and humans) to interpret than 22 separate brands.
# MAGIC - **Luxury**: top third of brand_avg_price
# MAGIC - **Mid**: middle third
# MAGIC - **Economy**: bottom third

# COMMAND ----------

# Use tertiles of the brand-level average price (not row-level), so each brand gets one tier
brand_avg_series = df.groupby('brand')['price'].mean().sort_values()
tier_map = {}
n = len(brand_avg_series)
for i, brand in enumerate(brand_avg_series.index):
    if i < n / 3:
        tier_map[brand] = 'economy'
    elif i < 2 * n / 3:
        tier_map[brand] = 'mid'
    else:
        tier_map[brand] = 'luxury'

df['brand_tier'] = df['brand'].map(tier_map)
print(df['brand_tier'].value_counts())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4.3 Power-to-weight ratio
# MAGIC
# MAGIC A classic automotive metric: how much horsepower does the car have *per unit of weight*. Sports cars score high; trucks score low. This single number captures something both `horsepower` and `curbweight` together cannot.

# COMMAND ----------

df['power_to_weight'] = df['horsepower'] / df['curbweight']
print(df[['horsepower', 'curbweight', 'power_to_weight']].describe())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4.4 Log price
# MAGIC
# MAGIC EDA showed price is right-skewed. `log(price)` is closer to a normal distribution, which helps linear models. We keep both — modeling notebooks decide which to use.

# COMMAND ----------

df['log_price'] = np.log1p(df['price'])
print(f"Skewness: price={df['price'].skew():.2f}, log_price={df['log_price'].skew():.2f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4.5 Binned versions (for EDA only)
# MAGIC
# MAGIC These get **dropped before modeling** in notebook 05 — they're useful for plots/groupings, not for the model itself (would create redundancy with the originals).

# COMMAND ----------

df['hp_bin']         = pd.qcut(df['horsepower'], q=4, labels=['low', 'medium', 'high', 'very_high'])
df['enginesize_bin'] = pd.qcut(df['enginesize'], q=4, labels=['small', 'medium', 'large', 'xlarge'])
df['weight_bin']     = pd.qcut(df['curbweight'], q=4, labels=['light', 'medium', 'heavy', 'very_heavy'])

# Convert to string so they survive the CSV roundtrip cleanly
df['hp_bin'] = df['hp_bin'].astype(str)
df['enginesize_bin'] = df['enginesize_bin'].astype(str)
df['weight_bin'] = df['weight_bin'].astype(str)

print(df[['hp_bin', 'enginesize_bin', 'weight_bin']].describe())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4.6 Save engineered dataset

# COMMAND ----------

print(f"Final shape: {df.shape}")
print(f"New columns added: brand_avg_price, brand_tier, power_to_weight, log_price, hp_bin, enginesize_bin, weight_bin")

df.to_csv(ENGINEERED_PATH, index=False)
print(f"✅ Engineered data saved to {ENGINEERED_PATH}")
