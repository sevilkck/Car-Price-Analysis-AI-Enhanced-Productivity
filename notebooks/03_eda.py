# Databricks notebook source
# MAGIC %md
# MAGIC # Notebook 03: Exploratory Data Analysis
# MAGIC
# MAGIC **Objective:** Understand distributions, correlations, and outliers before we engineer features.

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

df = pd.read_csv(CLEANED_PATH)
print(f"Loaded: {df.shape}")
df.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.1 Target variable distribution (price)

# COMMAND ----------

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].hist(df['price'], bins=30, color='steelblue', edgecolor='white')
axes[0].set_title('Price Distribution')
axes[0].set_xlabel('Price'); axes[0].set_ylabel('Frequency')

axes[1].boxplot(df['price'], vert=True)
axes[1].set_title('Price Boxplot'); axes[1].set_ylabel('Price')

axes[2].hist(np.log1p(df['price']), bins=30, color='coral', edgecolor='white')
axes[2].set_title('Log(Price) Distribution')
axes[2].set_xlabel('Log(Price)'); axes[2].set_ylabel('Frequency')

plt.tight_layout()
plt.show()

print(f"Skewness of price: {df['price'].skew():.3f}")
print(f"Skewness of log(price): {np.log1p(df['price']).skew():.3f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.2 Numerical features

# COMMAND ----------

numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
numerical_cols.remove('price')
print(f"Numerical features: {len(numerical_cols)}")

# COMMAND ----------

n_cols = 4
n_rows = (len(numerical_cols) + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 4))
axes = axes.flatten()

for i, col in enumerate(numerical_cols):
    axes[i].hist(df[col], bins=25, color='steelblue', edgecolor='white')
    axes[i].set_title(f'{col} (skew={df[col].skew():.2f})')
for j in range(i+1, len(axes)):
    axes[j].set_visible(False)
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.3 Skewness summary

# COMMAND ----------

skew_df = df[numerical_cols + ['price']].skew().sort_values(ascending=False)
skew_df = pd.DataFrame({'Feature': skew_df.index, 'Skewness': skew_df.values})
skew_df['Status'] = skew_df['Skewness'].apply(
    lambda x: 'Highly Skewed' if abs(x) > 1 else ('Moderately Skewed' if abs(x) > 0.5 else 'OK')
)
skew_df

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.4 Categorical features

# COMMAND ----------

cat_cols = df.select_dtypes(include='object').columns.tolist()
print(f"Categorical features: {cat_cols}")

n_cols_plot = 3
n_rows_plot = (len(cat_cols) + n_cols_plot - 1) // n_cols_plot
fig, axes = plt.subplots(n_rows_plot, n_cols_plot, figsize=(18, n_rows_plot * 4))
axes = axes.flatten()
for i, col in enumerate(cat_cols):
    df[col].value_counts().plot(kind='bar', ax=axes[i], color='steelblue', edgecolor='white')
    axes[i].set_title(col)
    axes[i].tick_params(axis='x', rotation=45)
for j in range(i+1, len(axes)):
    axes[j].set_visible(False)
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.5 Price by brand

# COMMAND ----------

brand_avg = df.groupby('brand')['price'].mean().sort_values(ascending=False)
plt.figure(figsize=(14, 6))
brand_avg.plot(kind='bar', color='steelblue', edgecolor='white')
plt.title('Average Price by Brand')
plt.ylabel('Average Price ($)'); plt.xlabel('Brand')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.6 Correlation analysis

# COMMAND ----------

corr = df[numerical_cols + ['price']].corr()
plt.figure(figsize=(16, 12))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, vmin=-1, vmax=1, linewidths=0.5)
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.show()

# COMMAND ----------

price_corr = corr['price'].drop('price').sort_values(ascending=False)
print("Correlation with Price:")
print(price_corr.to_string())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.7 Outlier counts (IQR method)

# COMMAND ----------

def count_outliers(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    return ((series < Q1 - 1.5 * IQR) | (series > Q3 + 1.5 * IQR)).sum()

print("Outlier counts (IQR method):")
for col in numerical_cols + ['price']:
    n = count_outliers(df[col])
    if n > 0:
        print(f"  {col}: {n} outliers")

# COMMAND ----------

# MAGIC %md
# MAGIC ## EDA Summary
# MAGIC
# MAGIC 1. **Price is right-skewed** → log transformation will help linear models
# MAGIC 2. **Top correlated**: enginesize, curbweight, horsepower, carwidth
# MAGIC 3. **Multicollinearity**: citympg ↔ highwaympg, carlength ↔ wheelbase
# MAGIC 4. **Brand strongly influences price** — luxury brands much higher
# MAGIC
# MAGIC EDA produces no new dataset — `cleaned` flows straight to feature engineering.

# COMMAND ----------

print("✅ EDA complete. Next: 04_feature_engineering")
