# ML Car Price Prediction

Can machine learning accurately predict used car prices, and does manual model building outperform automated tools?  
This project answers that question through a full end-to-end regression workflow using traditional models, AutoML, and explainability methods.


---

## Tools & Skills Used

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=flat&logo=matplotlib&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-3776AB?style=flat&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![PyCaret](https://img.shields.io/badge/PyCaret-1B9E77?style=flat&logoColor=white)
![SHAP](https://img.shields.io/badge/SHAP-4B0082?style=flat&logoColor=white)
![LIME](https://img.shields.io/badge/LIME-32CD32?style=flat&logoColor=white)

---

## Quick Access

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Workflow Phases](#workflow-phases)
- [Model Results](#model-results)
- [Key Findings](#key-findings)

---

# Project Overview

This project builds a complete machine learning pipeline to predict car prices based on vehicle specifications such as engine size, weight, fuel type, and brand.

The workflow compares:

1. **Manual Machine Learning Models**
2. **Automated Machine Learning using PyCaret**
3. **Explainability methods** to understand why models make predictions

The goal was not only to build accurate models, but also to understand which features most strongly influence car prices.

---

# Dataset

This project uses the **Car Price Dataset** from Kaggle.

- **205 cars**
- **26 features**
- Target variable: **price**

Features include:

- engine size  
- curb weight  
- horsepower  
- fuel type  
- body style  
- brand name  
- fuel economy

---

# Workflow Phases

| Phase | Notebook | Description |
|------|----------|-------------|
| 1 | `01_Data_Gathering.ipynb` | Load dataset and inspect raw data |
| 2 | `02_Data_Cleaning.ipynb` | Handle missing values, fix inconsistencies, clean columns |
| 3 | `03_EDA.ipynb` | Exploratory data analysis, distributions, correlations |
| 4 | `04_Feature_Engineering.ipynb` | Create new features such as avg_mpg, brand tier, power-to-weight |
| 5 | `05_Preprocessing.ipynb` | Encoding, scaling, train-test split |
| 6 | `06_Train_Models.ipynb` | Train Linear Regression, Decision Tree, XGBoost |
| 7 | `07_Evaluation.ipynb` | Compare MAE, RMSE, R² and visualize results |
| 8 | `08_Explainability.ipynb` | Model interpretation with coefficients and tree logic |
| 9 | `09_PyCaret.ipynb` | AutoML benchmark with multiple regression models |
| 10 | `10_SHAP_LIME.ipynb` | SHAP and LIME explainability for black-box models |

---

# Models Used

## Manual Models

- Linear Regression  
- Decision Tree  
- XGBoost  

## Automated Models

Using **PyCaret**, around 17 regression models were benchmarked automatically, including:

- Random Forest  
- Extra Trees  
- CatBoost  
- LightGBM  
- XGBoost  

---

# Evaluation Metrics

Models were compared using:

| Metric | Meaning |
|-------|---------|
| MAE | Average prediction error |
| RMSE | Penalizes larger mistakes more strongly |
| R² | Percentage of price variation explained |

---

# Model Results

## Manual Workflow

| Model | R² | RMSE |
|------|----|------|
| Linear Regression | 0.89 | $2,888 |
| Decision Tree | 0.85 | $3,400 |
| XGBoost | 0.94 | $2,182 |

### Best Manual Model: **XGBoost**

---

## PyCaret Benchmark

PyCaret ranked tree-based ensemble models highest.

Top performers included:

- Extra Trees  
- Random Forest  
- XGBoost  

This confirmed that ensemble models were strongest for this dataset.

---

# Explainability Analysis

To understand predictions from black-box models, two explainability tools were used:

## SHAP

Explains how much each feature increases or decreases predictions.

## LIME

Creates local explanations for individual predictions.

### Most Important Features Across Methods:

- engine size  
- curb weight  
- brand tier / brand value

Because multiple methods agreed, confidence in the model findings increased.

---

# Key Findings

## 1. Ensemble Models Performed Best

Tree-based models such as XGBoost, Random Forest, and Extra Trees achieved the strongest results.

## 2. Feature Engineering Added Real Value

Custom features such as:

- average MPG  
- power-to-weight ratio  
- brand tier

improved model performance.

## 3. Explainability Increased Trust

SHAP and LIME helped explain why the models made their predictions.

---

# Limitations

- Small dataset (**205 cars**)  
- Missing important variables such as:
  - mileage  
  - manufacturing year  
  - inflation / market timing

---

# Future Improvements

- Use a larger and newer dataset  
- Hyperparameter tuning  
- Add mileage and year features  
- Deploy as a real price prediction app

---
