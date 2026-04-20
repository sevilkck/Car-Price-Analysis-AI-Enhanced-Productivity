# Car Price Prediction — Databricks Pipeline

End-to-end machine learning pipeline that predicts used-car prices, orchestrated as a sequential Databricks workflow.

> **Status**: ready to deploy. Built with Databricks Asset Bundles, MLflow tracking, and DBFS storage.

---

## What this pipeline does

Given a dataset of 205 cars with 26 attributes (engine size, horsepower, body type, brand, etc.), the pipeline trains three regression models that predict the sale price:

| Model | Strength |
|---|---|
| **Linear Regression** | Highly interpretable — every feature gets a $ coefficient |
| **Decision Tree** | Captures non-linear splits, visually explainable |
| **XGBoost** | Best raw accuracy via gradient boosting |

All runs are logged to MLflow (built into Databricks) so you can compare experiments over time.

---

## Pipeline structure

```
┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│ 01 Data          │───▶│ 02 Data Cleaning │───▶│ 03 EDA           │
│    Gathering     │    │                  │    │                  │
└──────────────────┘    └──────────────────┘    └──────────────────┘
                                                          │
                                                          ▼
┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│ 06 Train Models  │◀───│ 05 Preprocessing │◀───│ 04 Feature       │
│   (+ MLflow)     │    │                  │    │    Engineering   │
└──────────────────┘    └──────────────────┘    └──────────────────┘
         │
         ▼
┌──────────────────┐    ┌──────────────────┐
│ 07 Evaluation    │───▶│ 08 Explainability│
└──────────────────┘    └──────────────────┘
```

Each notebook reads from DBFS, transforms, and writes back — paths are defined once in `00_config.py` and shared via `%run`.

---

## Repository layout

```
car_price_databricks/
├── README.md                    ← you are here
├── EXECUTIVE_SUMMARY.md         ← stakeholder-friendly results overview
├── databricks.yml               ← orchestration config (Asset Bundle)
├── notebooks/
│   ├── 00_config.py             ← shared paths & settings
│   ├── 01_data_gathering.py
│   ├── 02_data_cleaning.py
│   ├── 03_eda.py
│   ├── 04_feature_engineering.py
│   ├── 05_preprocessing.py
│   ├── 06_train_models.py
│   ├── 07_evaluation.py
│   └── 08_explainability.py
└── data/
    └── README.md                ← where to put CarPrice_Assignment.csv
```

---

## Setup — first-time install

### 1. Get the dataset into Databricks

Upload `CarPrice_Assignment.csv` to DBFS once:

**Via UI**: Databricks workspace → **Catalog** (left sidebar) → **DBFS** → **FileStore** → create folder `car_price/raw/` → upload the CSV.

**Via CLI**:
```bash
databricks fs cp ./data/CarPrice_Assignment.csv dbfs:/FileStore/car_price/raw/
```

### 2. Install the Databricks CLI (if you don't have it)

```bash
# macOS
brew tap databricks/tap
brew install databricks

# Verify
databricks --version
```

### 3. Authenticate

```bash
databricks configure
# Paste your workspace URL: https://your-workspace.cloud.databricks.com
# Paste a personal access token (User Settings → Developer → Access tokens → Generate new token)
```

### 4. Edit the workspace URL in `databricks.yml`

Open `databricks.yml` and replace:
```yaml
host: "https://YOUR-WORKSPACE.cloud.databricks.com"
```
with your actual workspace URL.

---

## Deploy and run

```bash
# Deploy notebooks + job config to your workspace
databricks bundle deploy

# Trigger the full pipeline once
databricks bundle run car_price_pipeline
```

**Or run from the UI**: Workspace → Workflows → "Car Price Prediction — Full Pipeline" → **Run now**.

The job spins up a single-node cluster, runs all 8 notebooks sequentially, then shuts the cluster down. Total time: ~5–10 minutes. Cost: a few cents.

---

## What you get after a successful run

- **DBFS** (`/dbfs/FileStore/car_price/`):
  - `models/*.pkl` — trained model artifacts
  - `results/model_results.csv` — comparison table
  - `splits/*.csv` — train/test data
- **MLflow** (Experiments tab in Databricks): every model run with metrics, parameters, and the model itself logged. Compare runs side-by-side.

---

## Local development

The notebooks are stored as Databricks `.py` source files (with `# COMMAND ----------` cell separators). They're plain Python — Git tracks them cleanly, line-by-line diffs work, code review is easy.

To open them as notebooks: import any `.py` file into Databricks via **Workspace → Import → File** and Databricks will render the cells.

---

## Troubleshooting

| Issue | Fix |
|---|---|
| `FileNotFoundError: CarPrice_Assignment.csv` | Upload the CSV to `/dbfs/FileStore/car_price/raw/` first |
| `ModuleNotFoundError: xgboost` | The `15.4.x-cpu-ml-scala2.12` runtime in `databricks.yml` includes it — make sure you didn't change that |
| `Permission denied` writing to DBFS | Switch to a Unity Catalog volume by editing `BASE_PATH` in `00_config.py` |
| Workspace URL wrong | Re-edit `databricks.yml` and re-run `databricks bundle deploy` |

---

## Reproducibility

- All splits use `random_state=42`
- All notebooks pin storage paths via `00_config.py` (single source of truth)
- MLflow auto-logs Git commit hash on every run (when deployed via bundle from a Git repo)
- Cluster spec is pinned in `databricks.yml` (Spark + Python versions)

---

## Source

Migrated from the Google Colab pipeline (8 notebooks). Changes for Databricks:
- `drive.mount()` removed — DBFS is always available
- `!pip install` → `%pip install` (Databricks magic)
- File paths centralized in `00_config.py`
- MLflow tracking added in notebook 06
- Sequential execution orchestrated by `databricks.yml`
