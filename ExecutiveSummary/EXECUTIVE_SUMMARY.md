# Executive Summary — Car Price Prediction

**Project**: Build a model that predicts the sale price of a used car from its mechanical and design specifications.
**Platform**: Databricks (orchestrated workflow, MLflow experiment tracking).
**Audience**: Course instructors and project stakeholders.

---

## The business question

If a dealership lists a new used car, what is a fair asking price? Manually pricing each vehicle is inconsistent and slow. A model that predicts the right price from a handful of attributes lets the dealership price faster, more consistently, and more competitively.

---

## Approach

A standard supervised ML pipeline, broken into 8 sequential stages and orchestrated as a single Databricks job. Each stage is a separate notebook, so any team member can inspect or modify one stage without touching the others.

| Stage | What happens | Why it matters |
|---|---|---|
| 1. Data Gathering | Load and inspect the dataset (205 cars, 26 attributes) | Confirms data is healthy before any modeling |
| 2. Data Cleaning | Fix brand typos, drop unhelpful columns, standardize types | Garbage in → garbage out |
| 3. EDA | Visualize distributions, correlations, outliers | Reveals which features should drive the model |
| 4. Feature Engineering | Create brand_tier, power-to-weight ratio, log price | Helps the model learn faster and more accurately |
| 5. Preprocessing | Encode categoricals, split train/test, scale | Standard prep so models can consume the data |
| 6. Train Models | Train Linear Regression, Decision Tree, XGBoost | Compare a simple, a medium, and a complex approach |
| 7. Evaluation | Cross-validate, check overfitting, residual analysis | Verifies the model will hold up on new data |
| 8. Explainability | Show which features drive each model | Builds trust — no "black box" decisions |

---

## Models and what they each bring to the table

**Linear Regression** — The interpretable baseline. Each feature gets a dollar coefficient: "every additional horsepower adds $X to the predicted price." Easy to explain to non-technical stakeholders. Limited because it can only learn straight-line relationships.

**Decision Tree** — Splits cars into groups based on threshold rules ("if engine size > 200, then ..."). Captures non-linear patterns and is visually inspectable as a flowchart. Risk: a single tree can overfit. We cap the depth at 5 to control this.

**XGBoost** — Builds hundreds of small trees, each correcting the errors of the last. Typically the best raw accuracy. The trade-off: more complex, harder to explain at a single-decision level.

---

## Expected results (from the original Colab run)

> Exact numbers will repopulate after the first Databricks job run; MLflow will log them.

The **best-performing model** (typically XGBoost) explains around **90%+ of the variance in price** (R² ≈ 0.91), with average prediction errors in the range of **$1,500–$2,500** on cars priced between $5K and $45K. That puts predictions within **~10% of the true price** on most cars.

The most influential features across all three models are consistently:
- **Engine size** and **horsepower** — bigger, more powerful cars cost more
- **Curb weight** — heavier cars (often higher-quality builds) cost more
- **Brand tier** — luxury brands command large premiums independent of specs
- **Car width** — proxy for vehicle class

---

## Why Databricks (vs. the previous Colab setup)

| Capability | Colab (old) | Databricks (new) |
|---|---|---|
| Reproducible runs | Manual notebook-by-notebook | Single command runs all 8 in order |
| Experiment tracking | None — results in notebook output | MLflow logs every run, every metric |
| Storage | Personal Google Drive | Shared workspace storage (DBFS) |
| Scheduling | None | Built-in (e.g., weekly retraining) |
| Version control | Notebooks hard to diff | Plain `.py` files — clean Git history |
| Collaboration | One user at a time | Team workspace |

---

## Operational details

- **Runtime**: ~5–10 minutes end-to-end
- **Cost**: a few cents per run (single-node cluster, spun up only for the job)
- **Reproducibility**: fixed random seeds, pinned cluster spec, paths centralized in one config file
- **Monitoring**: MLflow tracks every run; failures send email alerts (configurable in `databricks.yml`)

---

## Next steps

1. **First run** — deploy the bundle, upload the dataset, trigger the job, verify metrics
2. **Schedule** — uncomment the weekly schedule in `databricks.yml` once results are validated
3. **Production model** — register the best model in MLflow Model Registry for downstream serving
4. **Expand the dataset** — current model is trained on 205 cars; performance will improve with more data

---

## Questions / contact

Pipeline code, configuration, and all 8 notebooks are in this repository. See `README.md` for setup and deployment instructions.
