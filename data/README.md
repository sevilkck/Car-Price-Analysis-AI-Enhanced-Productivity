# Data folder

The dataset `CarPrice_Assignment.csv` does **not** live in this repo (data files don't belong in Git).

**Where to put it for the pipeline to find it:**

Upload to DBFS at this path:
```
/dbfs/FileStore/car_price/raw/CarPrice_Assignment.csv
```

Either via the Databricks UI (Catalog → DBFS → FileStore → upload) or via CLI:

```bash
databricks fs cp ./CarPrice_Assignment.csv dbfs:/FileStore/car_price/raw/
```

If you want to override this path, edit `RAW_DATA_PATH` in `notebooks/00_config.py`.
