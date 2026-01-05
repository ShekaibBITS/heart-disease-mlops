# Full pipeline (data + eda + train + mlflow logs)
python scripts/run_pipeline.py

# Or stage-wise:
python -m src.heartml.data_ingest
python -m src.heartml.eda
python -m src.heartml.train

# MLflow UI
mlflow ui --backend-store-uri mlruns --host 127.0.0.1 --port 5000
