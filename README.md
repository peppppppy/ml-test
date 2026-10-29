# ML Project Starter

A Python starter project for training a scikit-learn-compatible model, generating predictions, and viewing them in a Streamlit dashboard.

## Requirements and setup

Use Python 3.12 or 3.13 and Poetry:

```powershell
py -3.13 -m pip install poetry
py -3.13 -m poetry install
```

The project also provides Make targets on systems with `make`:

```text
make install-dev
make train
make predict
make tune
make dashboard
make lint
make type-check
make test
```

On Windows, run the equivalent Poetry commands directly:

```powershell
py -3.13 -m poetry run python -m src.main train
py -3.13 -m poetry run python -m src.main predict
py -3.13 -m poetry run python -m src.main tune
py -3.13 -m poetry run streamlit run src/streamlit_app.py
```

Training and prediction create a small synthetic example dataset only when the corresponding CSV is missing. Replace `data/train.csv` and `data/inference.csv` with your own data, then set `TARGET_COLUMN` and optionally `FEATURE_COLUMNS` in `src/settings.py`.

## Optional integrations

### MLflow

Set `MLFLOW_TRACKING_URI` in a root `.env` file to opt into experiment logging during `train`. For a local file store, use `file:./mlruns`. To use the helper directly, call `train_with_tracking(X, y)` from `src.model`. Run the UI with:

```powershell
py -3.13 -m poetry run mlflow ui
```

### Optuna

`tune` runs a Random Forest regression study on a held-out validation split and prints the best parameters and validation MAE. It expects a numeric target. To change the number of trials, call `tune_random_forest(X, y, n_trials=50)` from `src.tuning`.

### Supabase

Copy `.env.example` to `.env`, then set `SUPABASE_URL` and `SUPABASE_KEY` using your project credentials. Create a `predictions` table with `prediction` and `predicted_at` columns. `predict` writes locally to `data/predictions.json` and uploads the batch when credentials are configured. The dashboard reads from Supabase when credentials are present; otherwise it reads the local JSON file.

Use a least-privilege key appropriate to the deployment environment and do not commit `.env` or credentials.

## Validation and CI

```powershell
py -3.13 -m poetry run ruff check src tests
py -3.13 -m poetry run mypy src
py -3.13 -m poetry run pytest -q
```

CircleCI runs lint, type checking, and tests for pushes and pull requests.
