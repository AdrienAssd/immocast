# ImmoCast

ImmoCast estimates the price of an apartment in Paris from its living area, number of rooms and postal code. The model is trained on real sales from the French DVF open data (data.gouv.fr).

## Project structure

```
immocast/
├── app/                 Flask web application
│   ├── __init__.py      create_app()
│   ├── model_loader.py  model loading and prediction
│   ├── routes.py        input validation and routes
│   └── templates/
├── ml/                  data preparation and training
│   ├── prepare_data.py
│   ├── model.py         features, models and evaluation
│   ├── train.py         training with MLflow tracking
│   └── README_data.md
├── data/                DVC pointers only (*.dvc)
├── requirements.txt
└── README.md
```

## Run locally

Requirements: Python 3.12, Git.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### External services

| Service | Used for | URL |
|---|---|---|
| DagsHub Storage | DVC remote holding the dataset versions v1, v2 and v3 | https://dagshub.com/AdrienAssd/immocast |
| DagsHub MLflow | Experiment tracking and model registry | https://dagshub.com/AdrienAssd/immocast.mlflow |

Each team member is a collaborator of the DagsHub project and uses their own access token (DagsHub → Settings → Tokens). Tokens are never committed.

Configure DVC once after cloning. The token is written to `.dvc/config.local`, which is ignored by Git:

```bash
dvc remote modify origin --local access_key_id <dagshub-token>
dvc remote modify origin --local secret_access_key <dagshub-token>
```

Get the data, either from the DVC remote:

```bash
dvc pull
```

or rebuild it from data.gouv.fr:

```bash
python ml/prepare_data.py
```

Configure MLflow tracking on DagsHub (PowerShell, same token as DVC):

```powershell
$env:MLFLOW_TRACKING_URI = "https://dagshub.com/AdrienAssd/immocast.mlflow"
$env:MLFLOW_TRACKING_USERNAME = "<dagshub-username>"
$env:MLFLOW_TRACKING_PASSWORD = "<dagshub-token>"
```

Without these variables, runs are stored locally in `mlruns/` (ignored by Git).

Train a model and start the app:

```bash
python ml/train.py --dataset-version v1 --model rf
flask --app app run
```

`ml/train.py` logs each run in the `immocast` MLflow experiment: Git commit, dataset version, DVC hash, hyperparameters, MAE and MAPE on the held-out evaluation split, the model and the evaluation predictions. It also saves `models/model.joblib` for the local app.

| Option | Effect |
|---|---|
| `--dataset-version v1` | Dataset version under `data/` |
| `--model rf` or `--model hgb` | Random forest baseline or histogram gradient boosting candidate |
| `--register immocast-dev` | Registers the model as a new version of `immocast-dev` |
| `--set-champion` | Moves the `champion` alias to that version |

`immocast-dev` is the development registry model. The production model `immocast` is only updated by the continuous training workflow.

Open http://localhost:5000.

### Sample inputs

| Living area | Rooms | Postal code | Expected |
|---|---|---|---|
| 45 | 2 | 75011 | Estimated price |
| 120 | 5 | 75007 | Estimated price |
| 45 | 2 | 69001 | Error: not a Paris postal code |
| 2 | 1 | 75011 | Error: living area out of range |

## Branch flow

| Branch | Role | Updated by |
|---|---|---|
| `main` | Production | Pull request from `staging` only |
| `staging` | Release candidate validation | Pull request from `dev` only |
| `dev` | Integration of the team's work | Pull requests from `feature/*` and `fix/*` |
| `feature/*`, `fix/*` | One short-lived task | Created from `dev`, deleted after merge |

```
feature/* ──PR──▶ dev ──PR──▶ staging ──PR──▶ main
```

Rules:

1. Create every branch from an up-to-date `dev`:
   ```bash
   git switch dev
   git pull
   git switch -c feature/<task>
   ```
2. One branch = one task. Open a pull request into `dev` when the task works locally.
3. Another team member reviews and approves the pull request before it is merged.
4. `dev`, `staging` and `main` are protected: no direct push, pull request and approval required.
5. `dev` is promoted to `staging`, then `staging` to `main`, through pull requests when a milestone is ready.
6. Merged branches are deleted.

Data, models, virtual environments and credentials are never committed (see `.gitignore`).

## Workflows

| Workflow | Trigger | What it does |
|---|---|---|
| `CI/CD` (`.github/workflows/cicd.yml`) | Pull request into `dev`, `staging` or `main` | Ruff lint, pytest, Docker build, container started with a test model and smoke-tested |
| `CI/CD` | Push to `staging` | Same checks on the release candidate (staging validation) |
| `CI/CD` | Push to `main` | Same checks, then push `immocast:<commit sha>` to Docker Hub, deploy to Render and smoke-test the live URL |

The `Lint and tests` and `Docker build and smoke test` jobs are required checks on `dev`, `staging` and `main`. Deployment details are in `DEPLOYMENT.md`.

## Team

| Member | Scope |
|---|---|
| Thibault| Data, DVC, training, MLflow, promotion rules |
| Alexandre | Flask application, model loading, tests |
| Adrien | Docker, CI/CD, Render deployment, continuous training workflow |
