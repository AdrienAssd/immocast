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
│   ├── train.py
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

Get the data, either from the DVC remote:

```bash
dvc pull
```

or rebuild it from data.gouv.fr:

```bash
python ml/prepare_data.py
```

Train a model and start the app:

```bash
python ml/train.py --dataset-version v1
flask --app app run
```

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

No GitHub Actions workflow yet. Each workflow and its trigger will be documented here when it is added.

## Team

| Member | Scope |
|---|---|
| Thibault| Data, DVC, training, MLflow, promotion rules |
| Alexandre | Flask application, model loading, tests |
| Adrien | Docker, CI/CD, Render deployment, continuous training workflow |
