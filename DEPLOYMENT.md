# Deployment

ImmoCast runs as a single Docker container on Render. GitHub Actions tests every change, validates release candidates in a container, and deploys `main` to the production service.

## Architecture

```
GitHub (main) ──▶ GitHub Actions ──push──▶ Docker Hub (immocast:<commit sha>)
                        │                           │
                        └──deploy hook──▶ Render ◀──┘ pulls the image
                                            │
                                            └──▶ DagsHub MLflow (model registry, at startup)
```

| Component | Service |
|---|---|
| Web application and inference | One Docker container (Flask served by waitress) |
| Image registry | Docker Hub, private repository `immocast` |
| Hosting | Render web service (image-backed) |
| Model registry | DagsHub MLflow |
| Dataset storage | DagsHub Storage (DVC remote) |

## Workflow triggers

Workflow file: `.github/workflows/cicd.yml`.

| Trigger | Jobs | Outcome |
|---|---|---|
| Pull request into `dev`, `staging` or `main` | `Lint and tests`, `Docker build and smoke test` | Ruff, pytest, Docker build, container started with a test model and smoke-tested. Required checks: a failing pull request cannot be merged. |
| Push to `staging` (merge of `dev`) | `Lint and tests`, `Docker build and smoke test` | The release candidate is validated in a container started and tested by Actions. |
| Push to `main` (merge of `staging`) | `Lint and tests`, `Docker build and smoke test`, `Deploy to Render` | The image is pushed as `immocast:<commit sha>`, Render deploys it through the deploy hook, the workflow waits for readiness and smoke-tests the live URL. |

The smoke test (`scripts/smoke_test.py`) checks:

1. `/health` returns 200 within the timeout.
2. `/model-info` reports a ready model, the expected model version and the expected commit.
3. `/predict` returns a price per m² between 3,000 and 30,000 € for a valid input and the expected model version.
4. `/predict` rejects a postal code outside Paris.

In production, the expected model version is the version selected by the `champion` alias in the MLflow registry. Until a champion exists, the workflow only checks that the new image is live.

## Continuous training workflow

Workflow file: `.github/workflows/ct-pipeline.yml`, started manually from *Actions → Continuous training → Run workflow* on `main`. Runs are serialized so two operations never move the production model at the same time.

| Input | Values |
|---|---|
| `operation` | `pipeline` or `rollback` |
| `dataset_version` | `v1`, `v2` or `v3`: selecting a version simulates the delivery of a new data batch |
| `rollback_version` | A previously approved model version, for `rollback` only |

Pipeline steps (`ml/pipeline.py`):

1. Fetch the selected version with `dvc pull` and validate it: required columns, missing values, value ranges, duplicates, no overlap and chronological order between training and evaluation. A failure stops the run before any registry change.
2. Load the champion (`models:/immocast@champion`), score it on the evaluation split and apply the retraining rule. Without a champion, train a baseline to bootstrap.
3. Train a candidate on the training split and compare it with the champion on the same evaluation rows using the quality floor and the minimum improvement in `ml/policy.py`.
4. Register a passing candidate, tag it `approved=true` and move the `champion` alias to it. A failing candidate is not registered and the champion is kept, with the reason in `decision.json`.
5. Call `POST /reload` on the production service (protected by `RELOAD_TOKEN`) and run the smoke test against the live URL with the expected champion version.

The decision, the metrics, the registry versions before and after and the DVC pointer are uploaded as the `ct-results-*` artifact and logged in the MLflow experiment `immocast-ct`.

Additional secret: `RELOAD_TOKEN` (same value as in the Render environment).

## Image identification

Every deployed image is tagged with the full Git commit SHA: `<dockerhub-user>/immocast:<sha>`. The same SHA is baked into the image (`GIT_SHA`) and returned by `/model-info` as `git_sha`, so the commit running in production can always be checked.

## Configuration

All settings and credentials are provided at runtime. None are stored in the image or the repository.

### GitHub repository variables and secrets

| Name | Type | Used by |
|---|---|---|
| `DOCKERHUB_USERNAME` | Variable | Image name, Docker Hub login |
| `DOCKERHUB_TOKEN` | Secret | Docker Hub login (read and write token) |
| `RENDER_DEPLOY_HOOK` | Secret | Triggers the Render deployment |
| `PROD_URL` | Variable | Production smoke test |
| `MLFLOW_TRACKING_URI` | Variable | Expected champion version |
| `MLFLOW_TRACKING_USERNAME` | Variable | DagsHub login |
| `MLFLOW_TRACKING_PASSWORD` | Secret | DagsHub token |
| `MODEL_NAME` | Variable (optional) | Registry model name, `immocast` by default |

### Render environment

| Name | Value |
|---|---|
| `PORT` | Set by Render, the container listens on it |
| `MLFLOW_TRACKING_URI`, `MLFLOW_TRACKING_USERNAME`, `MLFLOW_TRACKING_PASSWORD` | DagsHub MLflow access |
| `MODEL_NAME` | `immocast` |
| `LOG_LEVEL` | Optional, `INFO` by default |

Render settings: image-backed web service, Docker Hub credentials for the private image, health check path `/health`.

## Reproduce a deployment

Deploy a given commit by hand:

```bash
docker build --platform linux/amd64 --build-arg GIT_SHA=<sha> -t <dockerhub-user>/immocast:<sha> .
docker push <dockerhub-user>/immocast:<sha>
curl -X POST "<render-deploy-hook>&imgURL=docker.io%2F<dockerhub-user>%2Fimmocast%3A<sha>"
python scripts/smoke_test.py --url <prod-url> --expected-sha <sha>
```

`--platform linux/amd64` is required when building on an Apple Silicon Mac: Render only runs `linux/amd64` images. GitHub Actions runners already build for that platform.

Run the production container locally with a model file:

```bash
docker run --rm -p 8000:8000 -v "${PWD}/models:/app/models:ro" -e GIT_SHA=local <dockerhub-user>/immocast:<sha>
```

## Roll back the application

Redeploy a previous image: run the deploy hook with the `imgURL` of an earlier commit SHA (see *Reproduce a deployment*), or pick the previous deploy in the Render dashboard and choose *Rollback*. Then run the smoke test with that SHA.

## Roll back the model

Run *Actions → Continuous training → Run workflow* with `operation=rollback` and `rollback_version=<approved version>`. The workflow moves the `champion` alias back, reloads the production model and verifies the restored version with a live prediction. A version that does not exist or was never approved is refused and the alias does not move.

Equivalent command:

```bash
python ml/pipeline.py --rollback <model version>
curl -X POST "<prod-url>/reload" -H "Authorization: Bearer <reload-token>"
```

## If the service does not become ready

Render only routes traffic to a new deploy once `/health` returns 200. If the new container never becomes ready, Render keeps serving the previous deploy and marks the new one as failed. The workflow smoke test then fails, so the problem is visible on the commit. Check the Render logs, fix the cause, or roll back.

After a model change, `/reload` loads the new champion before swapping it in. If the new version cannot be loaded, `/reload` returns 503 and the service keeps serving the previous model, so users are not affected. The workflow then fails at the reload or smoke test step, while the registry alias already points to the new version. Because a restart would load that alias, roll back right away with the continuous training workflow (`operation=rollback`, previous approved version from `decision.json`), then investigate the failed version.

## Monitoring and alerting

### Logs

The API logs JSON lines to standard output with `structlog`. The level comes from `LOG_LEVEL`. Each request gets a request ID (read from `X-Request-ID` or generated) bound to every line of that request and returned in the response header. Requests are logged with `method`, `path`, `status` and `duration_ms`; predictions with `model_version`, `prix_m2`, `arrondissement`, `latency_ms`, `n_features` and `n_missing`. Full inputs, the reload token and MLflow credentials are never logged. Rejected inputs are logged as `warning`, server errors as `error`. On Render the same lines appear in the service logs. Locally, Docker keeps at most 3 log files of 10 MB for the API.

### Metrics

| Metric | Type | Labels |
|---|---|---|
| `http_requests_total` | counter | `method`, `path`, `status` |
| `http_request_duration_seconds` | histogram | `path` |
| `predictions_total` | counter | `model_version` |
| `predicted_price_m2` | histogram | `model_version` |
| `model_info` | gauge, always 1 | `model_version` |
| `feature_drift` | gauge | `feature` |

`feature_drift` is the total variation distance between the last 500 inputs and the champion's training data, recomputed every 100 predictions with `compare_distributions` from `ml/drift.py`, the same code the pipeline uses for `drift.json`. The reference is the `training-profile.json` artifact of the run that approved the champion (its `approval_run_id` tag), so drift follows the champion after a promotion or a rollback. A model loaded from a file has no profile and publishes no drift.

### Stack

`docker compose up -d --build` starts the API, Prometheus (scrapes `api:8000/metrics` every 5 s) and Grafana. Grafana provisioning creates the Prometheus data source, the *ImmoCast API* dashboard (`monitoring/grafana/dashboards/immocast.json`), the e-mail contact point and the alert rules (`monitoring/grafana/provisioning/alerting/alerts.yml`).

### Alerts

| Alert | Condition | How to trigger it locally |
|---|---|---|
| API down | `up{job="api"}` is 0 for 1 minute | `docker compose stop api`, then `docker compose start api` |
| High error rate on /predict | More than 5 % of `/predict` requests return 5xx over 2 minutes | Set `MODEL_NAME=does-not-exist` in `.env`, `docker compose up -d api`, replay traffic (every prediction returns 503) |
| Feature drift | A feature's drift is at or above 0.20 for 2 minutes | `python scripts/replay.py --file data/v3/evaluation.csv --shifted --rate 5` |

The drift alert asks to investigate, not to retrain: retraining is decided by the performance rules of the continuous training workflow. Real DVF versions drift very little (about 0.02 between v1, v2 and v3), so the `--shifted` replay, which only sends large apartments in central arrondissements, simulates a change of clientele.

E-mails are sent through the SMTP server configured in `.env` (`SMTP_HOST` as `host:port`, `SMTP_USER`, `SMTP_PASSWORD`, `SMTP_FROM`) to `ALERT_EMAIL`. Mailtrap's free sandbox or a Gmail app password both work.

### Watch a rollback

Run the rollback workflow, then `docker compose restart api` (or `POST /reload`) and replay normal traffic. The *Model version served* and *Predictions per second by model version* panels change version, and the `prediction_made` log lines carry the new `model_version`.

## The twelve factors

| Factor | Implementation |
|---|---|
| I. Codebase | One GitHub repository. `dev`, `staging` and `main` are deployments of the same codebase at different stages. |
| II. Dependencies | Declared and pinned in `requirements.txt` (runtime) and `requirements-dev.txt` (tools). The image installs them in an isolated virtual environment. |
| III. Config | Read from environment variables: `PORT`, `MODEL_PATH`, `MLFLOW_*`, `MODEL_NAME`, `LOG_LEVEL`. Secrets live in GitHub secrets, Render environment and `.dvc/config.local`, never in Git or the image. |
| IV. Backing services | MLflow on DagsHub and the DVC remote are attached resources reached through URLs and credentials in the environment. They can be swapped without code changes. |
| V. Build, release, run | Build: `docker build` produces an image tagged with the commit. Release: that image plus the Render environment. Run: Render starts the container. A release is never modified, a new commit produces a new image. |
| VI. Processes | The app is stateless. The model is loaded from the registry at startup and the container keeps no user data. |
| VII. Port binding | waitress binds to the `PORT` given by the platform (`8000` by default). |
| VIII. Concurrency | Scaling is done by running more containers on Render. waitress serves requests with a thread pool inside each container. |
| IX. Disposability | The container starts in seconds, reports readiness through `/health` and can be stopped at any time. |
| X. Dev/prod parity | The same image is smoke-tested in Actions and deployed to Render. Developers can run it locally with Docker. |
| XI. Logs | The app writes logs to stdout. Render and `docker logs` collect them. |
| XII. Admin processes | Data preparation (`ml/prepare_data.py`) and training (`ml/train.py`) run as one-off commands from the same repository, locally or in GitHub Actions, never inside the web container. |
