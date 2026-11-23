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
docker build --build-arg GIT_SHA=<sha> -t <dockerhub-user>/immocast:<sha> .
docker push <dockerhub-user>/immocast:<sha>
curl -X POST "<render-deploy-hook>&imgURL=docker.io%2F<dockerhub-user>%2Fimmocast%3A<sha>"
python scripts/smoke_test.py --url <prod-url> --expected-sha <sha>
```

Run the production container locally with a model file:

```bash
docker run --rm -p 8000:8000 -v "${PWD}/models:/app/models:ro" -e GIT_SHA=local <dockerhub-user>/immocast:<sha>
```

## Roll back the application

Redeploy a previous image: run the deploy hook with the `imgURL` of an earlier commit SHA (see *Reproduce a deployment*), or pick the previous deploy in the Render dashboard and choose *Rollback*. Then run the smoke test with that SHA.

## If the service does not become ready

Render only routes traffic to a new deploy once `/health` returns 200. If the new container never becomes ready, Render keeps serving the previous deploy and marks the new one as failed. The workflow smoke test then fails, so the problem is visible on the commit. Check the Render logs, fix the cause, or roll back.

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
