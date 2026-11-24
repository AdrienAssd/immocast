# Experiments and model promotion

## Primary metric

**Mean absolute error (MAE) of the price per m², in €/m². Lower is better.**

The web app shows a price estimate to a user. The MAE says, in the unit the user sees, how far the estimate is from real sales on average ("the estimate is off by about 1,900 €/m²"). It is robust to the few extreme sales left in DVF, unlike the RMSE. The mean absolute percentage error (MAPE) is logged as a secondary metric.

All metrics are computed on the held-out evaluation split of each dataset version (see `ml/README_data.md`). That split is later in time than the training data and is never used for fitting or tuning. The champion and the candidate are always scored on the same evaluation rows.

## Models

| Name | Model | Role |
|---|---|---|
| `rf` | `RandomForestRegressor`, 200 trees, `min_samples_leaf=20` | Baseline |
| `hgb` | `HistGradientBoostingRegressor`, absolute error loss, 500 iterations, learning rate 0.1, `min_samples_leaf=100` | Candidate used by continuous training |

Features: living area, number of rooms, arrondissement. Target: price per m².

## Tracked runs

`ml/train.py` logs each training run in the MLflow experiment `immocast` on DagsHub. `ml/pipeline.py` logs each continuous training run in the experiment `immocast-ct`. Each run records the Git commit, the dataset version and its DVC hash, the hyperparameters, the metrics, the model and the evaluation predictions. Continuous training runs also record the retraining and promotion decision and the registry versions before and after (`decision/decision.json`).

- MLflow tracking server and registry: https://dagshub.com/AdrienAssd/immocast.mlflow
- Production registry model: `immocast`, alias `champion`
- Development registry model: `immocast-dev`

## Baseline and candidate on v1

| Run | Model | Dataset | MAE (€/m²) | MAPE |
|---|---|---|---|---|
| Baseline | `rf` | v1 | 1,921 | 20.6 % |
| Candidate | `hgb` | v1 | 1,898 | 20.3 % |

The candidate improves the MAE by 1.2 % on the same evaluation rows. A small part of the gain comes from the absolute error loss, which optimises the metric we report. Tuning the hyperparameters further changed the MAE by less than ±10 €/m². Most of the remaining error comes from information DVF does not contain (floor, condition, view), so `hgb` is kept as the production architecture.

## Quality gates

The gates are defined in `ml/policy.py` before any evaluation.

| Gate | Rule |
|---|---|
| Quality floor | A model can be approved only if its MAE is at most **2,300 €/m²**. |
| Retraining rule | Retrain when the champion's MAE on the new evaluation split is above the quality floor, or has degraded by **at least 5 %** compared with its MAE when it was approved. |
| Promotion rule | Promote the candidate when it meets the quality floor and improves on the champion's MAE by **at least 3 %** on the same evaluation rows. |
| Bootstrap | With no champion, the first candidate becomes champion if it meets the quality floor. |

The trigger is a drop in performance rather than every new data version: retraining costs time and every promotion changes production, so it only happens when the champion no longer holds up.

## Continuous training results

All runs below were started from the `Continuous training` workflow on `main` (commit `9ef795f`), against the shared DagsHub registry and the production service on Render. Each run reloaded the production model and verified the expected version with a live prediction.

| Run | Champion MAE | Candidate MAE | Decision | Champion before → after |
|---|---|---|---|---|
| v1 | none | 1,898 (MAPE 20.3 %) | **Bootstrapped**: below the 2,300 floor | none → **1** |
| v2 | 2,102 (+10.7 % since approval at 1,898) | 2,019 (MAPE 23.6 %) | **Retrained and promoted**: improvement 3.9 % ≥ 3 % | 1 → **2** |
| v3 | 1,985 (−1.7 % since approval at 2,019) | not trained | **Champion retained**: change below the 5 % retraining threshold | 2 → **2** |
| Rollback | – | – | **Rolled back** to the previously approved version 1 | 2 → **1** |

Why the decisions differ:

- **v2 retrains.** Paris prices fell in 2023 and 2024. The v1 champion, trained on 2021–2022 sales, overestimates the 2024 evaluation block, and its MAE degrades by 10.7 %, above the 5 % threshold. The candidate trained on data up to 2023 brings the MAE down to 2,019, a 3.9 % improvement on the same rows, so it clears the 3 % promotion gate.
- **v3 keeps the champion.** On the 2025 evaluation block the v2 champion scores 1,985, slightly better than when it was approved. No retraining is needed, so production is unchanged.
- **Rollback.** Version 1 was tagged `approved=true` when it was bootstrapped, so it is a valid rollback target. The alias moved from 2 to 1 and the application served version 1 afterwards.

With a stricter promotion gate of 4 %, the v2 candidate (3.9 %) would have been rejected and version 1 kept: the gate value is a trade-off between reacting to market changes and avoiding needless production changes.

### Production evidence

| Run | GitHub Actions run | MLflow run (`immocast-ct`) | Dataset and DVC hash | Decision | Registry versions |
|---|---|---|---|---|---|
| v1 | [45371133271](https://github.com/AdrienAssd/immocast/actions/runs/45371133271) | `3458c7bfa4064c9884dc56a7a7841a60` | v1, `689e331c235fbedada858e1c41c9c1c0.dir` | `bootstrapped` | candidate 1, champion none → 1 |
| v2 | [45371997114](https://github.com/AdrienAssd/immocast/actions/runs/45371997114) | `4b27216f9bba45d4b307c31bc5a26bf0` | v2, `27abb8e47b593cd059795c8c1cd7a08f.dir` | `promoted` | candidate 2, champion 1 → 2 |
| v3 | [45372707093](https://github.com/AdrienAssd/immocast/actions/runs/45372707093) | `a61629906a8e4af38a21eeb221104ea4` | v3, `6f87a8fb7f136f69a54a6bb323f8d545.dir` | `retained` | champion 2 → 2 |
| Rollback | [45373245558](https://github.com/AdrienAssd/immocast/actions/runs/45373245558) | `75885737daff4b069a1f433be45b68ef` | – | `rolled_back` | champion 2 → 1 |

Each workflow run keeps its `ct-results-*` artifact with `decision.json`, the champion and candidate predictions on the evaluation split, and the DVC pointer of the dataset version used. The same files are logged in the MLflow run under `decision/`.

## Production model choice

The production model is the registry version selected by the `champion` alias of `immocast`. Version 2 (`hgb` trained on v2) has the best evidence: it passed the promotion gate on the 2024 evaluation block and stayed stable on the 2025 block (MAE 1,985). Version 1 was restored during the rollback demonstration. Run the rollback workflow with version 2 to put the best approved model back in production.

## Rollback

```bash
python ml/pipeline.py --rollback <model version>
```

The target must be an existing version tagged `approved=true` in the registry. Otherwise the command fails and the `champion` alias does not move. After a rollback the application must refresh its model (see `DEPLOYMENT.md`).
