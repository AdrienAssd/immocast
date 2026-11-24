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

Expected decisions, measured locally with the same code and data:

| Run | Champion MAE | Candidate MAE | Decision |
|---|---|---|---|
| v1 | none | 1,898 | **Bootstrapped**: below the 2,300 floor |
| v2 | 2,102 (+10.7 % since approval) | 2,019 | **Retrained and promoted**: improvement 3.9 % ≥ 3 % |
| v3 | 1,985 (−1.7 % since approval) | not trained | **Champion retained**: degradation below 5 % |

The v2 champion degrades because Paris prices fell in 2023 and 2024: a model trained on 2021–2022 overestimates later sales. Retraining on data up to 2023 corrects it.

### Production evidence

Fill in after running the CT workflow on the shared registry:

| Run | Workflow run | MLflow run | Decision | Champion before | Champion after |
|---|---|---|---|---|---|
| v1 | `<link>` | `<run id>` | | | |
| v2 | `<link>` | `<run id>` | | | |
| v3 | `<link>` | `<run id>` | | | |
| Rollback | `<link>` | `<run id>` | | | |

## Rollback

```bash
python ml/pipeline.py --rollback <model version>
```

The target must be an existing version tagged `approved=true` in the registry. Otherwise the command fails and the `champion` alias does not move. After a rollback the application must refresh its model (see `DEPLOYMENT.md`).
