# Data

## Source and license

- Dataset: Demandes de Valeurs Foncières géolocalisées (DVF), Etalab
- Page: https://www.data.gouv.fr/fr/datasets/demandes-de-valeurs-foncieres-geolocalisees/
- Files: `https://files.data.gouv.fr/geo-dvf/latest/csv/<YEAR>/departements/75.csv.gz`, years 2021 to 2025
- License: Licence Ouverte / Open Licence 2.0
- Snapshot downloaded on: 2026-11-23 (data.gouv.fr keeps only the last five years, so the DVC snapshots are the reference copy)

## Cleaning (`ml/prepare_data.py`)

1. Keep `nature_mutation == "Vente"`.
2. Keep sales containing exactly one `Appartement` and optionally `Dépendance` lots (cellars, parking).
3. Keep Paris postal codes 75001 to 75020 and 75116.
4. Features: `surface` (living area), `pieces` (rooms), `arrondissement` (last two digits of the postal code).
5. Target: `prix_m2 = valeur_fonciere / surface`.
6. Drop outliers: surface outside 9 to 400 m², rooms outside 1 to 10, price per m² outside 3,000 to 30,000 €.

## Versions

Each version contains `train.csv` and `evaluation.csv`. The evaluation block is always later than the training data and never used for fitting or tuning. Both the champion and the candidate are scored on the same evaluation rows.

| Version | Train | Evaluation | Train rows | Evaluation rows | Change |
|---|---|---|---|---|---|
| v1 | 2021-01 to 2022-12 | 2023-01 to 2023-06 | 63,371 | 14,194 | Baseline dataset |
| v2 | 2021-01 to 2023-12 | 2024-01 to 2024-12 | 90,105 | 24,324 | New batch of labeled sales (2023), evaluation on 2024 after the price drop |
| v3 | 2021-01 to 2024-12 | 2025-01 to 2025-12 | 114,429 | 28,214 | New batch of labeled sales (2024), evaluation on 2025 |

## Split design

- Chronological split: `max(train.date_mutation) < min(evaluation.date_mutation)`, checked in `prepare_data.py`.
- No `id_mutation` appears in both files, checked in `prepare_data.py`.
- `id_mutation` and `date_mutation` are identifiers and are not used as features.

## Identification

| Version | Git commit | DVC hash (`md5` in `data/<version>.dvc`) |
|---|---|---|
| v1 | `<commit>` | `689e331c235fbedada858e1c41c9c1c0.dir` |
| v2 | `<commit>` | `27abb8e47b593cd059795c8c1cd7a08f.dir` |
| v3 | `<commit>` | `6f87a8fb7f136f69a54a6bb323f8d545.dir` |

## Reproduce

```bash
dvc pull
```

or rebuild from the source:

```bash
python ml/prepare_data.py
```
