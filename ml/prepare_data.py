from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd

BASE_URL = "https://files.data.gouv.fr/geo-dvf/latest/csv/{year}/departements/75.csv.gz"
YEARS = [2021, 2022, 2023]
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"

TRAIN_PERIOD = ("2021-01-01", "2022-12-31")
EVALUATION_PERIOD = ("2023-01-01", "2023-06-30")

COLUMNS = ["id_mutation", "date_mutation", "surface", "pieces", "arrondissement", "prix_m2"]


def download():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    for year in YEARS:
        path = RAW_DIR / f"dvf_75_{year}.csv.gz"
        if not path.exists():
            print(f"Downloading {year}...")
            urlretrieve(BASE_URL.format(year=year), path)
        paths.append(path)
    return paths


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw[raw["nature_mutation"] == "Vente"].copy()

    types = df.groupby("id_mutation")["type_local"].agg(list)
    def is_single_apartment(values):
        values = [v for v in values if isinstance(v, str)]
        return values.count("Appartement") == 1 and all(v in ("Appartement", "Dépendance") for v in values)
    keep = types[types.apply(is_single_apartment)].index

    df = df[df["id_mutation"].isin(keep) & (df["type_local"] == "Appartement")]
    df = df.drop_duplicates("id_mutation")

    df = df.dropna(subset=["valeur_fonciere", "surface_reelle_bati", "nombre_pieces_principales", "code_postal"])
    df["code_postal"] = df["code_postal"].astype(int).astype(str)
    df = df[df["code_postal"].str.match(r"^75(0(0[1-9]|1\d|20)|116)$")]

    df["arrondissement"] = df["code_postal"].str[-2:].astype(int)
    df["surface"] = df["surface_reelle_bati"].astype(float)
    df["pieces"] = df["nombre_pieces_principales"].astype(int)
    df["prix_m2"] = df["valeur_fonciere"] / df["surface"]

    df = df[df["surface"].between(9, 400) & df["pieces"].between(1, 10) & df["prix_m2"].between(3000, 30000)]
    df["date_mutation"] = pd.to_datetime(df["date_mutation"])
    return df[COLUMNS].sort_values("date_mutation").reset_index(drop=True)


def between(df, start, end):
    return df[(df["date_mutation"] >= start) & (df["date_mutation"] <= end)]


def main():
    raw = pd.concat((pd.read_csv(p, low_memory=False) for p in download()), ignore_index=True)
    df = clean(raw)
    train = between(df, *TRAIN_PERIOD)
    evaluation = between(df, *EVALUATION_PERIOD)
    DATA_DIR.mkdir(exist_ok=True)
    train.to_csv(DATA_DIR / "train.csv", index=False)
    evaluation.to_csv(DATA_DIR / "evaluation.csv", index=False)
    print(f"train={len(train)} evaluation={len(evaluation)}")


if __name__ == "__main__":
    main()
