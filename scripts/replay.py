"""Sends apartments from a dataset CSV to the API, one /predict request per row.

python scripts/replay.py --file data/v1/train.csv --shuffle      normal traffic
python scripts/replay.py --file data/v1/evaluation.csv --shifted  drifted traffic (large central flats)
python scripts/replay.py --file data/v1/evaluation.csv --broken   invalid requests (422)
"""

import argparse
import json
import time
import urllib.error
import urllib.request
import uuid
from collections import Counter

import pandas as pd

CENTRAL_ARRONDISSEMENTS = [1, 4, 6, 7, 8, 16]


def postal_code(arrondissement: int) -> str:
    return f"750{int(arrondissement):02d}"


def shift(rows: pd.DataFrame) -> pd.DataFrame:
    shifted = rows[(rows["surface"] >= 100) & rows["arrondissement"].isin(CENTRAL_ARRONDISSEMENTS)]
    if shifted.empty:
        raise SystemExit("No large central apartment in this file.")
    return shifted.sample(n=len(rows), replace=True, random_state=0)


def payloads(rows: pd.DataFrame, broken: bool):
    for row in rows.itertuples(index=False):
        payload = {
            "surface": float(row.surface),
            "pieces": int(row.pieces),
            "code_postal": postal_code(row.arrondissement),
        }
        if broken:
            payload["code_postal"] = "99999"
        yield payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True)
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--rate", type=float, default=20, help="requests per second")
    parser.add_argument("--limit", type=int, default=1000, help="number of rows to send")
    parser.add_argument("--shuffle", action="store_true", help="send the rows in random order")
    parser.add_argument("--shifted", action="store_true", help="only large apartments in central arrondissements")
    parser.add_argument("--broken", action="store_true", help="send an invalid postal code")
    args = parser.parse_args()

    rows = pd.read_csv(args.file, usecols=["surface", "pieces", "arrondissement"])
    if args.shuffle:
        rows = rows.sample(frac=1, random_state=None)
    rows = rows.head(args.limit)
    if args.shifted:
        rows = shift(rows)

    endpoint = args.url.rstrip("/") + "/predict"
    statuses = Counter()
    versions = Counter()
    for payload in payloads(rows, args.broken):
        request = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode(),
            method="POST",
            headers={"Content-Type": "application/json", "X-Request-ID": uuid.uuid4().hex[:12]},
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                statuses[response.status] += 1
                versions[json.loads(response.read()).get("model_version")] += 1
        except urllib.error.HTTPError as error:
            statuses[error.code] += 1
        except urllib.error.URLError:
            statuses["unreachable"] += 1
        time.sleep(1 / args.rate)

    print("Responses by status:", dict(statuses))
    if versions:
        print("Predictions by model version:", dict(versions))


if __name__ == "__main__":
    main()
