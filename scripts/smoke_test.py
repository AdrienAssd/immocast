import argparse
import json
import sys
import time
import urllib.error
import urllib.request

VALID_INPUT = {"surface": 45, "pieces": 2, "code_postal": "75011"}
INVALID_INPUT = {"surface": 45, "pieces": 2, "code_postal": "69001"}
PRICE_RANGE = (3000, 30000)


def request(url, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            return response.status, json.loads(response.read() or b"{}")
    except urllib.error.HTTPError as e:
        body = e.read()
        try:
            return e.code, json.loads(body or b"{}")
        except json.JSONDecodeError:
            return e.code, {}


def wait_until_ready(base, timeout):
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        try:
            status, body = request(f"{base}/health")
            last = (status, body)
            if status == 200:
                return body
        except (urllib.error.URLError, ConnectionError, TimeoutError) as e:
            last = str(e)
        time.sleep(5)
    sys.exit(f"Service not ready after {timeout}s: {last}")


def check(condition, message):
    if not condition:
        sys.exit(f"FAIL: {message}")
    print(f"OK: {message}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--expected-version")
    parser.add_argument("--expected-sha")
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()
    base = args.url.rstrip("/")

    wait_until_ready(base, args.timeout)
    print(f"OK: {base}/health returns 200")

    status, info = request(f"{base}/model-info")
    check(status == 200 and info.get("ready") is True, "/model-info reports a ready model")
    if args.expected_version:
        check(str(info.get("model_version")) == args.expected_version,
              f"loaded model version is {args.expected_version} (got {info.get('model_version')})")
    if args.expected_sha:
        check(info.get("git_sha") == args.expected_sha,
              f"deployed image is commit {args.expected_sha} (got {info.get('git_sha')})")

    status, body = request(f"{base}/predict", VALID_INPUT)
    check(status == 200, f"/predict accepts {VALID_INPUT} (status {status})")
    check(PRICE_RANGE[0] <= body.get("prix_m2", 0) <= PRICE_RANGE[1],
          f"price per m² {body.get('prix_m2')} is within {PRICE_RANGE}")
    check(body.get("prix_total", 0) > 0, f"total price {body.get('prix_total')} is returned")
    if args.expected_version:
        check(str(body.get("model_version")) == args.expected_version, "prediction made with the expected model version")

    status, body = request(f"{base}/predict", INVALID_INPUT)
    check(status in (400, 422), f"/predict rejects a non-Paris postal code (status {status})")

    print("Smoke test passed")


if __name__ == "__main__":
    main()
