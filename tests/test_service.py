import os
import subprocess
import sys
import time
from pathlib import Path

import requests


TEST_PORT = int(os.getenv("TEST_PORT", "6002"))
BASE_URL = f"http://127.0.0.1:{TEST_PORT}"
WORKDIR = Path(__file__).resolve().parent


def wait_for_health(timeout: int = 20) -> None:
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(f"{BASE_URL}/api/v1/health", timeout=2)
            if r.status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(0.5)
    raise RuntimeError("service did not become healthy in time")


def run_test() -> None:
    proc = subprocess.Popen(
        [sys.executable, "app.py"],
        cwd=str(WORKDIR),
        env={**os.environ, "PORT": str(TEST_PORT)},
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        wait_for_health()

        health = requests.get(f"{BASE_URL}/api/v1/health", timeout=5).json()
        print("health:", health)

        tokens = requests.get(f"{BASE_URL}/api/v1/tokens", timeout=5).json().get("tokens", [])
        print("token count:", len(tokens))
        if not tokens:
            raise RuntimeError("no token in pool")

        # OpenAI-compatible endpoint smoke test
        payload = {"prompt": "test image", "size": "1024x1024"}
        r = requests.post(f"{BASE_URL}/v1/images/generations", json=payload, timeout=120)
        print("generation status:", r.status_code)
        data = r.json()

        if r.status_code != 200:
            raise RuntimeError(f"generation failed: {data}")

        url = data.get("data", [{}])[0].get("url")
        if not url:
            raise RuntimeError(f"no image url in response: {data}")

        img = requests.get(f"{BASE_URL}{url}", timeout=30)
        if img.status_code != 200:
            raise RuntimeError("image fetch failed")

        print("PASS: service is usable, image endpoint works")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    run_test()
