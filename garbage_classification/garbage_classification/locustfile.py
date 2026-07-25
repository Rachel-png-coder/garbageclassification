"""
locustfile.py
-------------
Simulates a flood of prediction requests against the deployed API to measure
latency / throughput at different scale-out levels (see docker-compose.yml).

Setup:
    pip install locust
    # put a few sample jpg images in ./locust_samples/ (one per class is fine)

Run (headless, 50 users, spawn 5/sec, 2 minutes):
    locust -f locustfile.py --host http://localhost:8000 \
           --headless -u 50 -r 5 -t 2m --csv results/run_1container

Or with the web UI:
    locust -f locustfile.py --host http://localhost:8000
    # then open http://localhost:8089

Repeat once per docker-compose --scale api=N value (1, 2, 3...) and compare
the --csv outputs' Average/P95 Response Time and Requests/s columns in your
README results table.
"""

import os
import random
import glob
from locust import HttpUser, task, between

SAMPLE_DIR = os.environ.get("LOCUST_SAMPLE_DIR", "locust_samples")
SAMPLE_IMAGES = glob.glob(os.path.join(SAMPLE_DIR, "*.jpg")) + \
                glob.glob(os.path.join(SAMPLE_DIR, "*.jpeg")) + \
                glob.glob(os.path.join(SAMPLE_DIR, "*.png"))

if not SAMPLE_IMAGES:
    raise RuntimeError(
        f"No sample images found in '{SAMPLE_DIR}/'. "
        "Copy a handful of test images there before running Locust "
        "(e.g. cp data/test/*/*.jpg -> locust_samples/, pick ~10)."
    )


class GarbageClassifierUser(HttpUser):
    wait_time = between(0.5, 2.0)  # seconds between requests per simulated user

    @task(5)
    def predict(self):
        image_path = random.choice(SAMPLE_IMAGES)
        with open(image_path, "rb") as f:
            files = {"file": (os.path.basename(image_path), f, "image/jpeg")}
            self.client.post("/predict", files=files, name="/predict")

    @task(1)
    def health_check(self):
        self.client.get("/health", name="/health")
