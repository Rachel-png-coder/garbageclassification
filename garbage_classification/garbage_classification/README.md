# Garbage Classification -- End-to-End ML Pipeline

**African Leadership University -- Machine Learning Pipeline Summative Assignment**

An end-to-end machine learning system that classifies images of waste into
6 categories -- `cardboard, glass, metal, paper, plastic, trash` -- covering
data acquisition, preprocessing, model training/evaluation, deployment as an
API + web dashboard, bulk data upload, on-demand retraining, and load testing.

**Video Demo:** `<PASTE YOUR YOUTUBE LINK HERE>`
**Live API URL:** `<PASTE YOUR DEPLOYED API URL HERE, e.g. Render>`
**Live UI URL:** `<PASTE YOUR DEPLOYED STREAMLIT URL HERE>`

---

## 1. Project Description

The dataset is the [Garbage Classification dataset on Kaggle](https://www.kaggle.com/),
~13,900 images across 6 balanced classes, all pre-cropped to 256x256 RGB.

The model is a **MobileNetV2 transfer-learning classifier** (frozen-base warmup
followed by fine-tuning the top layers), trained with data augmentation,
dropout + L2 regularization, early stopping, and LR scheduling. See
`notebook/garbage_classification.ipynb` for the full training + evaluation
walkthrough, including 3 dataset visualizations with written interpretation
and 6 evaluation metrics (accuracy, precision, recall, F1, loss, AUC).

The trained model is served through a **FastAPI** backend (`api/main.py`)
and a **Streamlit** dashboard (`ui/app.py`) that lets a user:
- Upload a single image and get a live prediction with confidence scores
- View dataset insights/visualizations
- Bulk-upload new labeled images
- Trigger retraining on the newly uploaded data
- Monitor API uptime / model status

Both services are Dockerized, and `locustfile.py` simulates a flood of
prediction requests to measure latency/throughput at different levels of
horizontal scaling.

---

## 2. Repository Structure

```
garbage-classification/
├── README.md
├── requirements.txt
├── docker-compose.yml
├── nginx.conf                 # load balancer used when scaling API containers
├── locustfile.py              # Locust load test
├── locust_samples/            # small sample images used by Locust
│
├── notebook/
│   └── garbage_classification.ipynb   # full training + evaluation notebook
│
├── src/
│   ├── preprocessing.py       # data split + tf.data pipelines + shared image preprocessing
│   ├── database.py            # SQLite log of every uploaded retraining image
│   ├── model.py               # MobileNetV2 model build/compile/fine-tune
│   ├── prediction.py          # single-image inference (used by the API)
│   └── retrain.py             # preprocess uploads -> merge into train/ -> retrain the model
│
├── api/
│   ├── main.py                # FastAPI app (predict / upload / retrain / health / insights)
│   └── Dockerfile
│
├── ui/
│   ├── app.py                 # Streamlit dashboard
│   └── Dockerfile
│
├── data/
│   ├── train/<class>/         # training images (populate via the split script/notebook)
│   ├── test/<class>/          # held-out test images
│   ├── uploads/<class>/       # raw landing zone for bulk-uploaded retraining images
│   ├── preprocessed/<class>/  # validated/resized copies, produced before merging into train/
│   └── uploads.db             # SQLite log of every upload (see src/database.py)
│
└── models/
    └── garbage_model.h5       # trained model artifact (produced by the notebook)
```

> Note: the full dataset (~150MB) is **not** committed to this repo (large
> binary data doesn't belong in git). Only a handful of sample images are
> kept under `data/` and `locust_samples/` so the folder structure and
> Locust test work out of the box. See Section 3 to populate the full dataset.

---

## 3. Setup Instructions

### 3.1 Clone and install

```bash
git clone <YOUR_GITHUB_REPO_URL>
cd garbage-classification
python -m venv venv && source venv/bin/activate   # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### 3.2 Get the dataset and split it

1. Download the "Garbage Classification" dataset from Kaggle -- it should
   unzip into class subfolders: `cardboard/ glass/ metal/ paper/ plastic/ trash/`.
2. Split it into `data/train` and `data/test` (85/15) using the helper in
   `src/preprocessing.py`:

```bash
python -c "from src.preprocessing import split_dataset; \
  print(split_dataset('/path/to/Garbage_Dataset_Classification/images', 'data'))"
```

### 3.3 Train the model

Open `notebook/garbage_classification.ipynb` (recommended: **Google Colab**,
for a free GPU and unrestricted internet access to download ImageNet
pretrained weights) and run all cells top to bottom. It will:
- Explore + visualize the dataset (3 interpreted features)
- Build & train MobileNetV2 in two phases (frozen warmup + fine-tuning)
- Evaluate on the held-out test set with 6 metrics + confusion matrix
- Save the final model to `models/garbage_model.h5`

Copy the resulting `garbage_model.h5` into your local `models/` folder
before running the API.

### 3.4 Run the API locally

```bash
uvicorn api.main:app --reload --port 8000
```

Visit `http://localhost:8000/docs` for interactive Swagger docs.

### 3.5 Run the UI locally

```bash
streamlit run ui/app.py
```

By default it talks to `http://localhost:8000`. To point it at a deployed
API instead:

```bash
API_URL=https://your-api-url.onrender.com streamlit run ui/app.py
```

### 3.6 Run everything with Docker

```bash
docker compose up --build
```

- API: `http://localhost:8000`
- UI: `http://localhost:8501`

### 3.7 Deploy (recommended: Render)

Render was chosen for this project because it has a free tier, deploys
directly from a Dockerfile, and gives each service a public HTTPS URL --
no cloud account/billing setup needed, which keeps the deployment step
simple to reproduce for grading. (AWS/GCP work too, using the same
Dockerfiles, via ECS/Cloud Run.)

1. Push this repo to GitHub.
2. On Render: **New -> Web Service**, connect the repo, set:
   - Root Directory: repo root
   - Dockerfile Path: `api/Dockerfile`
   - Plan: Free
3. Repeat for the UI service with Dockerfile Path `ui/Dockerfile`, and set
   the `API_URL` environment variable to the deployed API's URL.
4. Once both are live, paste their URLs at the top of this README.

---

## 4. Using the App

| Action | Where |
|---|---|
| Predict one image | UI -> "Predict" tab, or `POST /predict` (multipart file) |
| See dataset insights | UI -> "Data Insights" tab, or `GET /insights` |
| Bulk-upload retraining images | UI -> "Upload & Retrain" tab, or `POST /upload` (label + files) |
| View upload audit log (DB) | `GET /uploads/log` |
| Trigger retraining | UI -> "Upload & Retrain" tab button, or `POST /retrain` |
| Check retrain progress | `GET /retrain/status` |
| Check uptime / model status | UI -> "Model Status" tab, or `GET /health`, `GET /model-info` |

---

## 5. Load Testing (Locust) -- Flood Request Simulation

```bash
pip install locust

# Baseline: 1 API container
docker compose up --build --scale api=1
locust -f locustfile.py --host http://localhost:8000 --headless -u 50 -r 5 -t 2m --csv results/1container

# Scale out to 3 API containers behind the nginx load balancer (nginx.conf)
docker compose up --build --scale api=3
locust -f locustfile.py --host http://localhost:8000 --headless -u 50 -r 5 -t 2m --csv results/3containers
```

Or use the interactive web UI: `locust -f locustfile.py --host http://localhost:8000`
then open `http://localhost:8089`.

### Results

*(Fill this in after running the tests above on your machine -- paste the
Requests/s and Average/P95 Response Time columns from each `*_stats.csv`.)*

| # API Containers | Requests/s | Avg Latency (ms) | P95 Latency (ms) | Failures |
|---|---|---|---|---|
| 1 | | | | |
| 2 | | | | |
| 3 | | | | |

**Observations:** *(e.g. "Latency stayed flat up to N concurrent users with
1 container, then degraded past that; adding containers X and Y kept P95
latency under Z ms at the same load.")*

---

## 6. Model Evaluation Summary

*(Paste the final metrics table from the notebook's Section 5 here once you've
run it, e.g.:)*

| Metric | Value |
|---|---|
| Test Accuracy | |
| Test Precision (weighted) | |
| Test Recall (weighted) | |
| Test F1-score (weighted) | |
| Test Loss | |
| Test AUC | |

---

## 7. Notes on Retraining

The retraining trigger demonstrates three explicit steps end-to-end:

1. **Upload + save to database** -- `POST /upload` saves each file to
   `data/uploads/<class>/` *and* logs a row (filename, label, path,
   timestamp) in the SQLite database `data/uploads.db` via
   `src/database.py`. Inspect the full log anytime with `GET /uploads/log`.
2. **Preprocessing the uploaded data** -- `POST /retrain` first calls
   `preprocess_uploaded_batch()` (`src/preprocessing.py`), which validates
   each pending upload (rejects corrupt files), converts to RGB, resizes
   to the model's input size, and writes cleaned copies to
   `data/preprocessed/<class>/`. Each cleaned file is marked
   `preprocessed = 1` in the database.
3. **Retraining from the custom pretrained model** -- the cleaned images
   are merged into `data/train/<class>/`, and `retrain_model()`
   (`src/retrain.py`) **warm-starts from the already-trained
   `garbage_model.h5`** (not raw ImageNet) and fine-tunes it at a low
   learning rate. The DB rows used are stamped with the retrain run's
   timestamp (`used_in_retrain_run`) so every model version has a
   traceable audit trail of exactly which uploads fed it.

The new model overwrites `models/garbage_model.h5` and the prediction
cache is cleared so the API immediately serves the new weights. A
timestamped copy of every retrained model is kept in
`models/retrain_logs/` for rollback/audit.
