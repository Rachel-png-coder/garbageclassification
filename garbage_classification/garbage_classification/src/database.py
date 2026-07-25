"""
database.py
-----------
Tiny SQLite layer that logs every uploaded retraining image as a database
record (filename, label, storage path, upload timestamp, preprocessing
status). This is what the "/upload" endpoint writes to and what
"/retrain" reads from -- satisfies the "Data file uploading + saving to
database" requirement explicitly, on top of the raw files themselves
living under data/uploads/<class>/.

We use SQLite (stdlib `sqlite3`, zero extra services to run) rather than
Postgres/MySQL so the whole project still runs with `docker compose up`
and no external DB dependency -- swap the connection string for a real
DB in a production deployment if needed.
"""

import sqlite3
from pathlib import Path
from datetime import datetime, timezone

DB_PATH = Path("data/uploads.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS uploads (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT NOT NULL,
    label TEXT NOT NULL,
    file_path TEXT NOT NULL,
    uploaded_at TEXT NOT NULL,
    preprocessed INTEGER NOT NULL DEFAULT 0,
    used_in_retrain_run TEXT
);
"""


def get_connection(db_path: str = None):
    db_path = Path(db_path) if db_path else DB_PATH
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute(SCHEMA)
    conn.commit()
    return conn


def log_upload(filename: str, label: str, file_path: str, db_path: str = None):
    """Called by the API's /upload endpoint for every saved file."""
    conn = get_connection(db_path)
    conn.execute(
        "INSERT INTO uploads (filename, label, file_path, uploaded_at, preprocessed) "
        "VALUES (?, ?, ?, ?, 0)",
        (filename, label, str(file_path), datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()
    conn.close()


def mark_preprocessed(file_path: str, db_path: str = None):
    conn = get_connection(db_path)
    conn.execute("UPDATE uploads SET preprocessed = 1 WHERE file_path = ?", (str(file_path),))
    conn.commit()
    conn.close()


def mark_used_in_retrain(file_paths, run_timestamp: str, db_path: str = None):
    conn = get_connection(db_path)
    conn.executemany(
        "UPDATE uploads SET used_in_retrain_run = ? WHERE file_path = ?",
        [(run_timestamp, str(p)) for p in file_paths],
    )
    conn.commit()
    conn.close()


def get_pending_uploads(db_path: str = None):
    """Uploads that haven't been folded into a retraining run yet."""
    conn = get_connection(db_path)
    rows = conn.execute(
        "SELECT id, filename, label, file_path, uploaded_at, preprocessed "
        "FROM uploads WHERE used_in_retrain_run IS NULL"
    ).fetchall()
    conn.close()
    columns = ["id", "filename", "label", "file_path", "uploaded_at", "preprocessed"]
    return [dict(zip(columns, row)) for row in rows]


def get_all_uploads(db_path: str = None):
    conn = get_connection(db_path)
    rows = conn.execute(
        "SELECT id, filename, label, file_path, uploaded_at, preprocessed, used_in_retrain_run "
        "FROM uploads ORDER BY uploaded_at DESC"
    ).fetchall()
    conn.close()
    columns = ["id", "filename", "label", "file_path", "uploaded_at", "preprocessed", "used_in_retrain_run"]
    return [dict(zip(columns, row)) for row in rows]
