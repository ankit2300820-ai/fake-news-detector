import sqlite3
from datetime import datetime

import pandas as pd

DB_FILE = "history.db"


def _connect():
    conn = sqlite3.connect(DB_FILE)
    conn.execute("""CREATE TABLE IF NOT EXISTS history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT, news TEXT, mode TEXT, model TEXT,
        prediction TEXT, confidence REAL, created_at TEXT, feedback TEXT)""")
    cols = [r[1] for r in conn.execute("PRAGMA table_info(history)")]
    for col in ("username", "feedback"):   # upgrade an older database
        if col not in cols:
            conn.execute(f"ALTER TABLE history ADD COLUMN {col} TEXT")
    return conn


def save_prediction(username, news, mode, model, prediction, confidence):
    with _connect() as conn:
        cur = conn.execute(
            "INSERT INTO history (username, news, mode, model, prediction, confidence, created_at) VALUES (?,?,?,?,?,?,?)",
            (username, news, mode, model, prediction, confidence, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    return cur.lastrowid


def set_feedback(row_id, value):
    with _connect() as conn:
        conn.execute("UPDATE history SET feedback=? WHERE id=?", (value, row_id))


def get_history(username):
    conn = _connect()
    df = pd.read_sql_query("SELECT * FROM history WHERE username=? ORDER BY id DESC", conn, params=(username,))
    conn.close()
    return df


def clear_history(username):
    with _connect() as conn:
        conn.execute("DELETE FROM history WHERE username=?", (username,))
