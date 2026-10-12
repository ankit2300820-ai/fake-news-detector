import hashlib
import hmac
import os
import sqlite3

DB_FILE = "history.db"


def _connect():
    conn = sqlite3.connect(DB_FILE)
    conn.execute("""CREATE TABLE IF NOT EXISTS users (
        username TEXT PRIMARY KEY, salt BLOB, password_hash BLOB)""")
    return conn


def _hash(password, salt):
    return hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 100_000)


def register(username, password):
    """Returns (ok, message)."""
    username = username.strip().lower()
    if len(username) < 3 or len(password) < 6:
        return False, "Username min 3 characters and password min 6 characters."
    salt = os.urandom(16)
    try:
        with _connect() as conn:
            conn.execute("INSERT INTO users VALUES (?,?,?)", (username, salt, _hash(password, salt)))
    except sqlite3.IntegrityError:
        return False, "This username already exists."
    return True, "Account created! You can login now."


def login(username, password):
    with _connect() as conn:
        row = conn.execute("SELECT salt, password_hash FROM users WHERE username=?",
                           (username.strip().lower(),)).fetchone()
    return bool(row) and hmac.compare_digest(row[1], _hash(password, row[0]))
