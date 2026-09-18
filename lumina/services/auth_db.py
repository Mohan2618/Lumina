import os
import sqlite3
import secrets
import time
from pathlib import Path

DB_PATH = os.environ.get("LUMINA_AUTH_DB", str(Path(__file__).resolve().parents[2] / "data" / "auth.db"))

def _connect():
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = _connect()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            first_name TEXT NOT NULL,
            last_name TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE COLLATE NOCASE,
            username TEXT NOT NULL UNIQUE COLLATE NOCASE,
            salt TEXT NOT NULL,
            password_hash TEXT NOT NULL,
            created_at REAL NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS otps (
            email TEXT PRIMARY KEY COLLATE NOCASE,
            code TEXT NOT NULL,
            expires_at REAL NOT NULL,
            attempts INTEGER NOT NULL DEFAULT 0,
            sent_at REAL NOT NULL
        )
    """)
    conn.commit()
    conn.close()

def create_user(first_name, last_name, email, username, salt, password_hash):
    init_db()
    conn = _connect()
    try:
        conn.execute(
            "INSERT INTO users(first_name,last_name,email,username,salt,password_hash,created_at) VALUES(?,?,?,?,?,?,?)",
            (first_name, last_name, email.lower(), username, salt, password_hash, time.time()),
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def get_user(identifier):
    init_db()
    conn = _connect()
    row = conn.execute(
        "SELECT id,first_name,last_name,email,username,salt,password_hash,created_at FROM users WHERE email = ? COLLATE NOCASE OR username = ? COLLATE NOCASE",
        (identifier, identifier),
    ).fetchone()
    conn.close()
    return dict(row) if row else None

def save_otp(email, code, ttl=300):
    init_db()
    conn = _connect()
    conn.execute(
        "INSERT INTO otps(email,code,expires_at,attempts,sent_at) VALUES(?,?,?,?,?) "
        "ON CONFLICT(email) DO UPDATE SET code=excluded.code,expires_at=excluded.expires_at,attempts=0,sent_at=excluded.sent_at",
        (email.lower(), code, time.time() + ttl, 0, time.time()),
    )
    conn.commit()
    conn.close()

def verify_otp(email, code, max_attempts=5):
    init_db()
    conn = _connect()
    row = conn.execute("SELECT code,expires_at,attempts FROM otps WHERE email=? COLLATE NOCASE", (email,)).fetchone()
    if not row:
        conn.close()
        return False, "No OTP found"
    if time.time() > row["expires_at"]:
        conn.execute("DELETE FROM otps WHERE email=? COLLATE NOCASE", (email,))
        conn.commit()
        conn.close()
        return False, "OTP expired"
    if row["attempts"] >= max_attempts:
        conn.execute("DELETE FROM otps WHERE email=? COLLATE NOCASE", (email,))
        conn.commit()
        conn.close()
        return False, "Too many OTP attempts"
    if not secrets.compare_digest(str(row["code"]), str(code or "")):
        conn.execute("UPDATE otps SET attempts=attempts+1 WHERE email=? COLLATE NOCASE", (email,))
        conn.commit()
        conn.close()
        return False, "Invalid OTP"
    conn.execute("DELETE FROM otps WHERE email=? COLLATE NOCASE", (email,))
    conn.commit()
    conn.close()
    return True, None
