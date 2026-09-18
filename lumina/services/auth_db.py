import os
import sqlite3
import secrets
import time
from pathlib import Path

DATABASE_URL = os.environ.get("DATABASE_URL", "").strip()
DB_PATH = os.environ.get(
    "LUMINA_AUTH_DB",
    str(Path(__file__).resolve().parents[2] / "data" / "auth.db"),
)

if DATABASE_URL:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    from psycopg2 import IntegrityError as PostgresIntegrityError


def _is_postgres():
    return bool(DATABASE_URL)


def _connect():
    if _is_postgres():
        return psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)

    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


def _execute(conn, query, params=()):
    if _is_postgres():
        query = query.replace("?", "%s")
        return conn.cursor().execute(query, params)
    return conn.execute(query, params)


def init_db():
    conn = _connect()
    try:
        if _is_postgres():
            _execute(conn, """
                CREATE TABLE IF NOT EXISTS users (
                    id BIGSERIAL PRIMARY KEY,
                    first_name TEXT NOT NULL,
                    last_name TEXT NOT NULL,
                    email TEXT NOT NULL UNIQUE,
                    username TEXT NOT NULL UNIQUE,
                    salt TEXT NOT NULL,
                    password_hash TEXT NOT NULL,
                    created_at DOUBLE PRECISION NOT NULL
                )
            """)
            _execute(conn, """
                CREATE UNIQUE INDEX IF NOT EXISTS users_email_lower_idx
                ON users (LOWER(email))
            """)
            _execute(conn, """
                CREATE UNIQUE INDEX IF NOT EXISTS users_username_lower_idx
                ON users (LOWER(username))
            """)
            _execute(conn, """
                CREATE TABLE IF NOT EXISTS otps (
                    email TEXT PRIMARY KEY,
                    code TEXT NOT NULL,
                    expires_at DOUBLE PRECISION NOT NULL,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    sent_at DOUBLE PRECISION NOT NULL
                )
            """)
            _execute(conn, """
                CREATE UNIQUE INDEX IF NOT EXISTS otps_email_lower_idx
                ON otps (LOWER(email))
            """)
        else:
            _execute(conn, """
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
            _execute(conn, """
                CREATE TABLE IF NOT EXISTS otps (
                    email TEXT PRIMARY KEY COLLATE NOCASE,
                    code TEXT NOT NULL,
                    expires_at REAL NOT NULL,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    sent_at REAL NOT NULL
                )
            """)

        conn.commit()
    finally:
        conn.close()


def create_user(first_name, last_name, email, username, salt, password_hash):
    init_db()
    conn = _connect()
    try:
        _execute(
            conn,
            "INSERT INTO users(first_name,last_name,email,username,salt,password_hash,created_at) VALUES(?,?,?,?,?,?,?)",
            (first_name, last_name, email.lower(), username, salt, password_hash, time.time()),
        )
        conn.commit()
        return True
    except (sqlite3.IntegrityError, PostgresIntegrityError if _is_postgres() else sqlite3.IntegrityError):
        conn.rollback()
        return False
    finally:
        conn.close()


def get_user(identifier):
    init_db()
    conn = _connect()
    try:
        if _is_postgres():
            cur = conn.cursor()
            cur.execute(
                """
                SELECT id,first_name,last_name,email,username,salt,password_hash,created_at
                FROM users
                WHERE LOWER(email) = LOWER(%s) OR LOWER(username) = LOWER(%s)
                """,
                (identifier, identifier),
            )
            row = cur.fetchone()
        else:
            row = _execute(
                conn,
                """
                SELECT id,first_name,last_name,email,username,salt,password_hash,created_at
                FROM users
                WHERE email = ? COLLATE NOCASE OR username = ? COLLATE NOCASE
                """,
                (identifier, identifier),
            ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def save_otp(email, code, ttl=300):
    init_db()
    conn = _connect()
    try:
        if _is_postgres():
            _execute(
                conn,
                """
                INSERT INTO otps(email,code,expires_at,attempts,sent_at)
                VALUES(?,?,?,?,?)
                ON CONFLICT (email) DO UPDATE SET
                    code=EXCLUDED.code,
                    expires_at=EXCLUDED.expires_at,
                    attempts=0,
                    sent_at=EXCLUDED.sent_at
                """,
                (email.lower(), code, time.time() + ttl, 0, time.time()),
            )
        else:
            _execute(
                conn,
                """
                INSERT INTO otps(email,code,expires_at,attempts,sent_at)
                VALUES(?,?,?,?,?)
                ON CONFLICT(email) DO UPDATE SET
                    code=excluded.code,
                    expires_at=excluded.expires_at,
                    attempts=0,
                    sent_at=excluded.sent_at
                """,
                (email.lower(), code, time.time() + ttl, 0, time.time()),
            )
        conn.commit()
    finally:
        conn.close()


def verify_otp(email, code, max_attempts=5):
    init_db()
    conn = _connect()
    try:
        row = _execute(
            conn,
            "SELECT code,expires_at,attempts FROM otps WHERE email=?",
            (email,),
        ).fetchone()

        if not row:
            return False, "No OTP found"

        if time.time() > row["expires_at"]:
            _execute(conn, "DELETE FROM otps WHERE email=?", (email,))
            conn.commit()
            return False, "OTP expired"

        if row["attempts"] >= max_attempts:
            _execute(conn, "DELETE FROM otps WHERE email=?", (email,))
            conn.commit()
            return False, "Too many OTP attempts"

        if not secrets.compare_digest(str(row["code"]), str(code or "")):
            _execute(
                conn,
                "UPDATE otps SET attempts=attempts+1 WHERE email=?",
                (email,),
            )
            conn.commit()
            return False, "Invalid OTP"

        _execute(conn, "DELETE FROM otps WHERE email=?", (email,))
        conn.commit()
        return True, None
    finally:
        conn.close()


def update_password(identifier, salt, password_hash):
    init_db()
    conn = _connect()
    try:
        cur = _execute(
            conn,
            """
            UPDATE users
            SET salt=?, password_hash=?
            WHERE LOWER(email)=LOWER(?) OR LOWER(username)=LOWER(?)
            """,
            (salt, password_hash, identifier, identifier),
        )
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()


def otp_can_send(email, cooldown=60):
    init_db()
    conn = _connect()
    try:
        row = _execute(
            conn,
            "SELECT sent_at FROM otps WHERE LOWER(email)=LOWER(?)",
            (email,),
        ).fetchone()
        return not row or time.time() - row["sent_at"] >= cooldown
    finally:
        conn.close()
