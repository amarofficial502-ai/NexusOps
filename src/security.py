
import base64
import hashlib
import hmac
import logging
import os
import sqlite3
import time
from pathlib import Path

import bcrypt
import pyotp
import streamlit as st
from cryptography.fernet import Fernet, InvalidToken


# =========================================================
# 1. CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "nexusops_security.db"

MAX_FAILED_ATTEMPTS = 5
LOCKOUT_SECONDS = 300
SESSION_SECONDS = 1800

ALLOWED_ROLES = {"Admin", "Manager", "Viewer"}


# =========================================================
# 2. ENCRYPTION
# =========================================================

def _fernet():
    """Create an encryption instance from the environment key."""
    raw_key = os.environ.get("NEXUSOPS_ENCRYPTION_KEY", "")

    if len(raw_key) < 32:
        raise RuntimeError(
            "NEXUSOPS_ENCRYPTION_KEY is missing or too short."
        )

    derived_key = hashlib.sha256(
        raw_key.encode("utf-8")
    ).digest()

    return Fernet(base64.urlsafe_b64encode(derived_key))


# =========================================================
# 3. DATABASE
# =========================================================

def _connect():
    """Open the security database."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(DB_PATH, timeout=10)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")

    return connection


def initialize_security_db():
    """Create security tables and add compatible missing columns."""
    with _connect() as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS users (
                username TEXT PRIMARY KEY,
                password_hash BLOB NOT NULL,
                totp_secret_enc TEXT NOT NULL,
                role TEXT NOT NULL
                    CHECK(role IN ('Admin', 'Manager', 'Viewer')),
                active INTEGER NOT NULL DEFAULT 0,
                failed_attempts INTEGER NOT NULL DEFAULT 0,
                locked_until REAL NOT NULL DEFAULT 0,
                created_at REAL NOT NULL
            )
        """)

        # Preserve compatibility with existing audit table schemas.
        connection.execute("""
            CREATE TABLE IF NOT EXISTS security_audit (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                event TEXT NOT NULL,
                created_at REAL NOT NULL
            )
        """)

        user_columns = {
            row["name"]
            for row in connection.execute(
                "PRAGMA table_info(users)"
            ).fetchall()
        }

        if "active" not in user_columns:
            connection.execute(
                "ALTER TABLE users "
                "ADD COLUMN active INTEGER NOT NULL DEFAULT 0"
            )

        if "failed_attempts" not in user_columns:
            connection.execute(
                "ALTER TABLE users "
                "ADD COLUMN failed_attempts INTEGER NOT NULL DEFAULT 0"
            )

        if "locked_until" not in user_columns:
            connection.execute(
                "ALTER TABLE users "
                "ADD COLUMN locked_until REAL NOT NULL DEFAULT 0"
            )

        connection.commit()


# =========================================================
# 4. AUDIT LOGGING
# =========================================================

def _audit(username, event):
    """Write an audit event using the existing table's timestamp column."""
    initialize_security_db()

    with _connect() as connection:
        columns = {
            row["name"]
            for row in connection.execute(
                "PRAGMA table_info(security_audit)"
            ).fetchall()
        }

        # Older databases may use created_at; newer ones may use occurred_at.
        if "occurred_at" in columns:
            timestamp_column = "occurred_at"
        elif "created_at" in columns:
            timestamp_column = "created_at"
        else:
            raise RuntimeError(
                "security_audit has no recognized timestamp column."
            )

        # The column name is selected only from these fixed values.
        connection.execute(
            f"""
            INSERT INTO security_audit
                (username, event, {timestamp_column})
            VALUES (?, ?, ?)
            """,
            (str(username)[:80], str(event)[:100], time.time()),
        )
        connection.commit()


# =========================================================
# 5. USER LOOKUP
# =========================================================

def _get_user(username):
    """Return an active user record."""
    initialize_security_db()
    username = username.strip().lower()

    with _connect() as connection:
        return connection.execute(
            """
            SELECT *
            FROM users
            WHERE username = ? AND active = 1
            """,
            (username,),
        ).fetchone()


# =========================================================
# 6. USER CREATION
# =========================================================

def create_user(username, password, role="Viewer"):
    """Create an inactive user requiring MFA activation."""
    username = username.strip().lower()

    if not username or len(username) > 80:
        raise ValueError("Username must contain 1-80 characters.")

    if not isinstance(password, str) or len(password) < 12:
        raise ValueError("Password must contain at least 12 characters.")

    password_bytes = password.encode("utf-8")

    if len(password_bytes) > 72:
        raise ValueError("Password exceeds bcrypt's 72-byte limit.")

    if role not in ALLOWED_ROLES:
        raise ValueError("Invalid role.")

    initialize_security_db()

    password_hash = bcrypt.hashpw(
        password_bytes,
        bcrypt.gensalt(rounds=12),
    )

    secret = pyotp.random_base32()

    encrypted_secret = _fernet().encrypt(
        secret.encode("utf-8")
    ).decode("ascii")

    try:
        with _connect() as connection:
            connection.execute(
                """
                INSERT INTO users (
                    username, password_hash, totp_secret_enc,
                    role, active, failed_attempts, locked_until,
                    created_at
                )
                VALUES (?, ?, ?, ?, 0, 0, 0, ?)
                """,
                (
                    username,
                    password_hash,
                    encrypted_secret,
                    role,
                    time.time(),
                ),
            )
            connection.commit()

    except sqlite3.IntegrityError as exc:
        raise ValueError("Username already exists.") from exc

    _audit(username, "account_created_pending_mfa")
    return secret


# =========================================================
# 7. MFA ACTIVATION
# =========================================================

def activate_user_after_mfa(username, secret, code):
    """Activate a pending account after validating TOTP."""
    username = username.strip().lower()

    if (
        not isinstance(code, str)
        or not code.isdigit()
        or len(code) != 6
    ):
        _audit(username, "mfa_activation_failed")
        return False

    initialize_security_db()

    with _connect() as connection:
        user = connection.execute(
            """
            SELECT totp_secret_enc, active
            FROM users
            WHERE username = ?
            """,
            (username,),
        ).fetchone()

    if user is None or user["active"]:
        _audit(username, "mfa_activation_rejected")
        return False

    try:
        stored_secret = _fernet().decrypt(
            user["totp_secret_enc"].encode("ascii")
        ).decode("utf-8")
    except (InvalidToken, ValueError):
        _audit(username, "mfa_secret_decryption_failed")
        return False

    if not hmac.compare_digest(stored_secret, secret):
        _audit(username, "mfa_secret_mismatch")
        return False

    if not pyotp.TOTP(stored_secret).verify(
        code,
        valid_window=1,
    ):
        _audit(username, "mfa_activation_failed")
        return False

    with _connect() as connection:
        connection.execute(
            """
            UPDATE users
            SET active = 1, failed_attempts = 0, locked_until = 0
            WHERE username = ? AND active = 0
            """,
            (username,),
        )
        connection.commit()

    _audit(username, "account_activated")
    return True


# =========================================================
# 8. FAILED-LOGIN TRACKING
# =========================================================

def _record_login_failure(username, event):
    """Count failed logins and lock an account at the configured limit."""
    now = time.time()
    initialize_security_db()

    with _connect() as connection:
        connection.execute("BEGIN IMMEDIATE")

        user = connection.execute(
            """
            SELECT failed_attempts, locked_until
            FROM users
            WHERE username = ? AND active = 1
            """,
            (username,),
        ).fetchone()

        if user is None:
            connection.commit()
            _audit(username, event)
            return

        if user["locked_until"] > now:
            connection.commit()
            _audit(username, "login_locked")
            return

        attempts = user["failed_attempts"] + 1

        lock_until = (
            now + LOCKOUT_SECONDS
            if attempts >= MAX_FAILED_ATTEMPTS
            else 0
        )

        connection.execute(
            """
            UPDATE users
            SET failed_attempts = ?, locked_until = ?
            WHERE username = ? AND active = 1
            """,
            (attempts, lock_until, username),
        )
        connection.commit()

    _audit(
        username,
        "account_locked" if lock_until else event,
    )


# =========================================================
# 9. AUTHENTICATION
# =========================================================

def authenticate(username, password, otp):
    """Authenticate with password and TOTP plus temporary lockout."""
    username = username.strip().lower()

    user = _get_user(username)

    if user and user["locked_until"] > time.time():
        _audit(username, "login_locked")
        return False

    # Perform a bcrypt check for unknown users as well.
    dummy_hash = bcrypt.hashpw(
        b"dummy-password-for-timing",
        bcrypt.gensalt(rounds=12),
    )

    password_bytes = (
        password.encode("utf-8")
        if isinstance(password, str)
        else b""
    )

    # Reject overlong passwords rather than silently truncating them.
    password_ok = False
    if len(password_bytes) <= 72:
        stored_hash = (
            bytes(user["password_hash"])
            if user
            else dummy_hash
        )
        password_ok = bcrypt.checkpw(
            password_bytes,
            stored_hash,
        )
    else:
        bcrypt.checkpw(b"invalid-password", dummy_hash)

    if user is None or not password_ok:
        _record_login_failure(username, "login_failed")
        return False

    try:
        secret = _fernet().decrypt(
            user["totp_secret_enc"].encode("ascii")
        ).decode("utf-8")

        otp_ok = (
            isinstance(otp, str)
            and otp.isdigit()
            and len(otp) == 6
            and pyotp.TOTP(secret).verify(
                otp,
                valid_window=1,
            )
        )

    except (InvalidToken, ValueError):
        _record_login_failure(
            username,
            "mfa_verification_error",
        )
        return False

    if not otp_ok:
        _record_login_failure(username, "mfa_failed")
        return False

    with _connect() as connection:
        connection.execute(
            """
            UPDATE users
            SET failed_attempts = 0, locked_until = 0
            WHERE username = ?
            """,
            (username,),
        )
        connection.commit()

    _audit(username, "login_success")
    return True


# =========================================================
# 10. ROLE CHECKS
# =========================================================

def has_role(*allowed_roles):
    """Check whether the logged-in user has an allowed role."""
    return (
        st.session_state.get("authenticated", False)
        and st.session_state.get("role") in allowed_roles
    )


# =========================================================
# 11. LOGOUT
# =========================================================

def logout():
    """Clear the Streamlit session and record logout."""
    username = st.session_state.get("username", "unknown")
    _audit(username, "logout")

    for key in list(st.session_state.keys()):
        del st.session_state[key]

    st.rerun()


# =========================================================
# 12. LOGIN UI AND SESSION TIMEOUT
# =========================================================

def require_login():
    """Require authentication before showing protected content."""
    if not st.session_state.get("authenticated", False):
        st.title("NexusOps Secure Login")
        st.caption("Authorized users only")

        with st.form("nexusops_login"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            otp = st.text_input(
                "6-digit authenticator code",
                max_chars=6,
            )
            submitted = st.form_submit_button("Sign in")

        if submitted:
            try:
                if authenticate(username, password, otp):
                    user = _get_user(username)

                    if user is None:
                        st.error("Login failed. Please try again.")
                        st.stop()

                    st.session_state["authenticated"] = True
                    st.session_state["username"] = user["username"]
                    st.session_state["role"] = user["role"]
                    st.session_state["login_time"] = time.time()

                    st.rerun()

                st.error(
                    "Login failed. Check credentials or try later."
                )

            except Exception:
                logging.exception("NexusOps secure login error")
                st.error(
                    "Secure login is unavailable. "
                    "Check server configuration or contact the administrator."
                )

        st.stop()

    login_time = st.session_state.get("login_time", 0)

    if (
        not login_time
        or time.time() - login_time > SESSION_SECONDS
    ):
        logout()
