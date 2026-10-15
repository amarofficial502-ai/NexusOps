
import time

import pyotp
import pytest

import src.security as security


TEST_PASSWORD = "Test-Password-2026!"
TEST_USERNAME = "test_admin"


@pytest.fixture
def isolated_security_db(tmp_path, monkeypatch):
    """Use a separate temporary database for every test."""
    monkeypatch.setenv(
        "NEXUSOPS_ENCRYPTION_KEY",
        "test-only-key-for-nexusops-unit-tests-2026",
    )

    monkeypatch.setattr(
        security,
        "DATA_DIR",
        tmp_path / "data",
    )
    monkeypatch.setattr(
        security,
        "DB_PATH",
        tmp_path / "data" / "test_security.db",
    )

    security.initialize_security_db()
    return security.DB_PATH


def create_active_user(username=TEST_USERNAME, role="Admin"):
    """Create a user and activate it using a valid TOTP code."""
    secret = security.create_user(
        username,
        TEST_PASSWORD,
        role=role,
    )

    current_code = pyotp.TOTP(secret).now()

    activated = security.activate_user_after_mfa(
        username,
        secret,
        current_code,
    )

    assert activated is True
    return secret


def get_user_record(username=TEST_USERNAME):
    """Read a user record from the isolated test database."""
    with security._connect() as connection:
        return connection.execute(
            "SELECT * FROM users WHERE username = ?",
            (username.lower(),),
        ).fetchone()


def get_wrong_totp(secret):
    """Return a six-digit code that is not currently valid."""
    totp = pyotp.TOTP(secret)

    for number in range(1_000_000):
        candidate = f"{number:06d}"
        if not totp.verify(candidate, valid_window=1):
            return candidate

    raise AssertionError("Could not generate an invalid test code.")


def test_new_user_is_inactive_until_mfa_verified(
    isolated_security_db,
):
    secret = security.create_user(
        TEST_USERNAME,
        TEST_PASSWORD,
        role="Admin",
    )

    user = get_user_record()

    assert user is not None
    assert user["active"] == 0
    assert isinstance(secret, str)
    assert len(secret) > 0


def test_wrong_mfa_code_does_not_activate_user(
    isolated_security_db,
):
    secret = security.create_user(
        TEST_USERNAME,
        TEST_PASSWORD,
        role="Admin",
    )

    wrong_code = get_wrong_totp(secret)

    result = security.activate_user_after_mfa(
        TEST_USERNAME,
        secret,
        wrong_code,
    )

    assert result is False
    assert get_user_record()["active"] == 0


def test_wrong_secret_does_not_activate_user(
    isolated_security_db,
):
    secret = security.create_user(
        TEST_USERNAME,
        TEST_PASSWORD,
        role="Admin",
    )

    wrong_secret = pyotp.random_base32()
    code = pyotp.TOTP(wrong_secret).now()

    result = security.activate_user_after_mfa(
        TEST_USERNAME,
        wrong_secret,
        code,
    )

    assert result is False
    assert get_user_record()["active"] == 0


def test_login_requires_password_and_valid_otp(
    isolated_security_db,
):
    secret = create_active_user()
    valid_code = pyotp.TOTP(secret).now()

    assert security.authenticate(
        TEST_USERNAME,
        "Definitely-Wrong-Password!",
        valid_code,
    ) is False

    # A wrong password must be recorded.
    assert get_user_record()["failed_attempts"] == 1

    # Use an invalid OTP with the correct password.
    wrong_code = get_wrong_totp(secret)

    assert security.authenticate(
        TEST_USERNAME,
        TEST_PASSWORD,
        wrong_code,
    ) is False

    # Both password and MFA failures must be counted.
    assert get_user_record()["failed_attempts"] == 2

    # A successful login resets the counter.
    assert security.authenticate(
        TEST_USERNAME,
        TEST_PASSWORD,
        pyotp.TOTP(secret).now(),
    ) is True

    user = get_user_record()
    assert user["failed_attempts"] == 0
    assert user["locked_until"] == 0


def test_duplicate_username_is_rejected(
    isolated_security_db,
):
    security.create_user(
        TEST_USERNAME,
        TEST_PASSWORD,
        role="Admin",
    )

    with pytest.raises(ValueError, match="already exists"):
        security.create_user(
            TEST_USERNAME,
            TEST_PASSWORD,
            role="Admin",
        )


def test_short_password_is_rejected(
    isolated_security_db,
):
    with pytest.raises(ValueError, match="at least 12"):
        security.create_user(
            "short_password_user",
            "tiny",
            role="Viewer",
        )


def test_invalid_role_is_rejected(
    isolated_security_db,
):
    with pytest.raises(ValueError, match="Invalid role"):
        security.create_user(
            "invalid_role_user",
            TEST_PASSWORD,
            role="SuperAdmin",
        )


def test_five_failed_attempts_lock_account(
    isolated_security_db,
):
    secret = create_active_user()
    wrong_code = get_wrong_totp(secret)

    for expected_attempts in range(1, security.MAX_FAILED_ATTEMPTS + 1):
        result = security.authenticate(
            TEST_USERNAME,
            "Definitely-Wrong-Password!",
            wrong_code,
        )

        assert result is False

        user = get_user_record()
        assert user["failed_attempts"] == expected_attempts

    user = get_user_record()

    assert user["locked_until"] > time.time()
    assert user["locked_until"] <= time.time() + security.LOCKOUT_SECONDS + 2


def test_locked_account_rejects_correct_credentials(
    isolated_security_db,
    monkeypatch,
):
    secret = create_active_user()
    wrong_code = get_wrong_totp(secret)

    for _ in range(security.MAX_FAILED_ATTEMPTS):
        assert security.authenticate(
            TEST_USERNAME,
            "Definitely-Wrong-Password!",
            wrong_code,
        ) is False

    # The account must be locked even if the credentials are correct.
    assert security.authenticate(
        TEST_USERNAME,
        TEST_PASSWORD,
        pyotp.TOTP(secret).now(),
    ) is False

    user = get_user_record()
    assert user["failed_attempts"] >= security.MAX_FAILED_ATTEMPTS
    assert user["locked_until"] > time.time()


def test_expired_lock_allows_login_again(
    isolated_security_db,
    monkeypatch,
):
    secret = create_active_user()
    wrong_code = get_wrong_totp(secret)

    for _ in range(security.MAX_FAILED_ATTEMPTS):
        assert security.authenticate(
            TEST_USERNAME,
            "Definitely-Wrong-Password!",
            wrong_code,
        ) is False

    # Simulate the end of the lockout period without waiting five minutes.
    with security._connect() as connection:
        connection.execute(
            """
            UPDATE users
            SET locked_until = ?, failed_attempts = ?
            WHERE username = ?
            """,
            (time.time() - 1, 0, TEST_USERNAME),
        )
        connection.commit()

    result = security.authenticate(
        TEST_USERNAME,
        TEST_PASSWORD,
        pyotp.TOTP(secret).now(),
    )

    assert result is True

    user = get_user_record()
    assert user["failed_attempts"] == 0
    assert user["locked_until"] == 0
