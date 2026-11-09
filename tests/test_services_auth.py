"""Юнит-тесты сервиса auth: политика паролей, bcrypt, JWT round-trip."""
import pytest

from backend.core.exceptions import InvalidCredentialsError, WeakPasswordError
from backend.services.auth import (
    create_access_token,
    decode_access_token,
    hash_password,
    validate_password,
    verify_password,
)


@pytest.mark.parametrize(
    "password",
    ["Sh0rt", "nocapitals1", "NODIGITSHERE", "bad1lower", "x" * 100],
)
def test_weak_passwords_rejected(password):
    with pytest.raises(WeakPasswordError):
        validate_password(password)


def test_valid_password_accepted():
    validate_password("Password1")  # не бросает


def test_bcrypt_roundtrip():
    hashed = hash_password("Password1")
    assert hashed != "Password1"
    assert verify_password("Password1", hashed)
    assert not verify_password("wrong", hashed)


def test_jwt_roundtrip():
    token = create_access_token(user_id=7, role="admin")
    payload = decode_access_token(token)
    assert payload["sub"] == "7"
    assert payload["role"] == "admin"


def test_jwt_garbage_rejected():
    with pytest.raises(InvalidCredentialsError):
        decode_access_token("not-a-token")
