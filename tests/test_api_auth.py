"""Интеграционные тесты auth-роутов: 201 / 401 / 403 / 409 / 400."""
import pytest

from backend.api.deps import require_admin, require_editor
from backend.core.constants import ROLE_ADMIN, ROLE_USER

pytestmark = pytest.mark.usefixtures("client")


async def _register(client, email="test@example.com", username="tester", password="Password1"):
    return await client.post(
        "/auth/register",
        json={"email": email, "username": username, "password": password},
    )


# --- регистрация ---

async def test_register_first_user_becomes_admin(client):
    resp = await _register(client)
    assert resp.status_code == 201
    assert resp.json()["role"] == ROLE_ADMIN


async def test_register_second_user_is_user(client):
    await _register(client)
    resp = await _register(client, email="second@example.com", username="second")
    assert resp.status_code == 201
    assert resp.json()["role"] == ROLE_USER


async def test_register_duplicate_email_409(client):
    await _register(client)
    resp = await _register(client, username="another")
    assert resp.status_code == 409


async def test_register_duplicate_username_409(client):
    await _register(client)
    resp = await _register(client, email="other@example.com")
    assert resp.status_code == 409


@pytest.mark.parametrize(
    "password",
    ["Sh0rt", "nocapital1", "NODIGITSHERE", "bad1lower", "x" * 100],
)
async def test_register_weak_password_400(client, password):
    resp = await _register(client, password=password)
    assert resp.status_code == 400


# --- логин ---

async def test_login_ok_returns_token(client):
    await _register(client)
    resp = await client.post(
        "/auth/login", json={"email": "test@example.com", "password": "Password1"}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["token_type"] == "bearer"
    assert len(body["access_token"]) > 20


async def test_login_wrong_password_401(client):
    await _register(client)
    resp = await client.post(
        "/auth/login", json={"email": "test@example.com", "password": "WrongPass1"}
    )
    assert resp.status_code == 401


async def test_login_unknown_email_401(client):
    resp = await client.post(
        "/auth/login", json={"email": "ghost@example.com", "password": "Password1"}
    )
    assert resp.status_code == 401


# --- /me и роли ---

async def test_me_with_token(client):
    await _register(client)
    token = (
        await client.post(
            "/auth/login", json={"email": "test@example.com", "password": "Password1"}
        )
    ).json()["access_token"]
    resp = await client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200
    assert resp.json()["email"] == "test@example.com"


async def test_me_without_token_401(client):
    resp = await client.get("/auth/me")
    assert resp.status_code == 401


async def test_me_with_garbage_token_401(client):
    resp = await client.get("/auth/me", headers={"Authorization": "Bearer garbage"})
    assert resp.status_code == 401


async def test_require_roles_rejects_low_role():
    class FakeUser:
        role = ROLE_USER

    with pytest.raises(Exception) as exc_info:
        await require_admin(FakeUser())
    assert getattr(exc_info.value, "status_code", None) == 403


async def test_require_roles_accepts_editor_and_admin():
    class Editor:
        role = "editor"

    class Admin:
        role = "admin"

    assert (await require_editor(Editor())).role == "editor"
    assert (await require_editor(Admin())).role == "admin"
