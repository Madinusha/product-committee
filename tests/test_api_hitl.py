"""Интеграционные тесты HITL (этап 10): очередь, роли, атомарный confirm, active learning."""
from sqlalchemy import select

from backend.data.importers import case_id
from backend.db.models import DecisionMemory

ENTRY = {
    "situation": "Отток 8%/мес, цель 5%, команда 2 человека, срок 1 месяц",
    "decision": (
        "Снизить отток с 8% до 5% за 1 месяц (отвечает PM): retention +2.5 п.п., "
        "замер - когортный анализ."
    ),
    "domain": "b2b_saas",
    "critic_score": 8.0,
}


async def _users(client) -> dict[str, str]:
    """Первый пользователь - admin, второй - обычный user."""
    await client.post(
        "/auth/register",
        json={"email": "admin@example.com", "username": "admin", "password": "Password1"},
    )
    await client.post(
        "/auth/register",
        json={"email": "user@example.com", "username": "user", "password": "Password1"},
    )
    tokens = {}
    for email in ("admin@example.com", "user@example.com"):
        resp = await client.post(
            "/auth/login", json={"email": email, "password": "Password1"}
        )
        tokens[email.split("@")[0]] = {"Authorization": f"Bearer {resp.json()['access_token']}"}
    return tokens


async def _create_entry(client, headers) -> int:
    resp = await client.post("/hitl", json=ENTRY, headers=headers)
    assert resp.status_code == 201
    return resp.json()["id"]


async def test_create_entry_requires_auth_only(client):
    headers = await _users(client)
    resp = await client.post("/hitl", json=ENTRY)
    assert resp.status_code == 401
    resp = await client.post("/hitl", json=ENTRY, headers=headers["user"])
    assert resp.status_code == 201  # обычный user может отправить решение на проверку
    assert resp.json()["status"] == "pending"


async def test_create_entry_unknown_domain_400(client):
    headers = await _users(client)
    resp = await client.post(
        "/hitl", json={**ENTRY, "domain": "game_dev"}, headers=headers["user"]
    )
    assert resp.status_code == 400


async def test_pending_requires_editor_role(client):
    headers = await _users(client)
    await _create_entry(client, headers["user"])
    resp = await client.get("/hitl/pending", headers=headers["user"])
    assert resp.status_code == 403  # обычный user очередь не видит
    resp = await client.get("/hitl/pending", headers=headers["admin"])
    assert resp.status_code == 200
    assert len(resp.json()["entries"]) == 1


async def test_approve_creates_memory_case_atomically(client, db_session_factory):
    headers = await _users(client)
    entry_id = await _create_entry(client, headers["user"])

    resp = await client.post(
        f"/hitl/{entry_id}/approve", headers=headers["admin"]
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["entry"]["status"] == "approved"
    assert body["memory_case_created"] is True

    # кейс появился в Decision Memory с детерминированным id
    async with db_session_factory() as session:
        memory = await session.get(DecisionMemory, case_id(ENTRY["situation"], ENTRY["decision"]))
    assert memory is not None
    assert memory.domain == "b2b_saas"
    assert memory.quality == 8  # из critic_score
    assert "продакт-владельцем" in memory.outcome


async def test_double_approve_409(client):
    headers = await _users(client)
    entry_id = await _create_entry(client, headers["user"])
    first = await client.post(f"/hitl/{entry_id}/approve", headers=headers["admin"])
    assert first.status_code == 200
    second = await client.post(f"/hitl/{entry_id}/approve", headers=headers["admin"])
    assert second.status_code == 409


async def test_approve_duplicate_case_does_not_duplicate_memory(client, db_session_factory):
    """Кейс уже в памяти (например, пришёл импортом): подтверждение фиксируется, дубль не создаётся."""
    headers = await _users(client)
    entry_id = await _create_entry(client, headers["user"])

    async with db_session_factory() as session:
        session.add(DecisionMemory(
            id=case_id(ENTRY["situation"], ENTRY["decision"]),
            situation=ENTRY["situation"], decision=ENTRY["decision"],
            outcome="пришёл импортом", domain="b2b_saas", quality=5,
        ))
        await session.commit()

    resp = await client.post(f"/hitl/{entry_id}/approve", headers=headers["admin"])
    assert resp.status_code == 200
    assert resp.json()["memory_case_created"] is False

    async with db_session_factory() as session:
        rows = (await session.execute(select(DecisionMemory))).scalars().all()
    assert len(rows) == 1


async def test_reject_does_not_touch_memory(client, db_session_factory):
    headers = await _users(client)
    entry_id = await _create_entry(client, headers["user"])
    resp = await client.post(f"/hitl/{entry_id}/reject", headers=headers["admin"])
    assert resp.status_code == 200
    assert resp.json()["status"] == "rejected"

    async with db_session_factory() as session:
        rows = (await session.execute(select(DecisionMemory))).scalars().all()
    assert rows == []  # отклонённое решение в память не попадает

    pending = await client.get("/hitl/pending", headers=headers["admin"])
    assert pending.json()["entries"] == []
    history = await client.get("/hitl/history", headers=headers["admin"])
    assert history.json()["entries"][0]["status"] == "rejected"


async def test_unknown_entry_404(client):
    headers = await _users(client)
    resp = await client.post("/hitl/999/approve", headers=headers["admin"])
    assert resp.status_code == 404
