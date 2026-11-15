"""Интеграционные тесты admin CRUD: роли, идемпотентный импорт, статистика кэша."""
import json

from backend.db.models import DecisionCache


async def _admin_token(client) -> str:
    await client.post(
        "/auth/register",
        json={"email": "admin@example.com", "username": "admin", "password": "Password1"},
    )
    resp = await client.post(
        "/auth/login", json={"email": "admin@example.com", "password": "Password1"}
    )
    return f"Bearer {resp.json()['access_token']}"


async def _user_token(client) -> str:
    await client.post(
        "/auth/register",
        json={"email": "user@example.com", "username": "user", "password": "Password1"},
    )
    resp = await client.post(
        "/auth/login", json={"email": "user@example.com", "password": "Password1"}
    )
    return f"Bearer {resp.json()['access_token']}"


GLOSSARY_JSON = json.dumps([
    {"term": "CAC", "definition": "стоимость привлечения клиента"},
    {"term": "LTV", "definition": "ценность клиента за срок жизни"},
])


async def test_admin_endpoints_require_admin(client):
    admin = await _admin_token(client)
    user = await _user_token(client)
    for method, path in (
        ("post", "/admin/glossary/import"),
        ("get", "/admin/glossary"),
        ("delete", "/admin/glossary/xyz"),
        ("get", "/admin/cache/stats"),
    ):
        resp = await client.request(method, path, headers={"Authorization": user})
        assert resp.status_code == 403, f"{method} {path} доступен не-admin"

    resp = await client.get("/admin/glossary", headers={"Authorization": admin})
    assert resp.status_code == 200


async def test_glossary_import_idempotent(client):
    admin = {"Authorization": await _admin_token(client)}
    resp = await client.post(
        "/admin/glossary/import",
        files={"file": ("glossary.json", GLOSSARY_JSON.encode(), "application/json")},
        headers=admin,
    )
    assert resp.status_code == 200
    assert resp.json()["created"] == 2

    # повторная загрузка: ничего не создаёт
    resp = await client.post(
        "/admin/glossary/import",
        files={"file": ("glossary.json", GLOSSARY_JSON.encode(), "application/json")},
        headers=admin,
    )
    assert resp.json() == {"created": 0, "skipped": 2, "errors": []}

    listing = await client.get("/admin/glossary", headers=admin)
    assert [e["term"] for e in listing.json()["entries"]] == ["CAC", "LTV"]


async def test_glossary_import_reports_bad_rows(client):
    admin = {"Authorization": await _admin_token(client)}
    bad = json.dumps([{"term": "ок", "definition": ""}, {"term": "RICE", "definition": "приоритизация"}])
    resp = await client.post(
        "/admin/glossary/import",
        files={"file": ("g.json", bad.encode(), "application/json")},
        headers=admin,
    )
    body = resp.json()
    assert body["created"] == 1 and len(body["errors"]) == 1


async def test_glossary_delete(client):
    admin = {"Authorization": await _admin_token(client)}
    await client.post(
        "/admin/glossary/import",
        files={"file": ("g.json", GLOSSARY_JSON.encode(), "application/json")},
        headers=admin,
    )
    term_id = (await client.get("/admin/glossary", headers=admin)).json()["entries"][0]["id"]
    resp = await client.delete(f"/admin/glossary/{term_id}", headers=admin)
    assert resp.json()["deleted"] == 1


async def test_cache_stats_and_clear(client, db_session_factory):
    admin = {"Authorization": await _admin_token(client)}
    async with db_session_factory() as session:
        session.add(DecisionCache(
            situation_hash="h1", domain="general", decision="d1",
            eval_mode="full", engine="ollama", model="qwen2.5:7b",
            critic_score=8.0, hits=3,
        ))
        session.add(DecisionCache(
            situation_hash="h2", domain="general", decision="d2",
            eval_mode="full", engine="ollama", model="qwen2.5:7b",
            critic_score=6.0, hits=1,
        ))
        await session.commit()

    stats = (await client.get("/admin/cache/stats", headers=admin)).json()
    assert stats["entries"] == 2
    assert stats["total_hits"] == 4
    assert stats["avg_critic_score"] == 7.0

    cleared = (await client.delete("/admin/cache", headers=admin)).json()
    assert cleared["deleted"] == 2
    stats = (await client.get("/admin/cache/stats", headers=admin)).json()
    assert stats["entries"] == 0


async def test_cases_import_and_list(client):
    admin = {"Authorization": await _admin_token(client)}
    cases = json.dumps([{
        "situation": "Отток 8%, цель 5%, срок месяц",
        "decision": "Снизить отток за месяц, отвечает PM, retention +2.5 п.п.",
        "outcome": "достигли 5%", "domain": "b2b_saas",
    }])
    resp = await client.post(
        "/admin/cases/import",
        files={"file": ("cases.json", cases.encode(), "application/json")},
        headers=admin,
    )
    assert resp.json()["created"] == 1

    listing = (await client.get("/admin/cases", headers=admin)).json()["entries"]
    assert listing[0]["domain"] == "b2b_saas"
