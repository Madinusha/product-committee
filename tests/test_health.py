"""Health-роуты: liveness и readiness."""

from backend.api.routes import health as health_route


async def test_healthz_always_ok(client):
    resp = await client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


async def test_health_ok_when_db_up(client, monkeypatch):
    async def ok():
        return True

    monkeypatch.setattr(health_route, "check_db", ok)
    resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["components"]["db"] == "ok"


async def test_health_503_when_db_down(client, monkeypatch):
    async def down():
        return False

    monkeypatch.setattr(health_route, "check_db", down)
    resp = await client.get("/health")
    assert resp.status_code == 503
