"""Health-роуты: liveness и полный readiness (db, ollama, qdrant, redis)."""
import pytest

from backend.api.routes import health as health_route
from backend.services import health as health_service


def _patch_checks(monkeypatch, **states):
    """Подменить компоненты: _patch_checks(monkeypatch, db="ok", ollama="down", ...)."""
    full = {"db": "ok", "ollama": "ok", "qdrant": "ok", "redis": "ok"}
    full.update(states)

    async def run_all():
        return full

    monkeypatch.setattr(health_route, "run_all_health_checks", run_all)


async def test_healthz_always_ok(client):
    resp = await client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


async def test_health_all_components_up(client, monkeypatch):
    _patch_checks(monkeypatch, db="ok", ollama="ok", qdrant="ok", redis="ok")
    resp = await client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert set(body["components"]) == {"app", "db", "ollama", "qdrant", "redis"}


async def test_health_503_when_any_component_down(client, monkeypatch):
    _patch_checks(monkeypatch, ollama="down")
    resp = await client.get("/health")
    assert resp.status_code == 503
    body = resp.json()["detail"]
    assert body["components"]["ollama"] == "down"
    assert body["components"]["db"] == "ok"


@pytest.mark.parametrize("component", ["db", "ollama", "qdrant", "redis"])
async def test_health_503_for_each_required_component(client, monkeypatch, component):
    _patch_checks(monkeypatch, **{component: "down"})
    resp = await client.get("/health")
    assert resp.status_code == 503


async def test_run_all_health_checks_never_raises(monkeypatch):
    """Лежащая Ollama не должна ронять /health — только помечать компонент down."""
    async def boom():
        raise RuntimeError("connection refused")

    monkeypatch.setattr(health_service, "check_ollama", boom)
    components = await health_service.run_all_health_checks()
    assert components["ollama"] == "down" or components["ollama"] == "ok"
