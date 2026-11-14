"""Тесты Celery (этап 11): задачи в eager-режиме, beat-расписание, graceful shutdown."""
from datetime import UTC

import pytest

from backend.core.state import StateKeys
from backend.worker import tasks as worker_tasks
from backend.worker.celery_app import celery_app, get_worker_loop


@pytest.fixture(autouse=True)
def eager_mode(monkeypatch):
    """Задачи выполняются синхронно в тестах, без брокера."""
    celery_app.conf.task_always_eager = True
    celery_app.conf.broker_url = "memory://"
    yield
    celery_app.conf.task_always_eager = False


def test_beat_schedule_has_cleanup_task():
    assert "cleanup-cache-nightly" in celery_app.conf.beat_schedule
    entry = celery_app.conf.beat_schedule["cleanup-cache-nightly"]
    assert entry["task"] == "backend.worker.tasks.cleanup_cache"


def test_worker_loop_is_persistent():
    """Повторные вызовы возвращают один и тот же loop (один loop на процесс)."""
    loop1 = get_worker_loop()
    loop2 = get_worker_loop()
    assert loop1 is loop2
    assert not loop1.is_closed()


def test_worker_loop_runs_coroutines():
    async def sample():
        return 42

    assert get_worker_loop().run_until_complete(sample()) == 42


async def test_reindex_task_uses_service(db_session_factory, monkeypatch):
    """Переиндексация: подсчёт кейсов от переданного сервиса (мок)."""
    import asyncio

    calls = []

    class FakeService:
        async def reindex(self, session):
            calls.append(session)
            return {"cases": 1300, "qdrant": 1300}

    monkeypatch.setattr(worker_tasks, "_build_reindex_service", lambda: FakeService())
    monkeypatch.setattr("backend.db.engine.get_session_factory", lambda: db_session_factory)
    result = await asyncio.to_thread(worker_tasks.reindex_after_hitl_task)
    assert result == {"cases": 1300, "qdrant": 1300}
    assert len(calls) == 1


async def test_batch_task_reports_errors_per_item(db_session_factory, monkeypatch):
    """Batch: ошибка одного элемента не валит остальные."""
    from backend.core.exceptions import UnknownEvalModeError

    async def fake_safe(session, situation, domain, mode, **kwargs):
        if mode == "bad-mode":
            return None, UnknownEvalModeError("неизвестный режим")
        return {
            StateKeys.DECISION: "Снизить отток до 5% за месяц (отвечает PM): retention +2.5 п.п.",
            StateKeys.CRITIC_SCORE: 8.0,
            StateKeys.EVAL_MODE: mode,
        }, None

    monkeypatch.setattr("backend.services.pipeline.safe_perform_decision", fake_safe)
    import asyncio

    result = await asyncio.to_thread(
        worker_tasks.batch_decide_task,
        [
            {"situation": "Отток 8%, цель 5%", "domain": "general", "mode": "agents_only"},
            {"situation": "Отток 8%, цель 5%", "domain": "general", "mode": "bad-mode"},
        ],
    )
    assert result[0]["status"] == "ok"
    assert result[0]["critic_score"] == 8.0
    assert result[1]["status"] == "error"


async def test_cleanup_cache_removes_unused_old_entries(db_session_factory, monkeypatch):
    from datetime import datetime, timedelta

    from backend.db.models import DecisionCache

    now = datetime.now(UTC)
    async with db_session_factory() as session:
        session.add(DecisionCache(
            situation_hash="old-unused", domain="general", decision="d",
            created_at=now - timedelta(days=40), hits=0,
        ))
        session.add(DecisionCache(
            situation_hash="old-used", domain="general", decision="d",
            created_at=now - timedelta(days=40), hits=2,
        ))
        session.add(DecisionCache(
            situation_hash="fresh", domain="general", decision="d",
            created_at=now, hits=0,
        ))
        await session.commit()

    monkeypatch.setattr("backend.db.engine.get_session_factory", lambda: db_session_factory)
    import asyncio

    result = await asyncio.to_thread(worker_tasks.cleanup_cache_task)

    assert result["total_before"] == 3
    assert result["deleted"] == 1  # только старая и ни разу не использованная
    async with db_session_factory() as session:
        remaining = {r.situation_hash for r in (await session.execute(
            __import__("sqlalchemy").select(DecisionCache)
        )).scalars().all()}
    assert remaining == {"old-used", "fresh"}


def test_trigger_reindex_false_when_broker_down(monkeypatch):
    """Брокер лежит: trigger_reindex возвращает False и не бросает исключений."""
    class FailingTask:
        def delay(self):
            raise ConnectionError("redis down")

    monkeypatch.setattr(worker_tasks, "reindex_after_hitl_task", FailingTask())
    assert worker_tasks.trigger_reindex() is False


def test_trigger_reindex_true_when_queued(monkeypatch):
    class QueuedTask:
        def __init__(self):
            self.called = False

        def delay(self):
            self.called = True
            return None

    task = QueuedTask()
    monkeypatch.setattr(worker_tasks, "_broker_alive", lambda: True)
    monkeypatch.setattr(worker_tasks, "reindex_after_hitl_task", task)
    assert worker_tasks.trigger_reindex() is True
    assert task.called


async def test_hitl_approve_triggers_reindex(client, db_session_factory, monkeypatch):
    """Связка HITL -> переиндексация: approve ставит задачу в очередь."""
    triggered = []
    monkeypatch.setattr(
        "backend.worker.tasks.trigger_reindex", lambda: triggered.append(1) or True
    )
    await client.post(
        "/auth/register",
        json={"email": "a@example.com", "username": "admin", "password": "Password1"},
    )
    token = (
        await client.post(
            "/auth/login", json={"email": "a@example.com", "password": "Password1"}
        )
    ).json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    entry = await client.post(
        "/hitl",
        json={
            "situation": "Отток 8%/мес, цель 5%, команда 2 человека, срок месяц",
            "decision": (
                "Снизить отток с 8% до 5% за 1 месяц (отвечает PM): "
                "retention +2.5 п.п., замер - когортный анализ."
            ),
            "domain": "general",
            "critic_score": 8.0,
        },
        headers=headers,
    )
    approve = await client.post(f"/hitl/{entry.json()['id']}/approve", headers=headers)
    assert approve.status_code == 200
    assert triggered  # переиндексация запущена
