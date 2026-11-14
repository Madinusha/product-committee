"""E2E-сценарий защиты (этап 12): регистрация -> решение -> кэш-хит ->
HITL-подтверждение -> кейс в памяти -> повторное решение.

LLM мокается; всё остальное - реальный стек (API, БД, кэш, HITL).
Запуск: pytest tests/test_e2e.py
"""
import pytest
from sqlalchemy import select

from backend.agents import Agent
from backend.data.importers import case_id
from backend.db.models import DecisionMemory, HitlQueue
from tests.conftest import make_llm_mock

SITUATION = "Отток 8%/мес, цель 5%, команда 2 человека, срок 1 месяц"
DECISION = (
    "Снизить отток с 8% до 5% за 1 месяц (отвечает PM): retention +2.5 п.п., "
    "замер - когортный анализ."
)
CRITIC_8 = (
    '{"justification": 9, "feasibility": 8, "risks": 7, "terminology": 9, '
    '"overall_score": 8.0, "suggestions": []}'
)


@pytest.fixture()
def e2e_llm(monkeypatch):
    monkeypatch.setattr(
        Agent, "_call_llm", make_llm_mock({"analyst": DECISION, "critic": CRITIC_8})
    )


@pytest.fixture()
def stub_retriever(monkeypatch):
    from backend.api.routes import decide as decide_module
    from backend.core.state import StateKeys

    class Stub:
        async def fill_context(self, session, state, *, hybrid=True):
            state[StateKeys.RETRIEVAL_BACKEND] = "none"
            return state

    monkeypatch.setattr(decide_module, "get_retriever", lambda: Stub())


async def test_full_e2e_cycle(client, e2e_llm, stub_retriever, db_session_factory):
    headers = {"Authorization": "pending"}
    # 1. регистрация: первый пользователь становится admin
    resp = await client.post(
        "/auth/register",
        json={"email": "owner@example.com", "username": "owner", "password": "Password1"},
    )
    assert resp.status_code == 201
    token = (
        await client.post(
            "/auth/login", json={"email": "owner@example.com", "password": "Password1"}
        )
    ).json()["access_token"]
    headers["Authorization"] = f"Bearer {token}"

    # 2. первое решение: полный pipeline (full-режим по умолчанию)
    first = await client.post(
        "/decide",
        json={"situation": SITUATION, "domain": "general", "mode": "full"},
        headers=headers,
    )
    assert first.status_code == 200
    body = first.json()
    assert body["decision"] == DECISION
    assert body["cached"] is False
    assert body["critic_score"] == 8.0
    assert body["stopped_reason"] == "success"

    # 3. повторное решение: кэш-хит
    second = await client.post(
        "/decide",
        json={"situation": SITUATION, "domain": "general", "mode": "full"},
        headers=headers,
    )
    assert second.status_code == 200
    assert second.json()["cached"] is True

    # 4. решение уходит в HITL и подтверждается продакт-владельцем
    entry = await client.post(
        "/hitl",
        json={
            "situation": SITUATION,
            "decision": DECISION,
            "domain": "general",
            "critic_score": 8.0,
        },
        headers=headers,
    )
    assert entry.status_code == 201
    entry_id = entry.json()["id"]

    pending = await client.get("/hitl/pending", headers=headers)
    assert len(pending.json()["entries"]) == 1

    approved = await client.post(f"/hitl/{entry_id}/approve", headers=headers)
    assert approved.status_code == 200
    assert approved.json()["memory_case_created"] is True

    # 5. кейс зафиксирован в Decision Memory (active learning замкнулся)
    async with db_session_factory() as session:
        memory = await session.get(DecisionMemory, case_id(SITUATION, DECISION))
        entries = (await session.execute(select(HitlQueue))).scalars().all()
    assert memory is not None and memory.quality == 8
    assert entries[0].status == "approved"

    # 6. здоровье системы: все компоненты (в тесте - db; остальные заглушены моками ниже)
    health = await client.get("/healthz")
    assert health.status_code == 200


