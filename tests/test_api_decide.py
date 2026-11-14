"""Интеграционные тесты decide-роутов: auth, кэш, режимы, ошибки, ensemble.

LLM мокается на уровне Agent._call_llm, retrieval - на уровне RetrieverAdapter
(роуты тестируются через реальный HTTP-слой, но без Ollama/Qdrant).
"""
import pytest

from backend.agents import Agent
from backend.core.state import StateKeys
from tests.conftest import make_llm_mock

GOOD_DRAFT = (
    "Снизить отток с 8% до 5% за 1 месяц (отвечает PM): retention +2.5 п.п., "
    "замер - когортный анализ."
)
CRITIC_8 = (
    '{"justification": 9, "feasibility": 8, "risks": 7, "terminology": 9, '
    '"overall_score": 8.0, "suggestions": []}'
)


@pytest.fixture()
def fake_llm(monkeypatch):
    monkeypatch.setattr(
        Agent, "_call_llm",
        make_llm_mock({"analyst": GOOD_DRAFT, "critic": CRITIC_8}),
    )


@pytest.fixture()
def fake_retriever(monkeypatch):
    """Подменяем синглтон retrieval в роутах на фейк (без БД-корпуса)."""
    from backend.api.routes import decide as decide_module

    class StubRetriever:
        async def fill_context(self, session, state, *, hybrid=True):
            state[StateKeys.RETRIEVAL_BACKEND] = "none"
            return state

    monkeypatch.setattr(decide_module, "get_retriever", lambda: StubRetriever())


async def _token(client) -> str:
    await client.post(
        "/auth/register",
        json={"email": "u@example.com", "username": "tester", "password": "Password1"},
    )
    response = await client.post(
        "/auth/login", json={"email": "u@example.com", "password": "Password1"}
    )
    return response.json()["access_token"]


async def test_decide_requires_auth(client, fake_llm, fake_retriever):
    response = await client.post(
        "/decide", json={"situation": "Отток 8 процентов, цель 5 процентов"}
    )
    assert response.status_code == 401


async def test_decide_ok_returns_solution(client, fake_llm, fake_retriever):
    token = await _token(client)
    response = await client.post(
        "/decide",
        json={"situation": "Отток 8%/мес, цель 5%, команда 2 человека", "domain": "general"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["decision"] == GOOD_DRAFT
    assert body["critic_score"] == 8.0
    assert body["cached"] is False
    assert body["retrieval_backend"] == "none"
    assert body["reasoning_log"]


async def test_decide_second_request_hits_cache(client, fake_llm, fake_retriever):
    token = await _token(client)
    payload = {
        "situation": "Отток 8%/мес, цель 5%, команда 2 человека",
        "domain": "general",
        "mode": "full",
    }
    headers = {"Authorization": f"Bearer {token}"}
    first = await client.post("/decide", json=payload, headers=headers)
    assert first.json()["cached"] is False
    second = await client.post("/decide", json=payload, headers=headers)
    assert second.status_code == 200
    assert second.json()["cached"] is True
    assert second.json()["decision"] == GOOD_DRAFT


async def test_decide_unknown_mode_400(client, fake_llm, fake_retriever):
    token = await _token(client)
    response = await client.post(
        "/decide",
        json={"situation": "Отток 8%/мес, цель 5 процентов", "mode": "agentic"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 400
    assert "baseline" in response.json()["detail"]  # список поддерживаемых в ошибке


async def test_decide_unknown_domain_400(client, fake_llm, fake_retriever):
    token = await _token(client)
    response = await client.post(
        "/decide",
        json={"situation": "Отток 8%/мес, цель 5 процентов", "domain": "game_dev"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 400


async def test_multi_several_domains(client, fake_llm, fake_retriever):
    token = await _token(client)
    response = await client.post(
        "/decide/multi",
        json={
            "situation": "Отток 8%/мес, цель 5%, команда 2 человека",
            "domains": ["general", "b2b_saas"],
        },
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    results = response.json()["results"]
    assert [r["domain"] for r in results] == ["general", "b2b_saas"]
    # только general попал в кэш - b2b_saas не кэшируется (см. cache.py)
    assert results[0]["cached"] is False and results[1]["cached"] is False
    second = await client.post(
        "/decide/multi",
        json={
            "situation": "Отток 8%/мес, цель 5%, команда 2 человека",
            "domains": ["general", "b2b_saas"],
        },
        headers={"Authorization": f"Bearer {token}"},
    )
    second_results = second.json()["results"]
    assert second_results[0]["cached"] is True
    assert second_results[1]["cached"] is False


async def test_batch_partial_errors(client, fake_llm, fake_retriever):
    token = await _token(client)
    response = await client.post(
        "/decide/batch",
        json={"items": [
            {"situation": "Отток 8%/мес, цель 5%, команда 2 человека", "domain": "general"},
            {"situation": "Отток 8%/мес, цель 5%, команда 2 человека", "mode": "bad-mode"},
        ]},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    results = response.json()["results"]
    assert results[0]["status"] == "ok"
    assert results[1]["status"] == "error"


async def test_ensemble_picks_best_by_critic_score(client, fake_llm, fake_retriever, monkeypatch):
    """Два прогона с разными оценками Critic'а: best = вариант с 8.0."""

    critic_responses = [
        '{"overall_score": 6.0, "suggestions": []}',
        '{"overall_score": 8.0, "suggestions": []}',
    ]
    counter = {"n": 0}

    async def scripted(self, prompt, system=None):
        if self.name == "critic":
            response = critic_responses[min(counter["n"], 1)]
            counter["n"] += 1
            return response
        return GOOD_DRAFT

    monkeypatch.setattr(Agent, "_call_llm", scripted)
    token = await _token(client)
    response = await client.post(
        "/decide/ensemble",
        json={
            "situation": "Отток 8%/мес, цель 5%, команда 2 человека",
            "domain": "b2b_saas",
            "variants": 2,
        },
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["best"]["critic_score"] == 8.0
    assert len(body["variants"]) == 2


async def test_ensemble_unknown_domain_400(client, fake_llm, fake_retriever):
    token = await _token(client)
    response = await client.post(
        "/decide/ensemble",
        json={"situation": "Отток 8%/мес, цель 5%, команда 2 человека", "domain": "crypto"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 400
