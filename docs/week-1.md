# Отчёт: (этапы 1–5 дорожки Decision Core)

## Тема
Моделирование продуктовой разработки: принятие решений с помощью мультиагентной
системы («Виртуальный продуктовый комитет»). Вход — описание продуктовой ситуации,
выход — обоснованное решение, прошедшее цикл Validator → Critic → Editor под
управлением Supervisor, с RAG-контекстом (глоссарий + Decision Memory).

## Разделение работ
- **Мадина (Decision Core)**: каркас, auth, агентное ядро, LLM-бэкенды — сделано на этой неделе.
- **Коля (Knowledge & RAG)**: глоссарий/Decision Memory, BM25+вектор+RRF, eval — начинает после передачи контракта `backend/core/state.py`.

## Сделано
- Скелет проекта: pyproject.toml (PEP 621, Poetry 2), .env.example, gitignore, структура каталогов.
- `core/`: config (pydantic-settings + production-валидация JWT_SECRET), константы
  (8 ablation-режимов, домены, StopReason), иерархия CommitteeException, логирование
  (цветная консоль + ротация + JSON-опция).
- `core/state.py` — **контракт состояния агентов** (точка стыка с RAG-дорожкой:
  `glossary_used`, `retrieved_context` заполняет Коля на этапе 7).
- `db/`: async engine (SQLAlchemy 2.0), модель User; остальные 4 таблицы — этап 3, Коля.
- `services/auth` + `api/routes/auth.py`: bcrypt напрямую, JWT HS256, роли
  (первый пользователь — admin), регистрация/логин/me; централизованный маппинг
  исключений на 400/401/403/409.
- `api/routes/health.py`: /healthz (liveness) и /health (readiness, 503 при падении БД).
- `utils/`: retry-декоратор, per-event-loop httpx-клиент Ollama, фабрика HF-моделей
  (синглтоны + unload, ленивый импорт transformers), clean_decision.
- `agents/`: BaseAgent (abstract) → Agent (движки auto|ollama|huggingface,
  fallback Ollama→HF, strict_engine, seed=42), Validator (7 rule-based проверок +
  SMART-правила: метрика/срок-ответственный), Critic (LLM-as-a-Judge, трёхуровневый
  few-shot universal/domain/domain-specific, JSON-сканер с балансом скобок, при сбое
  overall_score=None и critic_failed=True — fallback 5.0 запрещён), Editor
  (temperature 0.3, NO-CHANGE), Supervisor (адаптивный бюджет 3/2/1/0,
  snapshot/restore rollback с сохранением reasoning_log, флаги use_critic/use_editor).
- Тесты: **92 passed, покрытие 90.7%**

## Решения
1. State — dict с ключами из `StateKeys` (по референсу), контракт зафиксирован до старта RAG-дорожки.
2. Term-glossary проверка Validator'а имеет хук `backend.retrieval.similarity`
   (порог 0.7): пока RAG-модуля нет — точное совпадение, после этапа 7 подключится
   эмбеддинг-сходство без изменения агентов.
3. HF-fallback модель — Qwen2.5-1.5B-Instruct (полный 7B на CPU слишком медленный для live-демо).
