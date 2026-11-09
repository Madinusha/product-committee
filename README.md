# Виртуальный продуктовый комитет (VPC)

Мультиагентная система принятия продуктовых решений с RAG: на вход — описание
продуктовой ситуации (продукт, метрики, цель, ограничения), на выход —
обоснованное решение, прошедшее цикл согласования виртуальной командой
(Validator → Critic → Editor) под управлением Supervisor.

## Тема

**Моделирование продуктовой разработки: принятие решений с помощью
мультиагентной системы.** Система имитирует работу продуктовой команды:
черновик решения готовится с опорой на глоссарий продукт-менеджмента и память
прошлых кейсов (Decision Memory, гибридный поиск BM25 + векторный через RRF),
проверяется правилами (Validator), оценивается LLM-судьёй (Critic),
дорабатывается (Editor) — итеративно, с адаптивным бюджетом и rollback при
регрессии качества. Человек (продакт-владелец) подтверждает решение через
Human-in-the-Loop, и оно попадает в Decision Memory (active learning).

## Команда и роли

| Участник | Роль | Зона ответственности |
|---|---|---|
| Мадина | Backend/Decision Core | Каркас, auth, мультиагентное ядро (Validator/Critic/Editor/Supervisor), LLM-бэкенды, API, кэш, Celery |
| Коля | ML/RAG (Knowledge) | Глоссарий и Decision Memory, эмбеддинги, BM25, Qdrant, RRF, eval (8 режимов), активное обучение |

## Стек

Python 3.12+ (Poetry) · FastAPI · SQLAlchemy 2.0 async (PostgreSQL/SQLite) ·
JWT + bcrypt · Ollama (qwen2.5:7b) + HuggingFace Transformers (fallback) ·
rank-bm25 + sentence-transformers + Qdrant · Redis + Celery · Streamlit ·
Docker Compose · pytest (покрытие ≥ 70 %)

> Агентные фреймворки (LangChain/LangGraph/CrewAI) **не используются** —
> агенты реализованы самостоятельно на asyncio, по требованиям задания.

## Архитектура (агентный конвейер)

```
Ситуация → [RAG: глоссарий + похожие кейсы] → черновик решения
        → Validator (rule-based) → Critic (LLM-as-a-Judge, 0–10)
        → Editor (доработка) → ... цикл Supervisor'а (бюджет 3/2/1/0, rollback)
        → решение + reasoning_log → HITL → Decision Memory
```

## Запуск (dev)

```bash
poetry install
cp .env.example .env          # при первом запуске
python run.py                 # http://localhost:8000, docs: /docs
```

## Структура

```
backend/
  core/      config, constants, exceptions, logging, state-контракт агентов
  api/       FastAPI, deps, routes (health, auth, ... — этап 9: decide/batch/ensemble)
  db/        async engine, модели (users — этап 4; остальные таблицы — этап 3, Коля)
  services/  auth (bcrypt + JWT)
  agents/    base, validator, critic, editor, supervisor
  utils/     async_utils, http_client, hf_llm, llm_processing
  retrieval/ data/ hitl/ worker/ eval/   ← этапы 5–11
streamlit_app/   ← этап 12
scripts/         ← идемпотентные скрипты импорта (этап 6)
tests/           ← пишутся вместе с кодом
infrastructure/  ← Docker, nginx, compose (этап 3+)
```

## Статус по этапам

- [x] Этап 1: тема, окружение, структура, первый коммит
- [ ] Этап 2: config, исключения, логирование, FastAPI + /healthz
- [ ] Этап 3: БД, Docker
- [ ] Этап 4: аутентификация
- [ ] Этап 5: мультиагентное ядро
- [ ] Этапы 6–8: данные, RAG, интеграция LLM
- [ ] Этап 9: pipeline + API
- [ ] Этапы 10–11: HITL, Celery
- [ ] Этап 12: Streamlit, README, e2e
