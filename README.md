# Виртуальный продуктовый комитет (VPC)

Мультиагентная система принятия продуктовых решений с RAG: на вход - описание
продуктовой ситуации (продукт, метрики, цель, ограничения), на выход -
обоснованное решение, прошедшее цикл согласования виртуальной командой
(Validator → Critic → Editor) под управлением Supervisor.

## Тема

**Моделирование продуктовой разработки: принятие решений с помощью
мультиагентной системы.** Система имитирует работу продуктовой команды:
черновик решения готовится с опорой на глоссарий продукт-менеджмента и память
прошлых кейсов (Decision Memory, гибридный поиск BM25 + векторный через RRF),
проверяется правилами (Validator), оценивается LLM-судьёй (Critic),
дорабатывается (Editor) - итеративно, с адаптивным бюджетом и rollback при
регрессии качества. Человек (продакт-владелец) подтверждает решение через
Human-in-the-Loop, и оно попадает в Decision Memory (active learning).

## Команда и роли

| Участник | Роль | Зона ответственности |
|---|---|---|
| Мадина | Backend/Decision Core | Каркас, auth, мультиагентное ядро (Validator/Critic/Editor/Supervisor), LLM-бэкенды, pipeline с 8 ablation-режимами, decide-роуты, кэш, HITL-confirm, Celery, Streamlit (стр. 1-4) |
| Коля | ML/RAG (Knowledge) | Глоссарий и Decision Memory, эмбеддинги, BM25, Qdrant, RRF, retrieval-эксперименты (Recall/MRR, калибровка), eval (bootstrap, таблицы, leak detector), admin CRUD, Docker/nginx, e2e-тест, отчёт, Streamlit (стр. 5-8) |

## Стек

Python 3.12+ (Poetry) · FastAPI · SQLAlchemy 2.0 async (PostgreSQL/SQLite) ·
JWT + bcrypt · Ollama (qwen2.5:7b) + HuggingFace Transformers (fallback) ·
rank-bm25 + sentence-transformers + Qdrant · Redis + Celery · Streamlit ·
Docker Compose · pytest (покрытие ≥ 70 %)

> Агентные фреймворки (LangChain/LangGraph/CrewAI) **не используются** -
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
  api/       FastAPI, deps, routes: health, auth, decide/batch/ensemble (Мадина), admin (Коля)
  db/        async engine (Мадина), модели: users; glossary, decision_memory, cache, hitl (Коля)
  services/  auth (bcrypt + JWT)
  agents/    base, validator, critic, editor, supervisor
  utils/     async_utils, http_client, hf_llm, llm_processing
  retrieval/ data/ eval/   <- Коля: гибридный поиск, импорт, эксперименты
  hitl/ worker/            <- confirm (Мадина), Celery-воркер (Мадина)
streamlit_app/   <- этап 12: стр. 1-4 (Мадина), стр. 5-8 (Коля)
scripts/         <- идемпотентные скрипты импорта (Коля)
tests/           <- пишутся вместе с кодом
infrastructure/  <- Docker, nginx, compose (Коля)
```

## Статус по этапам

- [x] Этап 1: тема, окружение, структура, первый коммит
- [x] Этап 2: config, исключения, логирование, FastAPI + /healthz
- [x] Этап 3: БД (5 таблиц), модели
- [x] Этап 4: аутентификация
- [x] Этап 5: мультиагентное ядро
- [x] Этап 6: данные - глоссарий 506, кейсы 1100, идемпотентный импорт (SHA-256 id)
- [x] Этап 7 (частично): RAG - BM25 + Qdrant + RRF k=60, деградация до BM25, similarity-хук
- [ ] Этап 7 (остаток) / 8: интеграция LLM, pipeline с 8 режимами, кэш, decide-роуты
- [ ] Этапы 9–11: admin CRUD, HITL, Celery, связка HITL→переиндексация
- [ ] Этап 12: Streamlit, Docker, README, e2e
