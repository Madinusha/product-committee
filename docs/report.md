# Отчёт: мультиагентная система принятия продуктовых решений

## 1. Постановка

Тема: моделирование продуктовой разработки - принятие решений с помощью
мультиагентной системы. На вход - описание продуктовой ситуации, на выход -
обоснованное решение, прошедшее цикл согласования виртуальной команды.

## 2. Архитектура

Конвейер: RAG-контекст (глоссарий + похожие кейсы, BM25 + векторный -> RRF)
-> Analyst (черновик) -> Validator (rule-based) -> Critic (LLM-as-a-Judge)
-> Editor -> цикл Supervisor'а с адаптивным бюджетом и rollback.
L2-бэкенды: Ollama (qwen2.5:7b) с fallback на HuggingFace. HITL: подтверждённое
продакт-владельцем решение попадает в Decision Memory (active learning).

## 3. Ablation-эксперименты (8 режимов)

Режимы моделируют состав команды: baseline (только модель), glossary, tm,
tm_hybrid, agents_only, tm_hybrid_critic, tm_hybrid_editor, full.

<!-- ЗАПОЛНИТЬ после живого прогона: таблица score по 8 режимам на eval-наборе -->

## 4. Статистическая значимость

Paired bootstrap (10k итераций, seed=42) для пар режимов full vs baseline.

<!-- ЗАПОЛНИТЬ: CI95 и p-value -->

## 5. Качество retrieval

Recall@1 / Recall@5 / MRR по каналам на 200 парах запрос-эталон:

<!-- ЗАПОЛНИТЬ: python scripts/eval_retrieval.py --limit 200 (после poetry install) -->

## 6. Калибровка порога similarity

<!-- ЗАПОЛНИТЬ: python scripts/calibrate_threshold.py -->

## 7. Инфраструктура и эксплуатация

Docker Compose (8 сервисов), nginx с security headers, health-checks,
readiness /health, кэш решений с составным ключом.

## 8. Выводы

<!-- ЗАПОЛНИТЬ: что дало наибольший прирост качества, где помогла память кейсов -->
