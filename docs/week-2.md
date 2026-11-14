# Отчёт: неделя 2 (этапы 3, 6, 7 - дорожка Knowledge & RAG)

## Сделано

### Данные (этап 3, 6)
- 4 таблицы БД: `glossary`, `decision_memory`, `decisions_cache` (UNIQUE-ключ
  из 6 колонок, сентинел `"unknown"` вместо NULL - NULL ломает UNIQUE),
  `hitl_queue` (statuses pending/approved/rejected).
- Идемпотентный импорт: id = SHA-256 от нормализованного содержимого, поэтому
  повторный прогон не создаёт дублей (создано 506/1100 → повтор: пропущено 506/1100).
- Скрипты: `scripts/prepare_eval_datasets.py`, `add_glossary.py`, `add_cases.py`,
  `create_admin.py` - все идемпотентны.
- Датасеты (синтетические, `backend/data/datasets/`): глоссарий **506** реальных
  терминов PM (требование ≥500), память кейсов **1100** (требование ≥1000),
  eval-набор 120 ситуаций с эталонными решениями.

### Retrieval (этап 7)
- `retrieval/embeddings.py` - синглтон sentence-transformers
  (paraphrase-multilingual-MiniLM-L12-v2, 384), ленивая загрузка (torch не нужен для импорта).
- `retrieval/bm25.py` - BM25Okapi с dirty-флагом и ленивой пересборкой.
- `retrieval/qdrant.py` - AsyncQdrantClient, cosine, id точки = uuid5(case_id) -
  идемпотентный upsert.
- `retrieval/hybrid.py` - RRF k=60 с дедупликацией. Почему RRF: BM25-скоры
  (0..10+) и cosine (0..1) несравнимы; RRF работает только с рангами,
  k=60 гасит вклад верхних позиций.
- `retrieval/search.py` - RetrievalService: гибрид → при недоступности Qdrant
  деградация до BM25 (`retrieval_backend: hybrid|bm25|none` в state).
- `retrieval/similarity.py` - хук `term_similarity` (порог 0.7) для
  Validator'а: substring → 1.0, иначе cosine эмбеддингов; без torch
  деградирует до substring (Validator продолжает работать).
- Заполнение state-контракта: `glossary_used`, `retrieved_context`, `retrieval_backend`.

### Eval (этап 9, подготовка)
- `eval/bootstrap.py` - парный бутстрэп (10k итераций, seed=42) с CI95 и p-value.
- `eval/tables.py` - Markdown + LaTeX (booktabs) таблицы для отчёта.
- `eval/leak_detector.py` - проверка утечки eval↔память: на сгенерированных
  датасетах утечек 0 (eval-шаблоны намеренно другие).

## Проверки
- Тесты: **122 passed**, покрытие **88%** (требование ≥70%).
- ruff чисто по backend/, tests/, scripts/.
- Живой прогон: create_admin → add_glossary (506) → add_cases (1100) → повтор
  (0 создано, всё пропущено) - идемпотентность подтверждена на SQLite.

## Выводы из отладки
1. На маленьком корпусе BM25Okapi зануляет IDF частых терминов (механизм
   epsilon) - на боевых 1100+ кейсах не проявится, но тесты учитывают.
2. id кейсов для Qdrant - uuid5, для БД - sha256: разные пространства,
   связанные через payload.case_id.

## Что дальше
- pipeline.py (8 режимов), кэш, decide-роуты: retrieval подключается одним
  вызовом `RetrievalService.fill_context`.
- Полный `poetry install` (torch), реальный прогон эмбеддингов и Qdrant
  в docker-compose, постраничная вычитка глоссария.
