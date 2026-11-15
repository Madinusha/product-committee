"""Retrieval-эксперимент: Recall@k / MRR для каналов BM25 / вектор / RRF.

Запуск: python scripts/eval_retrieval.py [--limit 200]
Без torch считается только BM25-канал; после poetry install все три.
Результат печатается Markdown-таблицей (готова для отчёта).
"""
import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.data.repository import CaseEntry, get_cases
from backend.db.engine import get_session_factory, init_db
from backend.eval.retrieval_metrics import evaluate_channel
from backend.eval.tables import markdown_table
from backend.retrieval.bm25 import BM25Index
from backend.retrieval.hybrid import rrf_merge

DATASETS = Path("backend/data/datasets")


async def load_cases() -> list[CaseEntry]:
    await init_db()
    async with get_session_factory()() as session:
        return await get_cases(session)


async def bm25_channel(index: BM25Index, pairs: list[dict], top_k: int = 10) -> list[list[str]]:
    """Ранжирования BM25-канала."""
    return [
        [doc_id for doc_id, _ in await index.search(pair["query"], top_k=top_k)]
        for pair in pairs
    ]


async def vector_channel(service, pairs: list[dict], top_k: int = 10) -> list[list[str]] | None:
    """Ранжирования векторного канала; None, если эмбеддинги недоступны (нет torch)."""
    try:
        vectors = service._embeddings.encode([pair["query"] for pair in pairs])
    except Exception as exc:
        print(f"Векторный канал недоступен ({exc}) - пропускаю; выполните poetry install")
        return None

    rankings = []
    for vector in vectors:
        hits = await service._qdrant.search(vector, top_k=top_k)
        rankings.append([case_id for case_id, _ in hits])
    return rankings


async def main(limit: int) -> None:
    pairs = json.loads((DATASETS / "retrieval_pairs.json").read_text(encoding="utf-8"))[:limit]
    relevant = [pair["case_id"] for pair in pairs]
    cases = await load_cases()
    if not cases:
        print("Decision Memory пуста - сначала выполните:")
        print("  python scripts/add_cases.py backend/data/datasets/decision_memory.json")
        sys.exit(1)

    async def fetch():
        return [(c.id, f"{c.situation} {c.outcome}") for c in cases]

    index = BM25Index(fetch)
    bm25_rankings = await bm25_channel(index, pairs)
    results = {"BM25": evaluate_channel(bm25_rankings, relevant)}

    # векторный канал: через тот же движок, что и продакшн (search.py)
    from backend.api.routes.decide import get_retriever

    service = get_retriever().service
    vector_rankings = await vector_channel(service, pairs)
    if vector_rankings is not None:
        results["Векторный"] = evaluate_channel(vector_rankings, relevant)
        hybrid = [
            [doc_id for doc_id, _ in rrf_merge([v, b])]
            for v, b in zip(vector_rankings, bm25_rankings, strict=True)
        ]
        results["RRF (гибрид)"] = evaluate_channel(hybrid, relevant)

    headers = ["Канал", "recall@1", "recall@5", "MRR", "Запросов"]
    rows = [[name] + [m.get("recall@1"), m.get("recall@5"), m.get("mrr"), m.get("n_queries")]
            for name, m in results.items()]
    print(markdown_table(headers, rows))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=200)
    args = parser.parse_args()
    asyncio.run(main(args.limit))
