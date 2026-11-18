"""Ablation-эксперимент: все 8 режимов на eval-наборе -> таблицы для отчёта.

Запуск: python scripts/run_ablation.py --limit 20 --out docs/ablation.md
Требует живой Ollama (оценки Critic'а) и импортированные данные:
  python scripts/add_glossary.py backend/data/datasets/glossary.json
  python scripts/add_cases.py backend/data/datasets/decision_memory.json
Результат: средние метрики по режимам + paired bootstrap против baseline.
"""
import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.core.constants import EVAL_MODES
from backend.core.state import StateKeys
from backend.db.engine import get_session_factory, init_db
from backend.eval.ablation import bootstrap_against_baseline, summarize
from backend.eval.tables import markdown_table
from backend.services.pipeline import DecisionPipeline, safe_perform_decision

DATASETS = Path("backend/data/datasets")


async def run_experiment(limit: int) -> tuple[dict, dict]:
    """Прогнать eval-ситуации через все режимы. Возвращает (runs_by_mode, backends)."""
    from backend.api.routes.decide import get_retriever

    situations = json.loads((DATASETS / "eval_set.json").read_text(encoding="utf-8"))[:limit]
    if not situations:
        print("eval_set.json пуст - сначала python scripts/prepare_eval_datasets.py")
        sys.exit(1)

    pipeline = DecisionPipeline(retriever=get_retriever())
    runs_by_mode: dict[str, list] = {mode: [] for mode in EVAL_MODES}
    backends: set[str] = set()
    errors: list[str] = []

    async with get_session_factory()() as session:
        for index, item in enumerate(situations):
            for mode in EVAL_MODES:
                state, error = await safe_perform_decision(
                    session,
                    item["situation"],
                    item.get("domain", "general"),
                    mode=mode,
                    retriever=pipeline.retriever,
                )
                runs_by_mode[mode].append(state)
                if state is not None and state.get(StateKeys.RETRIEVAL_BACKEND):
                    backends.add(state[StateKeys.RETRIEVAL_BACKEND])
            print(f"[{index + 1}/{len(situations)}] {item['situation'][:50]}...")
            if error is not None:
                errors.append(f"#{index}: {error.message}")

    return runs_by_mode, {"backends": sorted(backends), "errors": errors}


def build_report(runs_by_mode: dict, backends: dict) -> str:
    """Собрать Markdown-отчёт: таблица режимов + bootstrap против baseline."""
    summary = {mode: summarize(runs) for mode, runs in runs_by_mode.items()}
    summary_table = markdown_table(
        ["Режим", "avg score", "success", "валидация", "итераций", "ошибок", "n"],
        [
            [mode, m["avg_score"], m["success_rate"], m["validation_rate"],
             m["avg_iterations"], m["errors"], m["n"]]
            for mode, m in summary.items()
        ],
    )

    lines = [
        "# Ablation-эксперимент (8 режимов)",
        "",
        f"Каналы retrieval в прогоне: {', '.join(backends['backends']) or 'нет'}",
        "",
        summary_table,
        "",
        "## Paired bootstrap (score режима vs baseline, 10k итераций, seed=42)",
        "",
    ]
    if backends["errors"]:
        lines.append(f"Ошибки прогонов: {len(backends['errors'])}")
    for pair_name, report in bootstrap_against_baseline(runs_by_mode):
        lines.append(
            f"- **{pair_name}**: mean_diff={report['mean_diff']}, "
            f"CI95={report['ci95']}, p={report['p_value']}, "
            f"значимо={'да' if report['significant'] else 'нет'} (пар: {report['n_pairs']})"
        )
    return "\n".join(lines) + "\n"


async def main(limit: int, out: Path | None) -> None:
    await init_db()
    runs_by_mode, backends = await run_experiment(limit)
    report = build_report(runs_by_mode, backends)
    print(report)
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report, encoding="utf-8")
        print(f"Отчёт сохранён: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=10, help="сколько ситуаций из eval_set")
    parser.add_argument("--out", type=Path, default=None, help="куда сохранить Markdown")
    args = parser.parse_args()
    asyncio.run(main(args.limit, args.out))
