"""Калибровка порога similarity (проверка глоссария в Validator'е).

Запуск (после poetry install): python scripts/calibrate_threshold.py
Берёт случайные пары (решение, термин) из датасетов: "used" - термин реально
входит в решение, "unused" - не входит; печатает распределения и рекомендацию.
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.eval.calibration import calibrate_threshold
from backend.retrieval.similarity import term_similarity

DATASETS = Path("backend/data/datasets")


def main(sample_size: int, seed: int) -> None:
    cases = json.loads((DATASETS / "decision_memory.json").read_text(encoding="utf-8"))
    glossary = json.loads((DATASETS / "glossary.json").read_text(encoding="utf-8"))
    rng = random.Random(seed)

    used, unused = [], []
    for _ in range(sample_size):
        case = rng.choice(cases)
        term_row = rng.choice(glossary)
        term = term_row["term"]
        decision_lower = case["decision"].lower()
        score = term_similarity(case["decision"], term)
        if term.lower() in decision_lower:
            used.append(score)
        else:
            unused.append(score)

    if not used or not unused:
        print("Мало данных одного из классов - увеличьте --samples")
        sys.exit(1)

    report = calibrate_threshold(used, unused)
    print("Калибровка порога term_similarity:\n")
    for key, value in report.items():
        print(f"  {key}: {value}")
    print(f"\nПримеры used: {sorted(used, reverse=True)[:5]}")
    print(f"Примеры unused: {sorted(unused, reverse=True)[:5]}")
    if abs(report["suggested_threshold"] - report["current_threshold"]) <= 0.05:
        print("\nВывод: текущий порог 0.7 подтверждается экспериментом.")
    else:
        print(
            f"\nВывод: текущий порог 0.7 стоит пересмотреть "
            f"в сторону {report['suggested_threshold']}."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args.samples, args.seed)
