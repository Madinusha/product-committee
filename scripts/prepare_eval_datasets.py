"""Сгенерировать датасеты (глоссарий, память кейсов, eval-набор).

Запуск из корня проекта: python scripts/prepare_eval_datasets.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.data.seed_terms import CORE_TERMS  # noqa: F401 - проверка импорта пакета
from scripts.dataset_generator import generate_datasets


def main() -> None:
    counts = generate_datasets(Path("backend/data/datasets"))
    print("Датасеты готовы:", counts)


if __name__ == "__main__":
    main()
