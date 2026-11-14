"""Импорт кейсов Decision Memory из JSON/CSV: python scripts/add_cases.py <path>.

Форматы: JSON [{situation, decision, outcome, domain?, decision_type?, quality?}]
или CSV с одноимёнными колонками. Идемпотентно.
"""
import asyncio
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.data.importers import import_cases
from backend.db.engine import get_session_factory, init_db


def load_rows(path: Path) -> list[dict]:
    if path.suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    if path.suffix == ".csv":
        with path.open(encoding="utf-8-sig") as f:
            return list(csv.DictReader(f))
    raise ValueError(f"Неподдерживаемый формат: {path.suffix} (ожидался .json или .csv)")


async def run(path: Path) -> None:
    await init_db()
    async with get_session_factory()() as session:
        report = await import_cases(session, load_rows(path))
    print(f"Decision Memory: создано {report.created}, пропущено {report.skipped}")
    if report.errors:
        print("Ошибки строк:")
        for error in report.errors:
            print(" -", error)


def main() -> None:
    if len(sys.argv) != 2:
        print("Использование: python scripts/add_cases.py <path.csv|.json>")
        sys.exit(1)
    asyncio.run(run(Path(sys.argv[1])))


if __name__ == "__main__":
    main()
