"""Импортеры: идемпотентность, детерминированные id, ошибки строк."""
from sqlalchemy import select

from backend.data.importers import case_id, glossary_id, import_cases, import_glossary
from backend.db.models import DecisionMemory, GlossaryTerm


def test_ids_are_deterministic_and_normalized():
    assert glossary_id("CAC") == glossary_id("  cac  ")
    assert case_id("Ситуация  одна.", "Решение") == case_id("ситуация одна.", "решение")
    assert glossary_id("CAC") != glossary_id("LTV")
    assert len(glossary_id("CAC")) == 64  # sha256 hex


async def test_glossary_import_idempotent(db_session_factory):
    rows = [
        {"term": "CAC", "definition": "стоимость привлечения клиента"},
        {"term": "LTV", "definition": "ценность клиента за срок жизни", "domain": "growth"},
    ]
    async with db_session_factory() as session:
        first = await import_glossary(session, rows)
        assert first.created == 2 and first.skipped == 0

        second = await import_glossary(session, rows)
        assert second.created == 0 and second.skipped == 2  # дублей нет

        # смена домена не создаёт новый термин - id от термина, не от домена
        third = await import_glossary(session, [{**rows[0], "domain": "b2b_saas"}])
        assert third.created == 0 and third.skipped == 1

        total = len((await session.execute(select(GlossaryTerm))).scalars().all())
        assert total == 2


async def test_cases_import_idempotent(db_session_factory):
    rows = [
        {"situation": "Отток 8%", "decision": "Снизить отток до 5% за месяц (отвечает PM)",
         "outcome": "достигли 5%", "domain": "b2b_saas", "decision_type": "удержание", "quality": 8},
    ]
    async with db_session_factory() as session:
        first = await import_cases(session, rows)
        assert first.created == 1
        second = await import_cases(session, rows)
        assert second.created == 0 and second.skipped == 1

    async with db_session_factory() as session:
        memory = await session.get(DecisionMemory, case_id("Отток 8%", rows[0]["decision"]))
    assert memory is not None and memory.quality == 8


async def test_import_reports_row_errors(db_session_factory):
    async with db_session_factory() as session:
        report = await import_glossary(session, [
            {"term": "", "definition": "пустой термин"},
            {"term": "ok", "definition": ""},  # нет definition
            {"term": "RICE", "definition": "приоритизация"},
        ])
    assert report.created == 1
    assert len(report.errors) == 2
    assert all("строка" in e for e in report.errors)
