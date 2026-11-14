"""Retrieval: BM25 (dirty-флаг), RRF, деградация Qdrant->BM25, fill_context."""
from backend.core.state import StateKeys, initial_state
from backend.data.repository import CaseEntry, GlossaryEntry
from backend.retrieval.bm25 import BM25Index, tokenize
from backend.retrieval.hybrid import rrf_merge
from backend.retrieval.qdrant import case_point_id
from backend.retrieval.search import RetrievalService

CORPUS = [
    ("case-1", "отток клиентов вырос снизить удержание онбординг"),
    ("case-2", "монетизация подписка тариф выручка прайсинг"),
    ("case-3", "приоритизация бэклог RICE инициативы"),
]


def _fake_cases() -> list[CaseEntry]:
    return [
        CaseEntry(
            "case-1", "Отток клиентов вырос", "Снизить отток за счёт онбординга",
            "достигли 5%", "b2b_saas", "удержание", 8,
        ),
        CaseEntry(
            "case-2", "Низкая выручка подписки", "Пересобрать прайсинг тарифов",
            "+15% MRR", "b2b_saas", "монетизация", 7,
        ),
        CaseEntry(
            "case-3", "Бэклог раздут", "Провести RICE-скоринг",
            "взяты 2 инициативы", "general", "приоритизация", 6,
        ),
    ]


def _fake_glossary() -> list[GlossaryEntry]:
    return [
        GlossaryEntry("g-1", "churn rate", "доля ушедших клиентов за период", "general"),
        GlossaryEntry("g-2", "RICE", "приоритизация по reach impact confidence effort", "general"),
    ]


class FakeEmbeddings:
    """Стаб эмбеддингов: текст с 'отток' вектор (1,0), остальное (0,1)."""

    def encode(self, texts):
        return [[1.0, 0.0] if "отток" in t else [0.0, 1.0] for t in texts]

    def encode_one(self, text):
        return self.encode([text])[0]

    @staticmethod
    def cosine(a, b):
        return sum(x * y for x, y in zip(a, b, strict=False))


class FakeQdrantDown:
    async def available(self):
        return False


class FakeQdrantUp:
    """Qdrant с жёсткой выдачей: всегда case-2 первым."""

    def __init__(self, hits):
        self._hits = hits

    async def available(self):
        return True

    async def search(self, vector, top_k=5):
        return self._hits[:top_k]


class FakeQdrantError:
    async def available(self):
        return True

    async def search(self, vector, top_k=5):
        raise RuntimeError("qdrant broken")


# токенизация и BM25

def test_tokenize_cyrillic_and_latin():
    assert tokenize("Отток D30 вырос!") == ["отток", "d30", "вырос"]


async def test_bm25_search_ranks_relevant_first():
    async def fetch():
        return CORPUS

    index = BM25Index(fetch)
    hits = await index.search("отток клиентов удержание", top_k=2)
    assert hits[0][0] == "case-1"
    assert hits[0][1] > 0
    assert len(hits) <= 2


async def test_bm25_dirty_flag_triggers_rebuild():
    corpus = list(CORPUS)

    async def fetch():
        return corpus

    index = BM25Index(fetch)
    assert index.size == 0  # индекс ленивый
    await index.search("отток")
    assert index.size == 3

    # редкий термин: на маленьком корпусе BM25Okapi зануляет IDF частых слов
    corpus.append(("case-4", "рефералы и вирусный рост через referral program"))
    hits = await index.search("отток")
    assert index.size == 3  # без dirty-флага корпус не пересобран

    index.mark_dirty()
    hits = await index.search("рефералы")
    assert index.size == 4  # пересборка подхватила новый документ
    assert any(doc_id == "case-4" for doc_id, _ in hits)


# RRF

def test_rrf_merge_dedups_and_ranks():
    merged = rrf_merge([["a", "b", "c"], ["a", "d"]], k=60)
    ids = [doc_id for doc_id, _ in merged]
    assert ids[0] == "a"  # в двух списках первым - суммарный вклад выше
    assert "b" in ids and "d" in ids


def test_rrf_score_formula():
    # doc 'a' первый в обоих списках: 1/(60+1) + 1/(60+1) = 2/61
    merged = rrf_merge([["a"], ["a"]], k=60)
    assert abs(merged[0][1] - 2 / 61) < 1e-9


def test_rrf_empty_input():
    assert rrf_merge([]) == []


# Qdrant

def test_case_point_id_stable():
    assert case_point_id("abc") == case_point_id("abc")
    assert case_point_id("abc") != case_point_id("xyz")


# RetrievalService: деградация и fill_context

def _make_service(qdrant=None):
    async def cases_fetcher():
        return _fake_cases()

    async def glossary_fetcher():
        return _fake_glossary()

    return RetrievalService(cases_fetcher, glossary_fetcher, FakeEmbeddings(), qdrant)


async def test_fill_context_without_qdrant_degrades_to_bm25(db_session_factory):
    service = _make_service()
    state = initial_state("Как снизить отток клиентов и улучшить удержание?", "b2b_saas")
    async with db_session_factory() as session:
        await service.fill_context(session, state)
    assert state[StateKeys.RETRIEVAL_BACKEND] == "bm25"
    assert state[StateKeys.RETRIEVED_CONTEXT] == []  # в БД кейсов нет - но BM25-корпус от fetcher'а пуст тоже


async def test_fill_context_hybrid_with_qdrant(db_session_factory):
    service = _make_service(qdrant=FakeQdrantUp([("case-1", 0.9), ("case-2", 0.8)]))
    # кейсы должны быть и в БД (payload-данные берутся оттуда), и в выдаче Qdrant
    async with db_session_factory() as session:
        for c in _fake_cases():
            from backend.db.models import DecisionMemory

            session.add(DecisionMemory(
                id=c.id, situation=c.situation, decision=c.decision,
                outcome=c.outcome, domain=c.domain, decision_type=c.decision_type,
                quality=c.quality,
            ))
        await session.commit()
        state = initial_state("Нужно снизить отток клиентов", "b2b_saas")
        await service.fill_context(session, state)
    assert state[StateKeys.RETRIEVAL_BACKEND] == "hybrid"
    context = state[StateKeys.RETRIEVED_CONTEXT]
    # case-1 и от BM25 (релевантен 'отток'), и от вектора - RRF ставит его первым
    assert context[0]["case_id"] == "case-1"
    assert context[0]["decision"] == "Снизить отток за счёт онбординга"


async def test_qdrant_error_falls_back_to_bm25(db_session_factory):
    service = _make_service(qdrant=FakeQdrantError())
    state = initial_state("Нужно снизить отток клиентов", "b2b_saas")
    async with db_session_factory() as session:
        await service.fill_context(session, state)
    assert state[StateKeys.RETRIEVAL_BACKEND] == "bm25"


async def test_empty_situation_means_no_retrieval(db_session_factory):
    service = _make_service()
    state = initial_state("", "general")
    async with db_session_factory() as session:
        await service.fill_context(session, state)
    assert state[StateKeys.RETRIEVAL_BACKEND] == "none"
    assert state[StateKeys.RETRIEVED_CONTEXT] == []
