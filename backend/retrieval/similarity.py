"""Similarity-хук для Validator'а: использован ли термин глоссария в решении.

Порог 0.7 задан контрактом в backend/agents/validator.py. Синхронный:
Validator.rule-based проверки не async, а encode sentence-transformers -
CPU-вызов; модель загружается лениво и только если substring не сработал.
"""
import logging
import re

logger = logging.getLogger("backend.retrieval")

_SIMILARITY_THRESHOLD = 0.7


def term_similarity(decision: str, term: str) -> float:
    """0..1: 1.0 при дословном вхождении, иначе cosine эмбеддингов.

    Если эмбеддинг-модель не установлена (лёгкая dev-среда без torch),
    возвращает 1.0/0.0 по подстроке - Validator продолжает работать.
    """
    norm_term = re.sub(r"\s+", " ", term.strip().lower())
    norm_decision = re.sub(r"\s+", " ", decision.lower())
    if norm_term and norm_term in norm_decision:
        return 1.0
    try:
        from backend.retrieval.embeddings import get_embedding_service

        service = get_embedding_service()
        vector = service.encode([term, decision])
        return float(service.cosine(vector[0], vector[1]))
    except Exception as exc:  # torch/sentence-transformers недоступны
        logger.debug("Эмбеддинги недоступны (%s) - только substring-проверка", exc)
        return 0.0
