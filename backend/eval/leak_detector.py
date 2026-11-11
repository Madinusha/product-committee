"""Детектор утечки: ситуации из eval-набора не должны совпадать с кейсами
Decision Memory - иначе retrieval найдёт «правильный ответ» и метрики
режимов будут завышены. Использовать до прогона eval (этап 9)."""
import re
from difflib import SequenceMatcher


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def detect_leaks(
    memory_situations: list[str], eval_situations: list[str], threshold: float = 0.9
) -> list[dict]:
    """Почти-дубликаты между памятью кейсов и eval-набором.

    Возвращает [{eval_index, memory_index, similarity}] для пар с
    SequenceMatcher >= threshold. Точное совпадение тоже ловится (1.0).
    """
    normalized_memory = [_normalize(s) for s in memory_situations]
    leaks = []
    for eval_index, eval_text in enumerate(eval_situations):
        norm_eval = _normalize(eval_text)
        for memory_index, memory_text in enumerate(normalized_memory):
            similarity = SequenceMatcher(None, norm_eval, memory_text).ratio()
            if similarity >= threshold:
                leaks.append({
                    "eval_index": eval_index,
                    "memory_index": memory_index,
                    "similarity": round(similarity, 4),
                })
    return leaks
