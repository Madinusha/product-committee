"""Генератор синтетических датасетов: глоссарий, память кейсов, eval-набор.

Кейсы собираются комбинаторикой шаблонов с seeded RNG - датасет
детерминированный, воспроизводится с любым прогоном. Кейсы синтетические:
для боевого использования замещаются реальными кейсам компании.
"""
import json
import random
from pathlib import Path

from backend.data.seed_domains import (
    B2B_SAAS_TERMS,
    EXPERIMENTATION_TERMS,
    FINTECH_TERMS,
    GROWTH_TERMS,
    MARKETPLACE_TERMS,
    MEDIA_TERMS,
)
from backend.data.seed_terms import CORE_TERMS

DOMAIN_PACKS = {
    "general": CORE_TERMS + GROWTH_TERMS + EXPERIMENTATION_TERMS,
    "b2b_saas": B2B_SAAS_TERMS,
    "marketplace": MARKETPLACE_TERMS,
    "fintech": FINTECH_TERMS,
    "media": MEDIA_TERMS,
}

MIN_GLOSSARY = 500
MIN_CASES = 1000
EVAL_SIZE = 120

PRODUCTS = {
    "b2b_saas": [
        "CRM для малого бизнеса", "сервис аналитики продаж",
        "платформа поддержки клиентов", "инструмент планирования проектов",
    ],
    "marketplace": [
        "маркетплейс мебели", "платформа репетиторов",
        "агрегатор доставки еды", "биржа фриланс-задач",
    ],
    "fintech": [
        "приложение для micro-кредитования", "сервис платежей для самозанятых",
        "факторинговая платформа", "инвестиционное приложение",
    ],
    "media": [
        "видеостриминг", "новостной агрегатор",
        "подкаст-платформа", "образовательная платформа",
    ],
}

METRICS = {
    "b2b_saas": [
        ("отток клиентов", "%/мес"), ("конверсия trial-to-paid", "%"),
        ("MRR", "руб/мес"), ("доля expansion-выручки", "%"),
    ],
    "marketplace": [
        ("конверсия в заказ", "%"), ("liquidity", "%"),
        ("средний чек", "руб"), ("доля повторных покупок", "%"),
    ],
    "fintech": [
        ("approval rate платежей", "%"), ("доля просрочек", "%"),
        ("checkout conversion", "%"), ("fraud rate", "%"),
    ],
    "media": [
        ("retention D30", "%"), ("watch time", "мин/день"),
        ("paywall conversion", "%"), ("engagement rate", "%"),
    ],
}

# Параметры: (текущее, цель, изменение)
SITUATION_BY_TYPE = {
    "приоритизация": (
        "Продукт «{product}»: {metric} сейчас {cur}{unit}, в бэклоге 12 инициатив, "
        "команда - {team} человек, горизонт {horizon}. Нужно выбрать, какие 2 инициативы делать."
    ),
    "запуск функции": (
        "Продукт «{product}»: {metric} {cur}{unit}, цель - {goal}{unit}. "
        "Планируем запуск новой функции при ограничениях: {constraint}."
    ),
    "рост метрики": (
        "Продукт «{product}»: {metric} стоит на {cur}{unit} уже {period}, цель - {goal}{unit}, "
        "бюджет - {budget}. Предложи план роста."
    ),
    "монетизация": (
        "Продукт «{product}»: {metric} {cur}{unit}, монетизация - {monetization}. "
        "Нужно поднять выручку на {lift}%, не задев {guardrail}."
    ),
    "удержание": (
        "Продукт «{product}»: {metric} упал с {prev}{unit} до {cur}{unit} за {period}. "
        "Цель - вернуть {goal}{unit}, ресурсы: {team} человека."
    ),
}

ACTIONS = {
    "приоритизация": [
        "провести RICE-скоринг бэклога и взять две top-инициативы",
        "разбить квартал на два discovery-цикла и отобрать инициативы по RICE",
    ],
    "запуск функции": [
        "выкатить функцию через feature flag на 10% пользователей",
        "запустить функцию в dark launch и проверить guardrail-метрики",
    ],
    "рост метрики": [
        "запустить A/B-тест улучшенного онбординга с целевым эффектом",
        "провести cohort analysis и усилить активацию новых пользователей",
    ],
    "монетизация": [
        "пересобрать пакинг тарифов и провести price testing",
        "ввести usage-based тариф для активных клиентов",
    ],
    "удержание": [
        "запустить reactivation-компанию по ушедшим пользователям",
        "улучшить time to value: новый onboarding-чеклист для новичков",
    ],
}

OWNERS = ["продакт-менеджер", "PM команды роста", "продакт направления", "владелец метрики"]
HORIZONS = ["1 месяц", "квартал", "6 недель", "2 месяца"]
CONSTRAINTS = [
    "команда занята текущими обязательствами", "нельзя менять API публично",
    "нет бюджета на маркетинг", "релизное окно раз в две недели",
]
MONETIZATION = ["freemium", "подписка", "комиссия с транзакции", "тариф по местам (seat-based)"]
GUARDRAILS = ["конверсию в оплату", "retention новых пользователей", "NPS", "активность ядра аудитории"]

EVAL_TEMPLATES = {
    "приоритизация": (
        "В «{product}» очередь из 15 задач, {metric} = {cur}{unit}, команда {team} чел., "
        "срок - {horizon}. Какие инициативы выбрать в первую очередь?"
    ),
    "запуск функции": (
        "Готовим в «{product}» новую функцию. {metric} = {cur}{unit} (цель {goal}{unit}). "
        "Ограничение: {constraint}. Как раскатывать?"
    ),
    "рост метрики": (
        "«{product}»: {metric} не растёт с {cur}{unit} уже {period}, цель {goal}{unit}. Что делать?"
    ),
    "монетизация": (
        "«{product}»: модель монетизации - {monetization}. "
        "Задача: +{lift}% выручки без падения {guardrail}. Как?"
    ),
    "удержание": (
        "В «{product}» {metric} снизился до {cur}{unit} (было {prev}{unit}). "
        "Команда: {team} чел. План удержания?"
    ),
}


def _build_glossary() -> list[dict]:
    rows = []
    for domain, pack in DOMAIN_PACKS.items():
        for term, definition in pack:
            rows.append({"term": term, "definition": definition, "domain": domain})
    return rows


def _situation(rng: random.Random, dtype: str, domain: str) -> tuple[str, dict]:
    product = rng.choice(PRODUCTS[domain])
    metric, unit = rng.choice(METRICS[domain])
    cur = rng.randint(2, 25) if unit == "%" else rng.choice([150_000, 400_000, 900_000])
    params = {
        "product": product,
        "metric": metric,
        "unit": unit,
        "cur": f"{cur:,}".replace(",", " ") if isinstance(cur, int) else cur,
        "goal": f"{cur * 1.2:.0f}" if unit == "%" else f"{cur * 2:,}".replace(",", " "),
        "team": rng.choice([2, 3, 4, 5]),
        "horizon": rng.choice(HORIZONS),
        "period": rng.choice(["3 месяца", "полгода", "6 недель"]),
        "budget": f"{rng.choice([300, 800, 1500])} тыс. руб",
        "constraint": rng.choice(CONSTRAINTS),
        "monetization": rng.choice(MONETIZATION),
        "guardrail": rng.choice(GUARDRAILS),
        "prev": f"{cur * 1.4:.0f}",
        "lift": rng.choice([10, 15, 20]),
    }
    return SITUATION_BY_TYPE[dtype].format(**params), params


def _decision(rng: random.Random, dtype: str, params: dict) -> str:
    action = rng.choice(ACTIONS[dtype])
    deadline = rng.choice(HORIZONS)
    owner = rng.choice(OWNERS)
    metric, unit = params["metric"], params["unit"]
    effect = rng.choice([2.0, 2.5, 3.0])
    return (
        f"Предлагаю {action}. Целевой эффект: {metric} {params['cur']}{unit} → "
        f"+{effect} п.п. за {deadline}. Замер - A/B-тест до/после, guardrail: {params['guardrail']}. "
        f"Отвечает: {owner}."
    )


def _outcome(rng: random.Random) -> str:
    good = rng.random() < 0.7
    if good:
        return f"Через {rng.choice(HORIZONS)}: метрика выросла, эффект подтверждён на A/B, решение масштабируется."
    return f"Через {rng.choice(HORIZONS)}: эффекта не получили - гипотеза не подтвердилась, направление закрыто."


def build_cases(count: int = 1100, seed: int = 42) -> list[dict]:
    """Синтетические кейсы: {situation, decision, outcome, domain, decision_type, quality}."""
    rng = random.Random(seed)
    domains = [d for d in DOMAIN_PACKS if d != "general"]
    types = list(SITUATION_BY_TYPE)
    per_cell = -(-count // (len(domains) * len(types)))
    cases = []
    for domain in domains:
        for dtype in types:
            for _ in range(per_cell):
                situation, params = _situation(rng, dtype, domain)
                cases.append({
                    "situation": situation,
                    "decision": _decision(rng, dtype, params),
                    "outcome": _outcome(rng),
                    "domain": domain,
                    "decision_type": dtype,
                    "quality": rng.choice([4, 5, 6, 7, 8, 9]),
                })
    return cases


def build_eval_set(size: int = EVAL_SIZE, seed: int = 43) -> list[dict]:
    """Eval-набор: другие шаблоны и числа - детектор утечки не должен их находить в памяти."""
    rng = random.Random(seed)
    domains = [d for d in DOMAIN_PACKS if d != "general"]
    types = list(EVAL_TEMPLATES)
    rows = []
    for i in range(size):
        domain = domains[i % len(domains)]
        dtype = types[i % len(types)]
        product = rng.choice(PRODUCTS[domain])
        metric, unit = rng.choice(METRICS[domain])
        cur = rng.randint(30, 80) if unit == "%" else rng.choice([120_000, 260_000, 540_000])
        params = {
            "product": product, "metric": metric, "unit": unit,
            "cur": f"{cur:,}".replace(",", " "),
            "goal": f"{cur * 1.3:.0f}", "team": rng.choice([2, 3, 4]),
            "horizon": rng.choice(HORIZONS), "period": rng.choice(["4 месяца", "полгода"]),
            "constraint": rng.choice(CONSTRAINTS), "monetization": rng.choice(MONETIZATION),
            "guardrail": rng.choice(GUARDRAILS), "prev": f"{cur * 1.35:.0f}",
            "lift": rng.choice([12, 18]),
        }
        situation = EVAL_TEMPLATES[dtype].format(**params)
        reference = _decision(rng, dtype, params)
        rows.append({
            "situation": situation,
            "domain": domain,
            "decision_type": dtype,
            "reference_decision": reference,
        })
    return rows


def generate_datasets(out_dir: Path) -> dict[str, int]:
    """Собрать и записать три датасета. Возвращает счётчики."""
    glossary = _build_glossary()
    cases = build_cases()
    eval_set = build_eval_set()
    if len(glossary) < MIN_GLOSSARY:
        raise ValueError(f"Глоссарий {len(glossary)} < {MIN_GLOSSARY} - требование задания")
    if len(cases) < MIN_CASES:
        raise ValueError(f"Кейсы {len(cases)} < {MIN_CASES} - требование задания")

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "glossary.json").write_text(
        json.dumps(glossary, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    (out_dir / "decision_memory.json").write_text(
        json.dumps(cases, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    (out_dir / "eval_set.json").write_text(
        json.dumps(eval_set, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    return {"glossary": len(glossary), "cases": len(cases), "eval": len(eval_set)}


if __name__ == "__main__":
    print(generate_datasets(Path("backend/data/datasets")))
