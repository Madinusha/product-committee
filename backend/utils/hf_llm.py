"""Фабрика HuggingFace LLM (синглтоны, выгрузка из памяти).

Полноценный бэкенд подключается на этапе 8 (fallback, когда Ollama выключен).
transformers импортируется лениво — модуль импортируется без тяжёлых
зависимостей, поэтому тесты не требуют torch.
"""
import gc
import logging
from typing import Any

from backend.core.config import get_settings

logger = logging.getLogger("backend.hf")

_registry: dict[str, dict[str, Any]] = {}


def _load(model_name: str) -> dict[str, Any]:
    """Ленивая загрузка tokenizer+model (transformers импортируется здесь)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info("Загрузка HF-модели %s ...", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype="auto", device_map="auto"
    )
    return {"tokenizer": tokenizer, "model": model}


def get_hf_llm(model_name: str | None = None) -> dict[str, Any]:
    """Синглтон по имени модели: повторный вызов не перезагружает веса."""
    name = model_name or get_settings().HF_MODEL_NAME
    if name not in _registry:
        _registry[name] = _load(name)
    return _registry[name]


async def generate_hf(
    prompt: str, system: str | None = None, max_new_tokens: int | None = None
) -> str:
    """Генерация через HF; вызывается только на fallback-пути."""
    import asyncio

    settings = get_settings()
    bundle = get_hf_llm()
    messages = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": prompt}
    ]
    inputs = bundle["tokenizer"].apply_chat_template(
        messages, return_tensors="pt", add_generation_prompt=True
    ).to(bundle["model"].device)

    def _generate() -> str:
        output = bundle["model"].generate(
            inputs, max_new_tokens=max_new_tokens or settings.HF_MAX_NEW_TOKENS, do_sample=False
        )
        return bundle["tokenizer"].decode(output[0][inputs.shape[1]:], skip_special_tokens=True)

    return await asyncio.to_thread(_generate)


def unload_hf_llm(model_name: str | None = None) -> None:
    """Выгрузить модель из памяти (обязателен между тестами)."""
    name = model_name or get_settings().HF_MODEL_NAME
    bundle = _registry.pop(name, None)
    if bundle is None:
        return
    del bundle["model"], bundle["tokenizer"]
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    logger.info("HF-модель %s выгружена", name)


def unload_all() -> None:
    for name in list(_registry):
        unload_hf_llm(name)
