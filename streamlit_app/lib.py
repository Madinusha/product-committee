"""Общие утилиты Streamlit: API-клиент и auth-guard.

Токен и пользователь живут в st.session_state; API-адрес настраивается
переменной окружения API_BASE_URL (по умолчанию локальный backend).
"""
import os

import requests
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")


def api_request(method: str, path: str, **kwargs) -> requests.Response:
    """Запрос к backend-API с JWT из session_state (если есть).

    Возвращает requests.Response - статус-коды обрабатывают страницы
    (401 -> перелогин, 403 -> подсказка про роль admin).
    """
    headers = dict(kwargs.pop("headers", {}))
    token = st.session_state.get("token")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        return requests.request(method, API_BASE_URL + path, headers=headers, timeout=30, **kwargs)
    except requests.ConnectionError:
        # свой класс-заглушку делать не стал: страницам хватает .status_code != 200
        class _OfflineResponse:
            status_code = 503
            text = "backend недоступен"

            @staticmethod
            def json() -> dict:
                return {"detail": "backend недоступен"}

        return _OfflineResponse()  # type: ignore[return-value]


def auth_guard() -> None:
    """Пропускать только вошедших; остальных останавливаем с подсказкой."""
    if not st.session_state.get("user"):
        st.info("Требуется вход: откройте главную страницу (логин) и войдите.")
        st.stop()
