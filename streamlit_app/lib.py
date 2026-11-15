"""Общая библиотека Streamlit-приложения: API-клиент, логин, auth-guard.

Правило разделения: Streamlit - только отображение, вся логика в FastAPI.
"""
import os

import httpx
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")


def api_request(method: str, path: str, *, json: dict | None = None, files=None) -> httpx.Response:
    """Запрос к API с токеном из session_state; 401 сбрасывает сессию."""
    headers = {}
    if st.session_state.get("token"):
        headers["Authorization"] = f"Bearer {st.session_state['token']}"
    response = httpx.request(
        method, f"{API_URL}{path}", json=json, files=files, headers=headers, timeout=600
    )
    if response.status_code == 401:
        st.session_state.clear()
        st.error("Сессия истекла - войдите заново")
        st.stop()
    return response


def auth_guard() -> None:
    """Пускает дальше только авторизованного; иначе показывает форму входа."""
    if st.session_state.get("token") and st.session_state.get("user"):
        st.sidebar.success(f"Вы вошли как {st.session_state['user']['username']}")
        if st.sidebar.button("Выйти"):
            st.session_state.clear()
            st.rerun()
        return

    st.title("Вход в Виртуальный продуктовый комитет")
    with st.form("login"):
        email = st.text_input("Email")
        password = st.text_input("Пароль", type="password")
        if st.form_submit_button("Войти"):
            response = httpx.post(
                f"{API_URL}/auth/login", json={"email": email, "password": password}, timeout=30
            )
            if response.status_code == 200:
                token = response.json()["access_token"]
                me = httpx.get(
                    f"{API_URL}/auth/me", headers={"Authorization": f"Bearer {token}"}, timeout=30
                ).json()
                st.session_state["token"] = token
                st.session_state["user"] = me
                st.rerun()
            else:
                st.error("Неверный email или пароль")
    st.stop()
