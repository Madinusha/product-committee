"""Главная страница Streamlit: логин и выход.

Запуск: streamlit run streamlit_app/main.py (страницы 5-8 подхватятся сами).
"""
import streamlit as st
from lib import API_BASE_URL, api_request

st.set_page_config(page_title="Виртуальный продуктовый комитет", page_icon=None)
st.title("Виртуальный продуктовый комитет")

if "user" not in st.session_state:
    st.session_state.user = None
if "token" not in st.session_state:
    st.session_state.token = None

if st.session_state.user:
    user = st.session_state.user
    st.success(f"Вы вошли как {user['username']} ({user['email']}, роль: {user['role']})")
    st.markdown("Навигация по разделам - в сайдбаре: Глоссарий, Кейсы, HITL, Evaluation.")
    if st.button("Выйти"):
        st.session_state.user = None
        st.session_state.token = None
        st.rerun()
else:
    with st.form("login"):
        email = st.text_input("Email")
        password = st.text_input("Пароль", type="password")
        submitted = st.form_submit_button("Войти")
    if submitted:
        response = api_request("POST", "/auth/login", json={"email": email, "password": password})
        if response.status_code == 200:
            st.session_state.token = response.json()["access_token"]
            me = api_request("GET", "/auth/me")
            if me.status_code == 200:
                st.session_state.user = me.json()
                st.rerun()
            else:  # токен есть, но /auth/me упал - не оставляем полусессию
                st.session_state.token = None
                st.error(f"Не удалось получить профиль: {me.status_code}")
        elif response.status_code == 401:
            st.error("Неверный email или пароль")
        else:
            st.error(f"Ошибка API ({response.status_code}). Backend: {API_BASE_URL}")

    st.markdown(
        "Нет аккаунта? Зарегистрируйтесь через API `POST /auth/register` "
        "или попросите админа создать его (scripts/create_admin.py)."
    )
