"""Страница 1: принятие решения - главная функция системы."""
import streamlit as st
from lib import api_request, auth_guard

st.set_page_config(page_title="Решение", page_icon=None, layout="wide")
auth_guard()
st.title(" Принятие решения")

with st.form("decide"):
    situation = st.text_area(
        "Опишите продуктовую ситуацию (продукт, метрики, цель, ограничения)",
        height=140,
        placeholder="Отток 8%/мес, цель 5%, команда 2 человека, срок 1 месяц",
    )
    col1, col2, col3 = st.columns(3)
    with col1:
        domain = st.selectbox("Домен", ["general", "b2b_saas", "marketplace", "fintech", "media"])
    with col2:
        mode = st.selectbox(
            "Ablation-режим (состав команды)",
            ["full", "baseline", "glossary", "tm", "tm_hybrid", "agents_only", "tm_hybrid_critic", "tm_hybrid_editor"],
        )
    with col3:
        decision_type = st.text_input("Тип решения (необязательно)")
    submitted = st.form_submit_button("Принять решение", type="primary", use_container_width=True)

if submitted:
    if len(situation.strip()) < 10:
        st.error("Опишите ситуацию подробнее (минимум 10 символов)")
        st.stop()
    with st.spinner("Виртуальная команда работает: Analyst -> Validator -> Critic -> Editor..."):
        response = api_request("POST", "/decide", json={
            "situation": situation.strip(),
            "domain": domain,
            "mode": mode,
            "decision_type": decision_type or None,
        })
    if response.status_code != 200:
        st.error(f"Ошибка API ({response.status_code}): {response.text}")
        st.stop()

    body = response.json()
    badge = " из кэша" if body["cached"] else f" {body['iterations']} итераций"
    st.success(f"Решение ({badge}, канал: {body['retrieval_backend']})")
    st.markdown(body["decision"])

    col1, col2, col3 = st.columns(3)
    col1.metric("Оценка Critic'а", f"{body['critic_score']}/10" if body["critic_score"] is not None else "-")
    col2.metric("Валидация", "пройдена" if body["validation_passed"] else "с нарушениями")
    col3.metric("Стоп-причина", body["stopped_reason"])

    with st.expander(" reasoning_log - как команда пришла к решению"):
        for line in body["reasoning_log"]:
            st.markdown(f"- {line}")

    if body["engine_fallback"]:
        st.warning("Ollama была недоступна - использован HuggingFace fallback.")
