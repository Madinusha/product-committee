"""Страница 3: ансамбль - N независимых решений, выбор лучшего по оценке Critic'а."""
import streamlit as st
from lib import api_request, auth_guard

st.set_page_config(page_title="Ensemble", page_icon=None)
auth_guard()
st.title(" Ансамбль решений")

with st.form("ensemble"):
    situation = st.text_area(
        "Продуктовая ситуация",
        height=120,
        placeholder="Отток 8%/мес, цель 5%, команда 2 человека, срок 1 месяц",
    )
    col1, col2 = st.columns(2)
    with col1:
        domain = st.selectbox("Домен", ["general", "b2b_saas", "marketplace", "fintech", "media"])
    with col2:
        variants = st.slider("Число вариантов", 2, 5, 3)
    submitted = st.form_submit_button("Сгенерировать варианты", type="primary")

if submitted:
    with st.spinner(f"Генерация {variants} вариантов (это дольше обычного решения)..."):
        response = api_request("POST", "/decide/ensemble", json={
            "situation": situation.strip(),
            "domain": domain,
            "variants": variants,
        })
    if response.status_code != 200:
        st.error(f"Ошибка API ({response.status_code}): {response.text}")
        st.stop()

    body = response.json()
    st.success("Лучший вариант (по оценке Critic'а):")
    st.markdown(body["best"]["decision"])
    st.metric("Score лучшего", f"{body['best']['critic_score']}/10")

    st.subheader("Все варианты")
    st.dataframe(
        [{
            "вариант": i + 1,
            "score": v["critic_score"],
            "решение": v["decision"][:100],
        } for i, v in enumerate(body["variants"])],
        use_container_width=True,
        hide_index=True,
    )
