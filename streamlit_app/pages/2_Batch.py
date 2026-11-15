"""Страница 2: пакетная обработка ситуаций."""
import json

import streamlit as st
from lib import api_request, auth_guard

st.set_page_config(page_title="Batch", page_icon=None)
auth_guard()
st.title(" Пакетная обработка")
st.markdown("Файл JSON: [{situation, domain?, mode?}] - до 20 ситуаций за раз.")

uploaded = st.file_uploader("Файл с ситуациями", type=["json"])
if uploaded and st.button("Обработать пакет"):
    try:
        items = json.loads(uploaded.getvalue())
    except json.JSONDecodeError as exc:
        st.error(f"Некорректный JSON: {exc}")
        st.stop()

    with st.spinner(f"Обработка {len(items)} ситуаций..."):
        response = api_request("POST", "/decide/batch", json={"items": items})
    if response.status_code != 200:
        st.error(f"Ошибка API: {response.text}")
        st.stop()

    results = response.json()["results"]
    ok = sum(1 for r in results if r["status"] == "ok")
    st.metric("Успешно", f"{ok}/{len(results)}")

    for item in results:
        if item["status"] == "ok":
            body = item["response"]
            with st.expander(f"#{item['index']} score={body['critic_score']} ({body['eval_mode']})"):
                st.markdown(body["decision"])
        else:
            with st.expander(f"#{item['index']} ошибка"):
                st.error(item["error"])
