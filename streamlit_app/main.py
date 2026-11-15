"""Точка входа Streamlit (этап 12): запуск - streamlit run streamlit_app/main.py.

Страницы лежат в streamlit_app/pages/ (нумерация = порядок в меню):
5_Глоссарий, 6_Кейсы, 7_HITL, 8_Evaluation - зона Коля;
1_Решение, 2_Batch, 3_Ensemble, 4_Health - зона Мадина (дозаполняются).
"""
import streamlit as st
from lib import auth_guard

st.set_page_config(page_title="Виртуальный продуктовый комитет", page_icon=None, layout="wide")

auth_guard()

st.title(" Виртуальный продуктовый комитет")
st.markdown(
    "Мультиагентная система принятия продуктовых решений: опишите ситуацию - "
    "виртуальная команда (Analyst, Validator, Critic, Editor) выработает и "
    "обоснует решение, а вы подтвердите его через очередь HITL."
)
st.page_link("pages/5_Глоссарий.py", label=" Глоссарий")
st.page_link("pages/6_Кейсы.py", label=" База кейсов")
st.page_link("pages/7_HITL.py", label=" HITL-очередь")
st.page_link("pages/8_Evaluation.py", label=" Evaluation")
