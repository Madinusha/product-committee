"""Страница 8: evaluation - метрики поиска, статистика кэша, состояние стенда."""
import subprocess
import sys

import streamlit as st
from lib import api_request, auth_guard

st.set_page_config(page_title="Evaluation", page_icon=None)
auth_guard()
st.title(" Evaluation")

st.subheader("Состояние компонентов")
health = api_request("GET", "/health")
if health.status_code == 200:
    for component, state in health.json()["components"].items():
        st.metric(component, state)
else:
    detail = health.json().get("detail", {})
    for component, state in detail.get("components", {}).items():
        st.metric(component, state)

if st.session_state.get("user", {}).get("role") == "admin":
    st.subheader("Статистика кэша решений")
    stats = api_request("GET", "/admin/cache/stats")
    if stats.status_code == 200:
        data = stats.json()
        col1, col2, col3 = st.columns(3)
        col1.metric("Записей в кэше", data["entries"])
        col2.metric("Попаданий суммарно", data["total_hits"])
        col3.metric("Средний score", data["avg_critic_score"] or "-")

st.subheader("Retrieval-эксперимент (Recall@k / MRR)")
st.markdown(
    "Каналы BM25 / векторный / гибридный RRF на парах запрос-эталон "
    "(backend/data/datasets/retrieval_pairs.json). Запуск:"
)
st.code("python scripts/eval_retrieval.py --limit 200", language="bash")
if st.button("Запустить эксперимент (BM25-канал доступен всегда)"):
    result = subprocess.run(
        [sys.executable, "scripts/eval_retrieval.py", "--limit", "100"],
        capture_output=True, text=True, timeout=600,
    )
    st.code(result.stdout or result.stderr, language="text")

st.subheader("Калибровка порога similarity (0.7)")
st.markdown("Требует установленных эмбеддингов (poetry install):")
st.code("python scripts/calibrate_threshold.py --samples 100", language="bash")
