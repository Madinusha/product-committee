"""Страница 4: здоровье системы - readiness всех компонентов."""
import streamlit as st
from lib import api_request, auth_guard

st.set_page_config(page_title="Health", page_icon=None)
auth_guard()
st.title(" Состояние системы")

if st.button("Проверить", type="primary"):
    st.rerun()

health = api_request("GET", "/health")
cols = st.columns(5)
labels = {"app": "Приложение", "db": "База данных", "ollama": "Ollama", "qdrant": "Qdrant", "redis": "Redis"}

if health.status_code == 200:
    components = health.json()["components"]
    st.success("Все компоненты в порядке")
elif health.status_code == 503:
    components = health.json()["detail"]["components"]
    st.error("Есть упавшие компоненты")
else:
    components = {}
    st.error(f"API недоступен ({health.status_code})")

for (key, label), col in zip(labels.items(), cols, strict=False):
    state = components.get(key, "-")
    col.metric(label, "" if state == "ok" else "" if state == "down" else state)

st.subheader("Дальше по этапам")
st.markdown(
    "- Celery-воркер: `celery -A backend.worker.celery_app worker` (batch, переиндексация)\n"
    "- Переиндексация после HITL-подтверждений происходит автоматически"
)
