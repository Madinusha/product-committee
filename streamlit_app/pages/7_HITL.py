"""Страница 7: HITL-очередь - подтверждение решений продакт-владельцем."""
import streamlit as st
from lib import api_request, auth_guard

st.set_page_config(page_title="HITL-очередь", page_icon=None)
auth_guard()
st.title(" Human-in-the-Loop: очередь подтверждений")

response = api_request("GET", "/hitl/pending")
if response.status_code == 403:
    st.info("Очередь доступна ролям editor/admin.")
    st.stop()
if response.status_code != 200:
    st.error(f"Ошибка API: {response.status_code}")
    st.stop()

entries = response.json()["entries"]
if not entries:
    st.success("Очередь пуста - все решения обработаны.")
    st.stop()

for entry in entries:
    with st.expander(f"#{entry['id']} [{entry['domain']}] score={entry['critic_score']}"):
        st.markdown(f"**Ситуация:** {entry['situation']}")
        st.markdown(f"**Решение:** {entry['decision']}")
        col1, col2 = st.columns(2)
        if col1.button("Подтвердить", key=f"approve-{entry['id']}"):
            approve = api_request("POST", f"/hitl/{entry['id']}/approve")
            if approve.status_code == 200:
                created = approve.json()["memory_case_created"]
                st.success("Подтверждено. Кейс " + ("добавлен в Decision Memory." if created else "уже был в памяти."))
                st.rerun()
            else:
                st.error(approve.text)
        if col2.button("Отклонить", key=f"reject-{entry['id']}"):
            reject = api_request("POST", f"/hitl/{entry['id']}/reject")
            if reject.status_code == 200:
                st.warning("Отклонено - в Decision Memory не попало.")
                st.rerun()
            else:
                st.error(reject.text)

st.subheader("История")
history = api_request("GET", "/hitl/history")
if history.status_code == 200:
    st.dataframe(
        [{
            "id": e["id"], "статус": e["status"], "домен": e["domain"],
            "ситуация": e["situation"][:60],
        } for e in history.json()["entries"]],
        use_container_width=True,
        hide_index=True,
    )
