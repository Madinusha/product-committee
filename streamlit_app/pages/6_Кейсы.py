"""Страница 6: база кейсов Decision Memory (просмотр + admin-импорт/удаление)."""
import streamlit as st
from lib import api_request, auth_guard

st.set_page_config(page_title="База кейсов", page_icon=None)
auth_guard()
st.title(" База кейсов Decision Memory")

offset = st.number_input("Смещение", min_value=0, value=0, step=50, key="cases_offset")
response = api_request("GET", f"/admin/cases?offset={offset}&limit=50")
if response.status_code == 200:
    entries = response.json()["entries"]
    st.dataframe(
        [{
            "ситуация": e["situation"][:80],
            "решение": e["decision"][:80],
            "результат": e["outcome"][:40],
            "домен": e["domain"],
            "оценка": e["quality"],
        } for e in entries],
        use_container_width=True,
        hide_index=True,
    )
elif response.status_code == 403:
    st.info("Список кейсов доступен роли admin.")
else:
    st.error(f"Ошибка API: {response.status_code}")

if st.session_state.get("user", {}).get("role") == "admin":
    st.subheader("Импорт кейсов (JSON: [{situation, decision, outcome, ...}])")
    uploaded = st.file_uploader("Файл кейсов", type=["json", "csv"])
    if uploaded and st.button("Импортировать кейсы"):
        upload = api_request(
            "POST", "/admin/cases/import",
            files={"file": (uploaded.name, uploaded.getvalue())},
        )
        if upload.status_code == 200:
            report = upload.json()
            st.success(f"Создано: {report['created']}, пропущено: {report['skipped']}")
        else:
            st.error(upload.text)
