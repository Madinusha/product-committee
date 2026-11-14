"""Страница 5: глоссарий продукт-менеджмента (просмотр + admin-импорт)."""

import streamlit as st
from lib import api_request, auth_guard

st.set_page_config(page_title="Глоссарий", page_icon=None)
auth_guard()
st.title(" Глоссарий продукт-менеджмента")

offset = st.number_input("Смещение", min_value=0, value=0, step=50)
response = api_request("GET", f"/admin/glossary?offset={offset}&limit=50")
if response.status_code == 200:
    entries = response.json()["entries"]
    st.dataframe(
        [{"term": e["term"], "определение": e["definition"], "домен": e["domain"]} for e in entries],
        use_container_width=True,
        hide_index=True,
    )
elif response.status_code == 403:
    st.info("Список глоссария доступен роли admin.")
else:
    st.error(f"Ошибка API: {response.status_code}")

if st.session_state.get("user", {}).get("role") == "admin":
    st.subheader("Импорт (JSON: [{term, definition, domain?}])")
    uploaded = st.file_uploader("Файл глоссария", type=["json", "csv"])
    if uploaded and st.button("Импортировать"):
        upload = api_request(
            "POST", "/admin/glossary/import",
            files={"file": (uploaded.name, uploaded.getvalue())},
        )
        if upload.status_code == 200:
            report = upload.json()
            st.success(f"Создано: {report['created']}, пропущено: {report['skipped']}")
        else:
            st.error(upload.text)
