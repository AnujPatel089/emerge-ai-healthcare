import streamlit as st

from frontend.api_client import api_request, json_or_error


def render_model_governance(auth_headers):
    st.title("Model Governance")
    data = json_or_error(api_request("GET", "/api/mlops/governance", headers=auth_headers()), {})
    if not data:
        return
    c1, c2 = st.columns(2)
    c1.metric("Production Model", data.get("production_model_version", "unknown"))
    c2.metric("Status", data.get("model_status", "unknown"))
    st.subheader("Metrics")
    st.json(data.get("metrics", {}))
    st.info(data.get("safety_disclaimer", "Possible risk only. Clinician review required."))
    st.subheader("Promotion History")
    st.dataframe(data.get("promotion_history", []), use_container_width=True)
    version = data.get("production_model_version")
    if version:
        card = json_or_error(api_request("GET", f"/api/mlops/governance/model-card/{version}", headers=auth_headers()), {})
        if card:
            st.markdown(card.get("card_markdown", ""))
