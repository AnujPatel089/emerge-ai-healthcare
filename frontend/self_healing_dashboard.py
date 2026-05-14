import pandas as pd
import streamlit as st

from frontend.api_client import api_request, json_or_error


def render_self_healing_dashboard(role: str, auth_headers):
    readonly = role == "doctor"
    st.title("Self-Healing System" if not readonly else "System Status")
    headers = auth_headers()

    status = json_or_error(api_request("GET", "/api/platform/self-healing/status", headers=headers), {})
    if not status:
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Overall", status.get("overall_status", "unknown"))
    c2.metric("Database", status.get("database", "unknown"))
    c3.metric("Model", status.get("model", "unknown"))
    c4.metric("Queue", status.get("queue", "unknown"))
    st.info(status.get("message", "AI-supported triage only. Clinician review required."))

    c5, c6, c7 = st.columns(3)
    c5.metric("Prediction", status.get("prediction", "unknown"))
    c6.metric("Upload/OCR", status.get("upload_ocr", "unknown"))
    c7.metric("Active Incidents", status.get("active_incidents", 0))

    if not readonly:
        a, b, c = st.columns(3)
        if a.button("Run Health Check", use_container_width=True):
            checked = json_or_error(api_request("POST", "/api/platform/self-healing/run-checks", headers=headers), {})
            st.json(checked)
        if b.button("Attempt Auto-Recovery", use_container_width=True):
            recovery = json_or_error(api_request("POST", "/api/platform/self-healing/recover", headers=headers), {})
            st.json(recovery)
        if c.button("Refresh Status", use_container_width=True):
            st.rerun()

    incidents = status.get("incidents", [])
    st.subheader("Incidents")
    if incidents:
        st.dataframe(pd.DataFrame(incidents), use_container_width=True)
        if not readonly:
            ids = [item["id"] for item in incidents]
            selected = st.selectbox("Resolve Incident", ids)
            if st.button("Resolve Selected Incident"):
                result = json_or_error(api_request("POST", f"/api/platform/self-healing/resolve-incident/{selected}", headers=headers), {})
                st.json(result)
    else:
        st.success("No active self-healing incidents.")
