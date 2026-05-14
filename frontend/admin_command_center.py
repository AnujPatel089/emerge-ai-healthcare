import pandas as pd
import streamlit as st

from frontend.api_client import api_request, json_or_error


def render_admin_command_center(role: str, auth_headers):
    st.title("Admin Command Center")
    data = json_or_error(api_request("GET", "/api/command-center/summary", headers=auth_headers()), {})
    if not data:
        return
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Queue", data.get("active_emergency_queue_count", 0))
    c2.metric("Critical Patients", data.get("critical_patients", 0))
    c3.metric("Active Nurses", data.get("active_nurses", 0))
    c4.metric("Active Incidents", data.get("active_incidents", 0))
    c5, c6 = st.columns(2)
    c5.metric("Failed Predictions", data.get("failed_predictions", 0))
    c6.metric("Model Health", (data.get("model_health") or {}).get("status", "unknown"))
    st.subheader("Nurse Workload")
    st.dataframe(pd.DataFrame(data.get("nurse_workload", [])), use_container_width=True)
    st.subheader("Backend Health")
    st.json(data.get("backend_health", {}))
