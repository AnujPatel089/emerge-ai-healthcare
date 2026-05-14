import pandas as pd
import streamlit as st

from frontend.api_client import api_request, json_or_error


def render_alerts_dashboard(role: str, auth_headers):
    st.title("Platform Alerts")
    headers = auth_headers()
    data = json_or_error(api_request("GET", "/api/platform/alerts", headers=headers), {})
    alerts = data.get("alerts", []) if data else []
    if alerts:
        st.dataframe(pd.DataFrame(alerts), use_container_width=True)
    else:
        st.success("No active alerts.")
    if role in ["admin", "super_admin"]:
        with st.expander("Create Alert"):
            alert_type = st.text_input("Alert Type", value="backend_degraded")
            severity = st.selectbox("Severity", ["warning", "critical", "healthy"])
            title = st.text_input("Title")
            message = st.text_area("Message")
            if st.button("Create Alert") and title:
                payload = {"alert_type": alert_type, "severity": severity, "title": title, "message": message}
                st.json(json_or_error(api_request("POST", "/api/platform/alerts/create", headers=headers, json=payload), {}))
        if alerts:
            selected = st.selectbox("Resolve Alert", [item["id"] for item in alerts])
            if st.button("Resolve Selected Alert"):
                st.json(json_or_error(api_request("POST", f"/api/platform/alerts/{selected}/resolve", headers=headers), {}))
