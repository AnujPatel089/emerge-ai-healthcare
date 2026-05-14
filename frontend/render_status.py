import pandas as pd
import streamlit as st

from frontend.api_client import api_request, json_or_error


def render_render_status(auth_headers):
    st.title("Render Deployment Status")
    data = json_or_error(api_request("GET", "/api/platform/render-status", headers=auth_headers()), {})
    if not data:
        return
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Render", "yes" if data.get("render_environment_detected") else "no")
    c2.metric("Database URL", "configured" if data.get("database_url_configured") else "missing")
    c3.metric("Secret Key", "configured" if data.get("secret_key_configured") else "missing")
    c4.metric("CORS", "configured" if data.get("cors_configured") else "missing")
    st.write("Backend URL:", data.get("backend_url") or "not set")
    st.write("Frontend URL:", data.get("frontend_url") or "not set")
    st.write("API_URL:", data.get("api_url") or "not set")
    st.dataframe(pd.DataFrame(data.get("checks", [])), use_container_width=True)
