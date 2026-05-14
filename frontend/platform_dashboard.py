import os
import time

import pandas as pd
import requests
import streamlit as st


API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")


def api_request(method: str, path: str, headers: dict, retries: int = 2, timeout: int = 12, **kwargs):
    last_error = None
    for attempt in range(retries + 1):
        try:
            response = requests.request(method, f"{API_URL}{path}", headers=headers, timeout=timeout, **kwargs)
            if response.status_code >= 500 and attempt < retries:
                time.sleep(0.5 * (attempt + 1))
                continue
            return response
        except requests.exceptions.Timeout as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(0.5 * (attempt + 1))
                continue
        except requests.exceptions.ConnectionError as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(0.5 * (attempt + 1))
                continue
    st.error(f"Could not reach backend at {API_URL}. Check Render service status and API_URL.")
    if last_error:
        st.caption(str(last_error))
    return None


def _json_or_error(response):
    if response is None:
        return None
    try:
        data = response.json()
    except Exception:
        st.error("Backend returned a non-JSON response.")
        st.caption(response.text[:500])
        return None
    if response.status_code >= 400:
        st.error(data.get("message", "Request failed."))
        if data.get("detail"):
            st.caption(data["detail"])
        return None
    return data


def render_platform_dashboard(role: str, auth_headers):
    is_admin = role in ["admin", "super_admin"]
    if role not in ["doctor", "admin", "super_admin"]:
        st.error("You do not have access to platform health.")
        return

    st.title("Platform Health" if is_admin else "System Status")
    headers = auth_headers()

    health = _json_or_error(api_request("GET", "/api/platform/health", headers))
    status = _json_or_error(api_request("GET", "/api/platform/status", headers))
    reliability = _json_or_error(api_request("GET", "/api/platform/ai-reliability", headers))

    if not health or not status:
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Backend", health.get("status", "unknown"))
    c2.metric("Database", health.get("database", "unknown"))
    c3.metric("Model Loaded", "yes" if health.get("model_loaded") else "no")
    c4.metric("Render", "yes" if health.get("render_environment") else "no")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Model Version", health.get("model_version", "unknown"))
    c6.metric("API Latency ms", health.get("api_latency_ms") or "n/a")
    c7.metric("Failed Predictions", health.get("failed_predictions", 0))
    c8.metric("Active Incidents", len(status.get("active_incidents", [])))

    c9, c10 = st.columns(2)
    c9.metric("Queue Count", health.get("queue_count", 0))
    c10.metric("Active Nurses", health.get("active_nurses", 0))

    if reliability:
        st.subheader("AI Reliability")
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Status", reliability.get("status", "unknown"))
        r2.metric("Low Confidence", reliability.get("low_confidence_predictions", 0))
        r3.metric("Possible Under-Triage", reliability.get("possible_under_triage_patterns", 0))
        r4.metric("Override Rate", reliability.get("doctor_override_rate", 0))
        st.info(reliability.get("message", "AI-supported triage only. Clinician review required."))
        alerts = reliability.get("alerts", [])
        if alerts:
            st.dataframe(pd.DataFrame(alerts), use_container_width=True)

    incidents = status.get("active_incidents", [])
    st.subheader("Active Incidents")
    if incidents:
        st.dataframe(pd.DataFrame(incidents), use_container_width=True)
    else:
        st.success("No active platform incidents.")

    if is_admin:
        st.subheader("Render Deployment Validation")
        validation = _json_or_error(api_request("GET", "/api/platform/deployment-validation", headers))
        if validation:
            st.metric("Validation", validation.get("status", "unknown"))
            st.dataframe(pd.DataFrame(validation.get("checks", [])), use_container_width=True)
            st.caption("Keep PostgreSQL as the source of truth for production monitoring data on Render.")
