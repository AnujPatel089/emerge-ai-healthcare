import os
import time
from typing import Any

import requests
import streamlit as st


API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")


def api_request(method: str, path: str, headers: dict | None = None, retries: int = 2, timeout: int = 15, **kwargs):
    headers = headers or {}
    last_error = None
    for attempt in range(retries + 1):
        try:
            # Render Free Tier services can sleep. A short retry window smooths
            # wake-up without exposing raw connection errors to clinicians/admins.
            response = requests.request(method, f"{API_URL}{path}", headers=headers, timeout=timeout, **kwargs)
            if response.status_code >= 500 and attempt < retries:
                time.sleep(0.5 * (attempt + 1))
                continue
            return response
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(0.5 * (attempt + 1))
                continue
    st.error("Backend is temporarily unavailable.")
    st.info("Render free tier may be waking up. Try again in a few seconds.")
    st.caption("System is running in degraded mode. Manual review required.")
    if last_error:
        st.caption(str(last_error))
    return None


def json_or_error(response, fallback: Any = None):
    if response is None:
        return fallback
    try:
        data = response.json()
    except Exception:
        st.error("Backend returned an unexpected response.")
        return fallback
    if response.status_code >= 400:
        st.error(data.get("message", "Request failed."))
        st.caption(data.get("detail", "Manual review required."))
        return fallback
    if data.get("prediction_source") == "fallback_rules":
        st.warning("Prediction fallback is active. Clinician review required.")
    return data
