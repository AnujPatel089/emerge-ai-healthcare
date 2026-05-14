import os

import pandas as pd
import plotly.express as px
import requests
import streamlit as st


API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")


def _get(path: str, headers: dict, timeout: int = 20):
    response = requests.get(f"{API_URL}{path}", headers=headers, timeout=timeout)
    if response.status_code >= 400:
        st.error(response.text)
        return None
    return response.json()


def _post(path: str, headers: dict, json=None, timeout: int = 60):
    response = requests.post(f"{API_URL}{path}", headers=headers, json=json, timeout=timeout)
    if response.status_code >= 400:
        st.error(response.text)
        return None
    return response.json()


def render_mlops_dashboard(role: str, auth_headers):
    headers = auth_headers()
    is_admin = role in ["admin", "super_admin"]
    st.title("MLOps Monitoring" if is_admin else "AI Model Health")

    health = _get("/api/mlops/model-health", headers)
    if not health:
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Model Version", health.get("model_version", "unknown"))
    c2.metric("Health", health.get("status", "unknown"))
    c3.metric("Predictions", health.get("prediction_count", 0))
    c4.metric("Drift", health.get("drift_status", "unknown"))

    c5, c6, c7 = st.columns(3)
    c5.metric("Avg Confidence", health.get("average_confidence") or "n/a")
    c6.metric("Override Rate", health.get("override_rate") or 0)
    c7.metric("Latency ms", health.get("latency_ms") or "n/a")
    st.info(health.get("recommendation", "Clinician review required for AI-supported triage."))

    esi_distribution = health.get("esi_distribution") or {}
    if esi_distribution:
        df = pd.DataFrame(
            [{"ESI": key, "Count": value} for key, value in esi_distribution.items()]
        )
        st.plotly_chart(px.bar(df, x="ESI", y="Count", title="ESI Distribution"), use_container_width=True)

    if not is_admin:
        return

    tab_monitoring, tab_registry, tab_drift, tab_retrain, tab_card, tab_deploy = st.tabs(
        ["Monitoring", "Registry", "Drift", "Retraining", "Model Card", "Render"]
    )

    with tab_monitoring:
        monitoring = _get("/api/mlops/prediction-monitoring", headers)
        rows = (monitoring or {}).get("predictions", [])
        if rows:
            table = pd.DataFrame(rows)
            st.dataframe(table, use_container_width=True)
        else:
            st.caption("No prediction monitoring rows yet.")

    with tab_registry:
        registry = _get("/api/mlops/model-registry", headers)
        models = (registry or {}).get("models", [])
        if models:
            table = pd.DataFrame(models)
            st.dataframe(table, use_container_width=True)
            staging_versions = [m["model_version"] for m in models if m.get("status") == "staging"]
            if staging_versions:
                selected = st.selectbox("Staging model to promote", staging_versions)
                if st.button("Promote Model", type="primary"):
                    result = _post("/api/mlops/promote-model", headers, json={"model_version": selected})
                    if result:
                        st.success(result.get("message", "Promotion completed."))
        else:
            st.caption("No registry records found.")

    with tab_drift:
        drift = _get("/api/mlops/drift-report", headers)
        if drift:
            st.metric("Drift Score", drift.get("drift_score"))
            st.metric("Drift Status", drift.get("drift_status"))
            feature_drift = drift.get("feature_drift") or {}
            if feature_drift:
                drift_df = pd.DataFrame(
                    [{"Feature": key, "Drift": value} for key, value in feature_drift.items()]
                )
                st.plotly_chart(px.bar(drift_df, x="Feature", y="Drift", title="Feature Drift"), use_container_width=True)
            st.info(drift.get("recommendation", "Continue monitoring."))
        if st.button("Refresh Drift Report"):
            refreshed = _post("/api/mlops/drift-report/refresh", headers)
            if refreshed:
                st.success("Drift report refreshed.")

    with tab_retrain:
        st.caption("Retraining is guarded for Render resources and never auto-deploys weak candidates.")
        if st.button("Run Safe Retraining", type="primary"):
            result = _post("/api/mlops/retrain", headers, timeout=120)
            if result:
                st.json(result)

    with tab_card:
        version = health.get("model_version", "production")
        card = _get(f"/api/mlops/model-card/{version}", headers)
        if card:
            st.markdown(card.get("card_markdown", ""))

    with tab_deploy:
        deployment = _get("/api/mlops/deployment-status", headers)
        if deployment:
            st.metric("Deployment Status", deployment.get("status"))
            st.caption(deployment.get("filesystem_warning", ""))
            st.dataframe(pd.DataFrame(deployment.get("checks", [])), use_container_width=True)
