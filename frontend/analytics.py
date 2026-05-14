"""
analytics.py — Predictive Analytics Page
EmergeAI Healthcare

Run standalone:
    streamlit run frontend/analytics.py

Or integrate into app.py navigation by importing and calling show_analytics_page().
"""

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import json
import os

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

# -------------------------------------------------------
# Page config (only used when run standalone)
# -------------------------------------------------------
try:
    st.set_page_config(
        page_title="EmergeAI — Predictive Analytics",
        page_icon="📈",
        layout="wide"
    )
except Exception:
    pass  # already set when imported into app.py


# -------------------------------------------------------
# Helpers
# -------------------------------------------------------

def auth_headers():
    return {"Authorization": f"Bearer {st.session_state.get('token', '')}"}


def handle_error(response):
    if response.status_code == 401:
        st.error("Session expired. Please login again.")
        st.stop()
    else:
        st.error(f"API error {response.status_code}: {response.text}")


def safe_json(value):
    if not value:
        return None
    try:
        return json.loads(value)
    except Exception:
        return value


def fetch_history():
    r = requests.get(f"{API_URL}/history", headers=auth_headers(), timeout=20)
    if r.status_code == 200:
        return r.json().get("history", [])
    handle_error(r)
    return []


def fetch_feedback():
    r = requests.get(f"{API_URL}/clinical-feedback", headers=auth_headers(), timeout=20)
    if r.status_code == 200:
        return r.json()
    return []


def fetch_dashboard(hours=24):
    r = requests.get(
        f"{API_URL}/v2/dashboard/summary",
        params={"hours": hours},
        headers=auth_headers(), timeout=15
    )
    if r.status_code == 200:
        return r.json()
    return {}


# -------------------------------------------------------
# Standalone login gate
# -------------------------------------------------------

def _login_gate():
    if not st.session_state.get("token"):
        st.title("📈 EmergeAI Analytics")
        st.warning("Please log in first.")
        u = st.text_input("Username")
        p = st.text_input("Password", type="password")
        if st.button("Login"):
            r = requests.post(
                f"{API_URL}/login",
                json={"username": u, "password": p}, timeout=10
            )
            if r.status_code == 200:
                d = r.json()
                st.session_state.token = d["access_token"]
                st.session_state.role = d["role"]
                st.session_state.username = u
                st.rerun()
            else:
                st.error("Invalid credentials.")
        st.stop()


# -------------------------------------------------------
# Main analytics page
# -------------------------------------------------------

def show_analytics_page():
    _login_gate()

    st.markdown("## 📈 Predictive Analytics")
    st.caption("Trends, patterns, and insights from your triage prediction history.")

    # ---- Filters ----
    f1, f2, f3 = st.columns(3)
    with f1:
        days_back = st.selectbox(
            "Time range", [1, 3, 7, 14, 30], index=2,
            format_func=lambda d: f"Last {d} day{'s' if d > 1 else ''}"
        )
    with f2:
        esi_filter = st.multiselect(
            "Filter ESI levels",
            ["1.0", "2.0", "3.0", "4.0", "5.0"],
            default=[]
        )
    with f3:
        arrival_filter = st.multiselect(
            "Filter arrival mode",
            ["Walk-in", "Ambulance", "Police", "Other"],
            default=[]
        )

    # ---- Load data ----
    with st.spinner("Loading analytics data..."):
        history = fetch_history()
        feedback = fetch_feedback()
        dashboard = fetch_dashboard(hours=days_back * 24)

    if not history:
        st.info("No prediction data found. Run some predictions first.")
        return

    # Build DataFrame
    df = pd.DataFrame(history)
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    df = df.dropna(subset=["created_at"])

    # Apply time filter
    cutoff = datetime.utcnow() - timedelta(days=days_back)
    df = df[df["created_at"] >= cutoff]

    # Apply optional filters
    if esi_filter:
        df = df[df["final_prediction"].astype(str).isin(esi_filter)]
    if arrival_filter:
        df = df[df["arrivalmode"].isin(arrival_filter)]

    if df.empty:
        st.warning("No data matches the selected filters.")
        return

    # -------------------------------------------------------
    # Section 1 — KPI row
    # -------------------------------------------------------
    st.markdown("---")
    st.markdown("### 🔢 Key Metrics")

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Total Predictions", len(df))

    avg_conf = df["confidence"].mean() if "confidence" in df.columns else 0
    k2.metric("Avg Confidence", f"{avg_conf:.1%}")

    high_acuity = df[df["final_prediction"].astype(str).isin(["1.0", "2.0"])]
    k3.metric("High Acuity (ESI 1-2)", len(high_acuity))

    override_count = len([f for f in feedback if not f.get("accepted", True)])
    k4.metric("Clinician Overrides", override_count)

    ambulance = df[df["arrivalmode"] == "Ambulance"] if "arrivalmode" in df.columns else pd.DataFrame()
    k5.metric("Ambulance Arrivals", len(ambulance))

    avg_hr = df["triage_vital_hr"].mean() if "triage_vital_hr" in df.columns else 0
    k6.metric("Avg Heart Rate", f"{avg_hr:.0f} bpm")

    # -------------------------------------------------------
    # Section 2 — Prediction volume over time
    # -------------------------------------------------------
    st.markdown("---")
    st.markdown("### 📅 Prediction Volume Over Time")

    df_time = df.copy()
    df_time["hour"] = df_time["created_at"].dt.floor("h")
    volume = df_time.groupby("hour").size().reset_index(name="count")

    fig_vol = px.area(
        volume, x="hour", y="count",
        title=f"Predictions per hour — last {days_back} day(s)",
        color_discrete_sequence=["#0b3d91"],
        labels={"hour": "Time", "count": "Predictions"}
    )
    fig_vol.update_layout(
        plot_bgcolor="#f8faff",
        paper_bgcolor="white",
        hovermode="x unified"
    )
    st.plotly_chart(fig_vol, use_container_width=True)

    # -------------------------------------------------------
    # Section 3 — ESI distribution + confidence
    # -------------------------------------------------------
    st.markdown("---")
    col_esi, col_conf = st.columns(2)

    with col_esi:
        st.markdown("### 🎯 ESI Level Distribution")
        esi_counts = df["final_prediction"].astype(str).value_counts().reset_index()
        esi_counts.columns = ["ESI Level", "Count"]
        esi_counts = esi_counts.sort_values("ESI Level")

        color_map = {
            "1.0": "#b00020", "2.0": "#e65100",
            "3.0": "#f9a825", "4.0": "#2e7d32", "5.0": "#1565c0"
        }
        fig_esi = px.bar(
            esi_counts, x="ESI Level", y="Count",
            color="ESI Level", color_discrete_map=color_map,
            title="Distribution of final ESI predictions"
        )
        fig_esi.update_layout(showlegend=False, plot_bgcolor="#f8faff")
        st.plotly_chart(fig_esi, use_container_width=True)

    with col_conf:
        st.markdown("### 📊 Confidence by ESI Level")
        if "confidence" in df.columns:
            conf_df = df[["final_prediction", "confidence"]].copy()
            conf_df["ESI Level"] = conf_df["final_prediction"].astype(str)
            conf_df["confidence"] = pd.to_numeric(conf_df["confidence"], errors="coerce")
            conf_df = conf_df.dropna(subset=["confidence"])

            fig_conf = px.box(
                conf_df, x="ESI Level", y="confidence",
                color="ESI Level", color_discrete_map=color_map,
                title="Model confidence distribution per ESI level",
                labels={"confidence": "Confidence"}
            )
            fig_conf.update_layout(showlegend=False, plot_bgcolor="#f8faff")
            st.plotly_chart(fig_conf, use_container_width=True)
        else:
            st.info("No confidence data available.")

    # -------------------------------------------------------
    # Section 4 — Vitals heatmap
    # -------------------------------------------------------
    st.markdown("---")
    st.markdown("### 🌡️ Vitals Profile by ESI Level")

    vital_cols = [
        "triage_vital_hr", "triage_vital_sbp", "triage_vital_dbp",
        "triage_vital_rr", "triage_vital_o2", "triage_vital_temp"
    ]
    vital_labels = ["Heart Rate", "Systolic BP", "Diastolic BP",
                    "Resp Rate", "O2 Sat", "Temperature"]

    available_vitals = [c for c in vital_cols if c in df.columns]
    if available_vitals:
        vitals_df = df[["final_prediction"] + available_vitals].copy()
        vitals_df["ESI"] = vitals_df["final_prediction"].astype(str)
        vitals_mean = vitals_df.groupby("ESI")[available_vitals].mean()

        # Normalize each column 0-1 for heatmap
        vitals_norm = (vitals_mean - vitals_mean.min()) / (
            vitals_mean.max() - vitals_mean.min() + 1e-9
        )
        vitals_norm.columns = [
            vital_labels[vital_cols.index(c)]
            for c in available_vitals
        ]

        fig_heat = px.imshow(
            vitals_norm,
            text_auto=".2f",
            color_continuous_scale="RdYlGn_r",
            title="Normalised mean vitals per ESI level (red = higher risk)",
            labels={"x": "Vital Sign", "y": "ESI Level", "color": "Normalised Value"}
        )
        fig_heat.update_layout(
            xaxis_title="Vital Sign",
            yaxis_title="ESI Level"
        )
        st.plotly_chart(fig_heat, use_container_width=True)

    # -------------------------------------------------------
    # Section 5 — Arrival mode + time of day
    # -------------------------------------------------------
    st.markdown("---")
    col_arr, col_hour = st.columns(2)

    with col_arr:
        st.markdown("### 🚑 Arrival Mode Breakdown")
        if "arrivalmode" in df.columns:
            arr_df = df["arrivalmode"].value_counts().reset_index()
            arr_df.columns = ["Mode", "Count"]
            fig_arr = px.pie(
                arr_df, names="Mode", values="Count",
                title="Arrival mode distribution",
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            st.plotly_chart(fig_arr, use_container_width=True)
        else:
            st.info("No arrival mode data.")

    with col_hour:
        st.markdown("### ⏰ Predictions by Hour of Day")
        df["hour_of_day"] = df["created_at"].dt.hour
        hour_counts = df.groupby("hour_of_day").size().reset_index(name="count")

        fig_hour = px.bar(
            hour_counts, x="hour_of_day", y="count",
            title="Volume by hour of day (busiest periods)",
            labels={"hour_of_day": "Hour (24h)", "count": "Predictions"},
            color="count",
            color_continuous_scale="Blues"
        )
        fig_hour.update_layout(plot_bgcolor="#f8faff", showlegend=False)
        st.plotly_chart(fig_hour, use_container_width=True)

    # -------------------------------------------------------
    # Section 6 — Symptom frequency
    # -------------------------------------------------------
    st.markdown("---")
    st.markdown("### 🩺 Symptom Frequency Analysis")

    symptom_cols = {
        "cc_chestpain": "Chest Pain",
        "cc_shortnessofbreath": "Shortness of Breath",
        "cc_headache": "Headache",
        "cc_fever": "Fever",
        "cc_abdominalpain": "Abdominal Pain",
        "cc_dizziness": "Dizziness",
        "cc_syncope": "Syncope",
        "cc_weakness": "Weakness"
    }

    available_sym = {k: v for k, v in symptom_cols.items() if k in df.columns}
    if available_sym:
        sym_totals = {
            label: int(df[col].sum())
            for col, label in available_sym.items()
        }
        sym_df = pd.DataFrame({
            "Symptom": list(sym_totals.keys()),
            "Count": list(sym_totals.values())
        }).sort_values("Count", ascending=True)

        fig_sym = px.bar(
            sym_df, x="Count", y="Symptom", orientation="h",
            title="Total symptom occurrences across all predictions",
            color="Count",
            color_continuous_scale="Oranges"
        )
        fig_sym.update_layout(plot_bgcolor="#f8faff", showlegend=False)
        st.plotly_chart(fig_sym, use_container_width=True)

        # Symptom co-occurrence — which symptoms appear together
        st.markdown("#### Symptom Co-occurrence by ESI Level")
        sym_esi = df[["final_prediction"] + list(available_sym.keys())].copy()
        sym_esi["ESI"] = sym_esi["final_prediction"].astype(str)
        sym_means = sym_esi.groupby("ESI")[list(available_sym.keys())].mean()
        sym_means.columns = list(available_sym.values())

        fig_co = px.imshow(
            sym_means,
            text_auto=".0%",
            color_continuous_scale="Blues",
            title="Symptom presence rate per ESI level",
            labels={"x": "Symptom", "y": "ESI Level", "color": "Rate"}
        )
        st.plotly_chart(fig_co, use_container_width=True)

    # -------------------------------------------------------
    # Section 7 — Override analysis
    # -------------------------------------------------------
    st.markdown("---")
    st.markdown("### 🔄 Clinician Override Analysis")

    if feedback:
        fb_df = pd.DataFrame(feedback)
        fb_df["created_at"] = pd.to_datetime(fb_df["created_at"], errors="coerce")

        ov1, ov2 = st.columns(2)

        with ov1:
            accept_counts = fb_df["accepted"].map(
                {True: "Accepted", False: "Overridden"}
            ).value_counts().reset_index()
            accept_counts.columns = ["Decision", "Count"]
            fig_fb = px.pie(
                accept_counts, names="Decision", values="Count",
                title="AI accepted vs overridden",
                color="Decision",
                color_discrete_map={"Accepted": "#2e7d32", "Overridden": "#e65100"}
            )
            st.plotly_chart(fig_fb, use_container_width=True)

        with ov2:
            overrides = fb_df[fb_df["accepted"] == False].copy()
            if not overrides.empty and "ai_prediction" in overrides.columns:
                overrides["ai"] = overrides["ai_prediction"].astype(str)
                overrides["clinician"] = overrides["clinician_prediction"].astype(str)
                flow = overrides.groupby(
                    ["ai", "clinician"]
                ).size().reset_index(name="count")

                fig_flow = px.bar(
                    flow, x="ai", y="count",
                    color="clinician",
                    title="Override direction (AI → Clinician ESI)",
                    labels={"ai": "AI Predicted ESI", "count": "Count",
                            "clinician": "Clinician ESI"},
                    barmode="group"
                )
                fig_flow.update_layout(plot_bgcolor="#f8faff")
                st.plotly_chart(fig_flow, use_container_width=True)
            else:
                st.info("No override data yet.")

        # Override trend over time
        if not overrides.empty:
            overrides = overrides.dropna(subset=["created_at"])
            overrides["day"] = overrides["created_at"].dt.floor("D")
            override_trend = overrides.groupby("day").size().reset_index(name="overrides")
            fig_ot = px.line(
                override_trend, x="day", y="overrides",
                title="Override count per day",
                markers=True,
                color_discrete_sequence=["#e65100"]
            )
            fig_ot.update_layout(plot_bgcolor="#f8faff")
            st.plotly_chart(fig_ot, use_container_width=True)
    else:
        st.info("No clinical feedback recorded yet.")

    # -------------------------------------------------------
    # Section 8 — Demographics
    # -------------------------------------------------------
    st.markdown("---")
    st.markdown("### 👥 Patient Demographics")

    d1, d2, d3 = st.columns(3)

    with d1:
        if "gender" in df.columns:
            gen_df = df["gender"].value_counts().reset_index()
            gen_df.columns = ["Gender", "Count"]
            fig_g = px.pie(gen_df, names="Gender", values="Count",
                           title="Gender distribution",
                           color_discrete_sequence=["#0b3d91", "#e65100"])
            st.plotly_chart(fig_g, use_container_width=True)

    with d2:
        if "age" in df.columns:
            age_series = pd.to_numeric(df["age"], errors="coerce").dropna()
            fig_age = px.histogram(
                age_series, nbins=12,
                title="Age distribution",
                labels={"value": "Age", "count": "Patients"},
                color_discrete_sequence=["#0b3d91"]
            )
            fig_age.update_layout(plot_bgcolor="#f8faff", showlegend=False)
            st.plotly_chart(fig_age, use_container_width=True)

    with d3:
        if "race" in df.columns:
            race_df = df["race"].value_counts().reset_index()
            race_df.columns = ["Race", "Count"]
            fig_r = px.bar(
                race_df, x="Count", y="Race", orientation="h",
                title="Race distribution",
                color_discrete_sequence=["#0b3d91"]
            )
            fig_r.update_layout(plot_bgcolor="#f8faff")
            st.plotly_chart(fig_r, use_container_width=True)

    # -------------------------------------------------------
    # Section 9 — Raw data explorer
    # -------------------------------------------------------
    st.markdown("---")
    st.markdown("### 🔍 Raw Data Explorer")

    with st.expander("View filtered prediction data"):
        show_cols = [
            "id", "created_at", "age", "gender", "arrivalmode",
            "final_prediction", "ml_prediction", "confidence",
            "triage_vital_hr", "triage_vital_sbp", "triage_vital_o2",
            "source", "feedback"
        ]
        available_cols = [c for c in show_cols if c in df.columns]
        st.dataframe(
            df[available_cols].sort_values("created_at", ascending=False),
            use_container_width=True
        )
        csv = df[available_cols].to_csv(index=False)
        st.download_button(
            "⬇️ Download as CSV",
            data=csv,
            file_name=f"emergeai_analytics_{datetime.utcnow().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv"
        )


# -------------------------------------------------------
# Standalone entry point
# -------------------------------------------------------
if __name__ == "__main__" or True:
    show_analytics_page()
