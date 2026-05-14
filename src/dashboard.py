"""
EmergeAI — Advanced Clinical Dashboard
Run alongside your existing Streamlit app or replace the home page with this.

    streamlit run frontend/dashboard.py
"""

import os
import json
import requests
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

API = os.getenv("API_URL", os.getenv("EMERGEAI_API", "http://127.0.0.1:8000"))

st.set_page_config(
    page_title="EmergeAI — Clinical Dashboard",
    page_icon="🏥",
    layout="wide",
)

# ---------- Auth ----------
if "token" not in st.session_state:
    st.session_state.token = None

with st.sidebar:
    st.markdown("### 🔐 Login")
    if not st.session_state.token:
        u = st.text_input("Username")
        p = st.text_input("Password", type="password")
        role = st.selectbox("Role", ["super_admin", "admin", "doctor", "nurse", "patient"])
        if st.button("Sign in"):
            try:
                health = requests.get(f"{API}/health", timeout=3)
                if health.status_code != 200:
                    st.error("Backend is offline. Start FastAPI on port 8000 before logging in.")
                    st.stop()
                r = requests.post(
                    f"{API}/login",
                    json={"username": u, "password": p, "role": role},
                    timeout=10,
                )
                r.raise_for_status()
                st.session_state.token = r.json()["access_token"]
                st.session_state.user = u
                st.rerun()
            except Exception as e:
                st.error(f"Login failed: {e}")
    else:
        st.success(f"Signed in as {st.session_state.get('user','user')}")
        if st.button("Sign out"):
            st.session_state.token = None
            st.rerun()

if not st.session_state.token:
    st.info("Sign in to access the dashboard.")
    st.stop()

HEADERS = {"Authorization": f"Bearer {st.session_state.token}"}


# ---------- Tabs ----------
tab_overview, tab_triage, tab_image, tab_reports = st.tabs(
    ["📊 Overview", "🚦 Triage", "🩻 Image Analysis", "📄 Reports"]
)


# ============================================================
# Overview tab
# ============================================================
with tab_overview:
    st.markdown("## Clinical Overview")
    window = st.selectbox("Time window", [6, 12, 24, 48, 72], index=2,
                          format_func=lambda h: f"Last {h}h")

    try:
        s = requests.get(f"{API}/v2/dashboard/summary",
                         params={"hours": window},
                         headers=HEADERS, timeout=10).json()
    except Exception as e:
        st.error(f"Could not load summary: {e}")
        s = {}

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total predictions", s.get("total_predictions", 0))
    c2.metric("Images analyzed", s.get("total_images", 0))
    c3.metric("Avg composite risk", f"{s.get('avg_composite_risk', 0)} / 100")
    c4.metric("High-acuity (ESI 1–2)", s.get("high_acuity_count", 0),
              delta_color="inverse")

    esi = s.get("esi_distribution", {})
    if esi:
        df = pd.DataFrame({
            "ESI Level": [f"Level {k}" for k in esi.keys()],
            "Count": list(esi.values()),
        })
        fig = px.bar(df, x="ESI Level", y="Count",
                     color="ESI Level",
                     color_discrete_map={
                         "Level 1": "#b00020", "Level 2": "#e65100",
                         "Level 3": "#f9a825", "Level 4": "#2e7d32",
                         "Level 5": "#1565c0",
                     },
                     title=f"ESI Distribution — last {window}h")
        st.plotly_chart(fig, use_container_width=True)


# ============================================================
# Triage tab — multi-modal score
# ============================================================
with tab_triage:
    st.markdown("## Multi-Modal Triage")
    st.caption("Combines vitals + symptoms + NLP free-text + image analysis.")

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("### Patient")
        pid = st.text_input("Patient ID", value="P-DEMO")
        age = st.number_input("Age", 0, 120, 45)
        sex = st.selectbox("Sex", ["M", "F", "Other"])

        st.markdown("### Vitals")
        hr = st.number_input("Heart rate (bpm)", 0, 250, 90)
        sbp = st.number_input("Systolic BP (mmHg)", 0, 300, 120)
        rr = st.number_input("Respiratory rate", 0, 60, 16)
        spo2 = st.number_input("SpO₂ (%)", 0, 100, 97)
        temp = st.number_input("Temperature (°C)", 30.0, 45.0, 36.8, step=0.1)
        consc = st.selectbox("Consciousness",
                             ["alert", "voice", "pain", "unresponsive"])

    with col_r:
        st.markdown("### Symptoms")
        symptom_options = [
            "chest_pain", "shortness_of_breath", "abdominal_pain",
            "headache", "fever", "nausea", "dizziness",
            "loss_of_consciousness", "severe_bleeding",
        ]
        picked = st.multiselect("Symptoms", symptom_options)
        sev_map = {}
        for s in picked:
            sev_map[s] = st.select_slider(
                f"  • {s} severity",
                options=["mild", "moderate", "severe", "critical"],
                value="moderate", key=f"sev_{s}",
            )

        st.markdown("### Free-text complaint (NLP)")
        free_text = st.text_area("Chief complaint",
                                 placeholder="e.g. crushing chest pain radiating to left arm…")

        st.markdown("### Linked image")
        image_pid = st.text_input("Image prediction_id (optional)")

    if st.button("🚦 Run Multi-Modal Triage", type="primary"):
        # Call NLP first if there's free text
        nlp_findings = None
        if free_text.strip():
            try:
                nlp_findings = requests.post(
                    f"{API}/extract-symptoms",
                    json={"text": free_text}, headers=HEADERS, timeout=15
                ).json()
            except Exception:
                st.warning("NLP extraction unavailable; proceeding without it.")

        payload = {
            "patient_id": pid, "age": age, "sex": sex,
            "vitals": {
                "heart_rate": hr, "systolic_bp": sbp, "respiratory_rate": rr,
                "spo2": spo2, "temperature": temp, "consciousness": consc,
            },
            "structured_symptoms": [
                {"name": s, "severity": sev_map[s]} for s in picked
            ],
            "nlp_findings": nlp_findings,
            "image_prediction_id": image_pid or None,
        }

        try:
            r = requests.post(f"{API}/v2/triage", json=payload,
                              headers=HEADERS, timeout=20)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            st.error(f"Triage failed: {e}")
            st.stop()

        triage = data["triage"]
        st.session_state.last_triage_id = data["prediction_id"]

        # Headline
        level = triage["esi_level"]
        color = {1: "🔴", 2: "🟠", 3: "🟡", 4: "🟢", 5: "🔵"}[level]
        st.markdown(f"### {color} {triage['esi_label']}")
        cc1, cc2, cc3 = st.columns(3)
        cc1.metric("ESI Level", level)
        cc2.metric("Composite risk", f"{triage['composite_risk']} / 100")
        cc3.metric("Total score", triage["total_score"])

        # Component breakdown
        contribs = triage["contributions"]
        contrib_df = pd.DataFrame({
            "Modality": list(contribs.keys()),
            "Contribution": list(contribs.values()),
        })
        fig = px.bar(contrib_df, x="Contribution", y="Modality",
                     orientation="h",
                     title="Score contribution by modality",
                     color="Contribution",
                     color_continuous_scale="Reds")
        st.plotly_chart(fig, use_container_width=True)

        if triage.get("red_flags"):
            st.error(f"🚨 Red flags: {', '.join(triage['red_flags'])}")

        with st.expander("Raw triage JSON"):
            st.json(triage)

        st.success(f"prediction_id: `{data['prediction_id']}`")


# ============================================================
# Image tab — DL analysis
# ============================================================
with tab_image:
    st.markdown("## Deep-Learning Image Analysis")
    up = st.file_uploader("Upload wound/injury image",
                          type=["jpg", "jpeg", "png"])
    pid_img = st.text_input("Patient ID", key="img_pid")

    if up and st.button("🔬 Analyze (CNN + classical CV)"):
        files = {"file": (up.name, up.getvalue(), up.type)}
        data = {"patient_id": pid_img}
        try:
            r = requests.post(f"{API}/v2/analyze-image-dl",
                              files=files, data=data,
                              headers=HEADERS, timeout=60)
            r.raise_for_status()
            res = r.json()
        except Exception as e:
            st.error(f"Analysis failed: {e}")
            st.stop()

        a = res["analysis"]
        st.image(up, width=300)
        st.success(f"prediction_id: `{res['prediction_id']}`")

        c1, c2 = st.columns(2)
        c1.metric("Wound class", a["wound_class"],
                  f"{a['wound_confidence']:.0%} conf.")
        c2.metric("Infection severity", a["infection_severity"],
                  f"{a['infection_confidence']:.0%} conf.")

        # Distribution charts
        wd = pd.DataFrame({
            "class": list(a["wound_distribution"].keys()),
            "prob": list(a["wound_distribution"].values()),
        }).sort_values("prob", ascending=True)
        st.plotly_chart(
            px.bar(wd, x="prob", y="class", orientation="h",
                   title="Wound class probabilities"),
            use_container_width=True,
        )

        ind = pd.DataFrame({
            "level": list(a["infection_distribution"].keys()),
            "prob": list(a["infection_distribution"].values()),
        })
        st.plotly_chart(
            px.bar(ind, x="level", y="prob",
                   title="Infection severity probabilities",
                   color="level",
                   color_discrete_map={
                       "none": "#1565c0", "mild": "#2e7d32",
                       "moderate": "#f9a825", "severe": "#e65100",
                       "critical": "#b00020",
                   }),
            use_container_width=True,
        )

        st.markdown("**Classical CV flags:** " +
                    (", ".join(k for k, v in (a.get("cv_flags") or {}).items() if v)
                     or "_none_"))


# ============================================================
# Reports tab
# ============================================================
with tab_reports:
    st.markdown("## Clinical PDF Reports")
    st.caption("Generates a clinician-facing PDF for any prediction_id.")

    rid = st.text_input("prediction_id",
                        value=st.session_state.get("last_triage_id", ""))

    if rid and st.button("📄 Download PDF report"):
        try:
            r = requests.get(f"{API}/v2/report/{rid}",
                             headers=HEADERS, timeout=30)
            r.raise_for_status()
            st.download_button(
                "Save PDF",
                data=r.content,
                file_name=f"emergeai_report_{rid}.pdf",
                mime="application/pdf",
            )
        except Exception as e:
            st.error(f"Could not generate report: {e}")
