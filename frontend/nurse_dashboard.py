"""
Nurse dashboard for EmergeAI Healthcare.

Shows the nurse's assigned patients, triage workflow controls, vitals entry,
notes, and the option to send patients to doctor review.

Role access: nurse (sees own patients), admin/doctor (sees all assignments).
"""

from __future__ import annotations

import requests
import pandas as pd
import streamlit as st


NURSE_ASSIGNMENT_COLUMNS = [
    "Patient ID",
    "Patient Name",
    "ESI Level",
    "ICU Risk",
    "Assignment Status",
    "Assigned Time",
    "Priority Level",
]

NURSE_ACTIONS = ["Start Triage", "View History", "Upload Reports", "Add Notes", "Send to Doctor"]

STATUS_FLOW = {
    "assigned": ["in_triage"],
    "in_triage": ["doctor_review"],
    "doctor_review": ["completed"],
}

ESI_COLORS = {
    "ESI 1": "#dc2626",
    "ESI 2": "#ea580c",
    "ESI 3": "#ca8a04",
    "ESI 4": "#2563eb",
    "ESI 5": "#16a34a",
}


def _headers(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def _safe_get(api_url: str, path: str, token: str) -> dict | None:
    try:
        resp = requests.get(f"{api_url}{path}", headers=_headers(token), timeout=20)
        if resp.status_code == 200:
            return resp.json()
        if resp.status_code == 404:
            return None
        st.error(f"API error {resp.status_code}: {resp.json().get('detail', resp.text)}")
        return None
    except requests.exceptions.ConnectionError:
        st.error("Backend is offline. Start FastAPI with: `uvicorn backend.main:app --reload --port 8000`")
        return None
    except Exception as exc:
        st.error(f"Request failed: {exc}")
        return None


def _safe_post(api_url: str, path: str, payload: dict, token: str) -> dict | None:
    try:
        resp = requests.post(f"{api_url}{path}", json=payload, headers=_headers(token), timeout=20)
        if resp.status_code == 200:
            return resp.json()
        st.error(f"API error {resp.status_code}: {resp.json().get('detail', resp.text)}")
        return None
    except requests.exceptions.ConnectionError:
        st.error("Backend is offline. Start FastAPI with: `uvicorn backend.main:app --reload --port 8000`")
        return None
    except Exception as exc:
        st.error(f"Request failed: {exc}")
        return None


def _esi_color(esi_text: str) -> str:
    for key, color in ESI_COLORS.items():
        if key in str(esi_text):
            return color
    return "#6b7280"


def render_nurse_dashboard(api_url: str, token: str, role: str, username: str) -> None:
    st.title("Nurse Patient Dashboard")
    st.caption(
        "Your assigned patients and triage workflow. "
        "Nurses see only their assigned patients."
        if role == "nurse"
        else "All patient assignments and triage workflow."
    )

    col_refresh, _ = st.columns([1, 5])
    with col_refresh:
        if st.button("Refresh", use_container_width=True):
            st.rerun()

    # Load my patients
    data = _safe_get(api_url, "/api/assignments/my-patients", token)
    if data is None:
        return

    patients = data.get("patients", [])

    if not patients:
        st.info(
            "No assigned patients yet. "
            "Ask an admin or doctor to run auto-assignment, or wait for a prediction to be created."
        )
        return

    # Metrics
    total = len(patients)
    in_triage = sum(1 for p in patients if str(p.get("assignment_status", "")).lower() in ("in_triage", "in triage"))
    doctor_review = sum(1 for p in patients if "doctor" in str(p.get("assignment_status", "")).lower())
    critical = sum(1 for p in patients if str(p.get("priority_level", "")).lower() in ("critical", "high"))

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Assigned", total)
    m2.metric("In Triage", in_triage)
    m3.metric("Doctor Review", doctor_review)
    m4.metric("Critical/High", critical)

    st.divider()

    for patient in patients:
        prediction_id = patient.get("prediction_id") or patient.get("patient_id")
        patient_name = patient.get("patient_name") or f"Patient #{prediction_id}"
        esi_level = patient.get("esi_level") or "—"
        icu_risk = "Yes" if patient.get("icu_risk") else "No"
        assignment_status = str(patient.get("assignment_status") or patient.get("status") or "assigned").lower()
        priority = patient.get("priority_level") or "—"
        assigned_at = str(patient.get("assigned_at") or "—")[:19]
        nurse_info = patient.get("nurse") or {}
        nurse_name = nurse_info.get("name") or patient.get("nurse_name") or username
        nurse_id = nurse_info.get("id") or patient.get("nurse_id")

        border_color = _esi_color(str(esi_level))

        with st.container():
            st.markdown(
                f"""
                <div style="border-left:5px solid {border_color};padding:12px 18px;
                    border-radius:10px;background:#f8fafc;margin-bottom:6px;">
                  <span style="font-weight:700;font-size:16px;">{patient_name}</span>
                  &nbsp; <span style="color:#64748b;font-size:13px;">ID #{prediction_id}</span>
                  <br/>
                  <small style="color:#64748b;">
                    ESI: <b>{esi_level}</b> &nbsp;|&nbsp;
                    ICU Risk: <b>{icu_risk}</b> &nbsp;|&nbsp;
                    Priority: <b>{priority}</b> &nbsp;|&nbsp;
                    Status: <b>{assignment_status.replace("_", " ").title()}</b> &nbsp;|&nbsp;
                    Assigned: {assigned_at}
                  </small>
                </div>
                """,
                unsafe_allow_html=True,
            )

            with st.expander(f"Triage actions — {patient_name}"):
                tab1, tab2, tab3 = st.tabs(["Status & Workflow", "Record Vitals", "Nurse Notes"])

                with tab1:
                    st.markdown("**Workflow Progress**")
                    steps = ["assigned", "in_triage", "doctor_review", "completed"]
                    step_labels = ["Assigned", "In Triage", "Doctor Review", "Completed"]
                    current_idx = next(
                        (i for i, s in enumerate(steps) if s == assignment_status.lower().replace(" ", "_")), 0
                    )
                    progress_html = " → ".join(
                        f"<b style='color:#2563eb'>{label}</b>"
                        if i == current_idx
                        else f"<span style='color:#94a3b8'>{label}</span>"
                        for i, label in enumerate(step_labels)
                    )
                    st.markdown(progress_html, unsafe_allow_html=True)

                    st.write("")
                    next_statuses = STATUS_FLOW.get(assignment_status.lower(), [])
                    if next_statuses:
                        btn_cols = st.columns(len(next_statuses))
                        for col, next_status in zip(btn_cols, next_statuses):
                            label_map = {
                                "in_triage": "Start Triage",
                                "doctor_review": "Send to Doctor",
                                "completed": "Mark Completed",
                            }
                            btn_label = label_map.get(next_status, next_status.replace("_", " ").title())
                            with col:
                                if st.button(btn_label, key=f"flow_{prediction_id}_{next_status}", type="primary"):
                                    result = _safe_post(
                                        api_url,
                                        "/api/assignments/update-status",
                                        {"prediction_id": prediction_id, "status": next_status},
                                        token,
                                    )
                                    if result:
                                        st.success(f"Status updated to {next_status.replace('_', ' ').title()}.")
                                        st.rerun()
                    else:
                        st.info("This patient is completed or in final status.")

                with tab2:
                    st.markdown("**Record Nurse Vitals**")
                    if not nurse_id:
                        st.info("Nurse ID not found. Use the Nurse Management page to look up your nurse ID.")
                    else:
                        with st.form(f"vitals_{prediction_id}_{nurse_id}"):
                            vcol1, vcol2 = st.columns(2)
                            with vcol1:
                                temperature = st.number_input("Temperature (°F)", 80.0, 115.0, 98.6, 0.1)
                                heart_rate = st.number_input("Heart Rate (bpm)", 0, 250, 80)
                                blood_pressure = st.text_input("Blood Pressure", placeholder="120/80")
                            with vcol2:
                                oxygen_level = st.number_input("Oxygen Level (%)", 0, 100, 98)
                                respiratory_rate = st.number_input("Respiratory Rate", 0, 80, 16)
                                pain_score = st.slider("Pain Score (0-10)", 0, 10, 0)
                            vitals_notes = st.text_area("Vitals Notes")
                            save_vitals = st.form_submit_button("Save Vitals", use_container_width=True)

                        if save_vitals:
                            payload = {
                                "prediction_id": int(prediction_id),
                                "nurse_id": int(nurse_id),
                                "temperature": float(temperature),
                                "heart_rate": int(heart_rate),
                                "blood_pressure": blood_pressure or None,
                                "oxygen_level": int(oxygen_level),
                                "respiratory_rate": int(respiratory_rate),
                                "pain_score": int(pain_score),
                                "notes": vitals_notes or None,
                            }
                            result = _safe_post(api_url, "/nurse-vitals/add", payload, token)
                            if result:
                                st.success("Vitals recorded.")

                with tab3:
                    st.markdown("**Add Nurse Notes / Care Task**")
                    if not nurse_id:
                        st.info("Nurse ID not found. Use Nurse Management to look up your nurse ID.")
                    else:
                        with st.form(f"task_{prediction_id}_{nurse_id}"):
                            task_title = st.text_input("Task Title", placeholder="e.g. IV line placement")
                            task_description = st.text_area("Notes / Description")
                            task_priority = st.selectbox("Priority", ["Low", "Medium", "High", "Critical"])
                            save_task = st.form_submit_button("Add Task", use_container_width=True)

                        if save_task:
                            if not task_title.strip():
                                st.warning("Task title is required.")
                            else:
                                payload = {
                                    "prediction_id": int(prediction_id),
                                    "nurse_id": int(nurse_id),
                                    "task_title": task_title.strip(),
                                    "task_description": task_description or None,
                                    "status": "Pending",
                                    "priority": task_priority,
                                }
                                result = _safe_post(api_url, "/nurse-tasks/add", payload, token)
                                if result:
                                    st.success("Task added.")

    st.divider()
    st.caption(
        "Triage flow: Assigned → In Triage → Doctor Review → Completed. "
        "Use the buttons above to move patients through the workflow."
    )
