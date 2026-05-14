"""
Doctor review dashboard for EmergeAI Healthcare.

Shows patients in doctor_review status, AI prediction details, nurse notes,
and provides diagnosis/treatment input and mark-completed controls.

Role access: doctor, admin, super_admin.
"""

from __future__ import annotations

import requests
import streamlit as st


DOCTOR_ACTIONS = ["Review Patient", "Add Diagnosis Notes", "Reassign Nurse", "Mark Completed"]

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


def render_doctor_dashboard(api_url: str, token: str, role: str, username: str) -> None:
    st.title("Doctor Review Dashboard")
    st.caption(
        "Patients waiting for doctor review. Review AI prediction, nurse notes, "
        "add diagnosis and treatment, then mark completed."
    )

    col_refresh, _ = st.columns([1, 5])
    with col_refresh:
        if st.button("Refresh", use_container_width=True):
            st.rerun()

    # Load all assignments and filter for doctor_review status
    data = _safe_get(api_url, "/api/assignments/my-patients", token)
    if data is None:
        return

    all_patients = data.get("patients", [])
    review_patients = [
        p for p in all_patients
        if "doctor" in str(p.get("assignment_status", "")).lower()
        or "review" in str(p.get("status", "")).lower()
    ]

    # Also get waiting queue to find any in doctor_review
    queue_data = _safe_get(api_url, "/api/assignments/waiting-queue", token)
    queue = (queue_data or {}).get("waiting_queue") or (queue_data or {}).get("queue") or []
    queue_review = [
        p for p in queue
        if "doctor" in str(p.get("assignment_status", "")).lower()
    ]

    # Merge without duplicates
    seen_ids = {p.get("prediction_id") for p in review_patients}
    for q in queue_review:
        qid = q.get("prediction_id")
        if qid not in seen_ids:
            review_patients.append(q)
            seen_ids.add(qid)

    if not review_patients:
        st.info(
            "No patients currently in Doctor Review status. "
            "Nurses must advance patients from 'In Triage' to 'Doctor Review' first."
        )

        # Show total queue summary
        if queue:
            st.subheader("Current Queue Overview")
            status_counts: dict[str, int] = {}
            for p in queue:
                s = str(p.get("assignment_status", "waiting")).lower()
                status_counts[s] = status_counts.get(s, 0) + 1
            for status, count in status_counts.items():
                st.metric(status.replace("_", " ").title(), count)
        return

    st.success(f"{len(review_patients)} patient(s) waiting for doctor review.")
    st.divider()

    for patient in review_patients:
        prediction_id = patient.get("prediction_id") or patient.get("patient_id")
        patient_name = patient.get("patient_name") or f"Patient #{prediction_id}"
        esi_level = patient.get("esi_level") or patient.get("esi_severity") or "—"
        icu_risk = "Yes" if patient.get("icu_risk") else "No"
        priority = patient.get("priority") or patient.get("priority_level") or "—"
        assigned_nurse = patient.get("assigned_nurse_name") or (patient.get("nurse") or {}).get("name") or "—"
        critical = patient.get("critical_status", False)

        border_color = _esi_color(str(esi_level))

        with st.container():
            st.markdown(
                f"""
                <div style="border-left:5px solid {border_color};padding:12px 18px;
                    border-radius:10px;background:#fff7ed;margin-bottom:6px;">
                  <span style="font-weight:700;font-size:16px;">{patient_name}</span>
                  &nbsp; <span style="color:#64748b;font-size:13px;">ID #{prediction_id}</span>
                  {"&nbsp;<span style='background:#dc2626;color:white;padding:2px 8px;border-radius:8px;font-size:11px'>CRITICAL</span>" if critical else ""}
                  <br/>
                  <small style="color:#64748b;">
                    ESI: <b>{esi_level}</b> &nbsp;|&nbsp;
                    ICU Risk: <b>{icu_risk}</b> &nbsp;|&nbsp;
                    Priority: <b>{priority}</b> &nbsp;|&nbsp;
                    Nurse: <b>{assigned_nurse}</b>
                  </small>
                </div>
                """,
                unsafe_allow_html=True,
            )

            with st.expander(f"Review {patient_name}"):
                tab1, tab2, tab3 = st.tabs(
                    ["AI Prediction & History", "Add Diagnosis & Treatment", "Reassign / Complete"]
                )

                with tab1:
                    st.markdown("**AI Triage Prediction**")
                    detail_data = _safe_get(api_url, f"/api/assignments/patient/{prediction_id}", token)
                    if detail_data and detail_data.get("patient"):
                        detail = detail_data["patient"]
                        pred_col1, pred_col2 = st.columns(2)
                        with pred_col1:
                            st.metric("ESI Level", detail.get("esi_severity", esi_level))
                            st.metric("Priority", detail.get("priority", priority))
                        with pred_col2:
                            st.metric("Critical Status", "Yes" if detail.get("critical_status") else "No")
                            st.metric("Wait Time", f"{detail.get('estimated_wait_time', '—')} min")

                    # Load prediction history for this patient
                    history_resp = _safe_get(api_url, "/history", token)
                    if history_resp:
                        history = history_resp.get("history", [])
                        this_record = next(
                            (r for r in history if r.get("id") == prediction_id), None
                        )
                        if this_record:
                            st.markdown("**Patient Vitals at Triage**")
                            v_col1, v_col2, v_col3 = st.columns(3)
                            with v_col1:
                                st.metric("Heart Rate", this_record.get("triage_vital_hr", "—"))
                                st.metric("SpO2", f"{this_record.get('triage_vital_o2', '—')}%")
                            with v_col2:
                                st.metric(
                                    "BP",
                                    f"{this_record.get('triage_vital_sbp', '—')}/"
                                    f"{this_record.get('triage_vital_dbp', '—')}",
                                )
                                st.metric("Temp", f"{this_record.get('triage_vital_temp', '—')}°F")
                            with v_col3:
                                st.metric("Resp Rate", this_record.get("triage_vital_rr", "—"))
                                st.metric("ML Prediction", this_record.get("ml_prediction", "—"))

                            if this_record.get("problem_description"):
                                st.info(f"Problem Description: {this_record['problem_description']}")
                            if this_record.get("clinical_explanations"):
                                st.markdown("**Clinical Explanation:**")
                                st.caption(this_record["clinical_explanations"])
                        else:
                            st.info("No detailed history found for this patient ID.")

                    # Check for existing doctor review
                    dr_data = _safe_get(api_url, f"/doctor-review/{prediction_id}", token)
                    if dr_data and dr_data.get("review"):
                        dr = dr_data["review"]
                        st.markdown("**Existing Doctor Review**")
                        st.write(f"**Diagnosis:** {dr.get('diagnosis', '—')}")
                        st.write(f"**Treatment:** {dr.get('treatment_plan', '—')}")
                        st.write(f"**Admit Status:** {dr.get('admit_status', '—')}")

                with tab2:
                    st.markdown(
                        "> This system provides AI-assisted decision support only. "
                        "A licensed clinician must verify all findings and make final clinical decisions."
                    )
                    with st.form(f"doctor_review_form_{prediction_id}"):
                        diagnosis = st.text_area(
                            "Diagnosis",
                            placeholder="e.g. Acute myocardial infarction with cardiogenic shock",
                        )
                        treatment_plan = st.text_area(
                            "Treatment Plan",
                            placeholder="e.g. IV fluids, aspirin, refer to cardiology",
                        )
                        medication_notes = st.text_area(
                            "Medication Notes (optional)",
                            placeholder="e.g. Aspirin 325mg PO, nitroglycerin 0.4mg SL",
                        )
                        admit_options = ["Not Admitted", "Admitted - General", "Admitted - ICU", "Observation", "Discharged"]
                        admit_status = st.selectbox("Admit / Discharge Status", admit_options)
                        follow_up = st.checkbox("Follow-up Required")
                        submit_review = st.form_submit_button(
                            "Save Doctor Review", type="primary", use_container_width=True
                        )

                    if submit_review:
                        if not diagnosis.strip() or not treatment_plan.strip():
                            st.warning("Diagnosis and treatment plan are required.")
                        else:
                            payload = {
                                "prediction_id": int(prediction_id),
                                "diagnosis": diagnosis.strip(),
                                "treatment_plan": treatment_plan.strip(),
                                "medication_notes": medication_notes or None,
                                "follow_up_required": follow_up,
                                "admit_status": admit_status,
                            }
                            result = _safe_post(api_url, "/doctor-review/add", payload, token)
                            if result:
                                st.success("Doctor review saved.")

                with tab3:
                    st.markdown("**Complete or Reassign**")

                    complete_col, reassign_col = st.columns(2)

                    with complete_col:
                        st.markdown("Mark this case as completed:")
                        if st.button(
                            "Mark Completed", key=f"complete_{prediction_id}", type="primary"
                        ):
                            result = _safe_post(
                                api_url,
                                "/api/assignments/update-status",
                                {"prediction_id": prediction_id, "status": "completed"},
                                token,
                            )
                            if result:
                                st.success("Case marked as completed.")
                                st.rerun()

                    with reassign_col:
                        if role in ["admin", "super_admin", "doctor"]:
                            st.markdown("Reassign to different nurse:")
                            nurses_data = _safe_get(api_url, "/nurses", token)
                            nurses = (nurses_data or {}).get("nurses", [])
                            active_nurses = [n for n in nurses if n.get("available_status") is not False]
                            if active_nurses:
                                nurse_map = {
                                    f"{n.get('name')} ({n.get('active_patient_count', 0)} pts)": n.get("id")
                                    for n in active_nurses
                                }
                                selected_nurse = st.selectbox(
                                    "Select Nurse",
                                    list(nurse_map.keys()),
                                    key=f"reassign_nurse_{prediction_id}",
                                )
                                if st.button("Reassign", key=f"do_reassign_{prediction_id}"):
                                    result = _safe_post(
                                        api_url,
                                        "/api/assignments/reassign-nurse",
                                        {
                                            "prediction_id": prediction_id,
                                            "nurse_id": nurse_map[selected_nurse],
                                        },
                                        token,
                                    )
                                    if result:
                                        st.success("Patient reassigned.")
                                        st.rerun()
                            else:
                                st.info("No available nurses for reassignment.")

    st.divider()
    st.caption(
        "DISCLAIMER: This system provides educational AI decision support only. "
        "It does not provide a final diagnosis. All findings must be verified by a licensed clinician."
    )
