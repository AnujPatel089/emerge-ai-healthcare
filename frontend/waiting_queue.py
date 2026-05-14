"""
Waiting Queue dashboard page for EmergeAI Healthcare.

Shows all active emergency queue patients, assignment status, and provides
controls for auto-assignment and manual nurse assignment.

Role rules:
- admin, doctor, super_admin: full view + assign/reassign
- nurse: sees only their assigned patients
- patient: no access (blocked at navigation level)
"""

from __future__ import annotations

import requests
import pandas as pd
import streamlit as st
from datetime import datetime


ESI_COLORS = {
    1: "#dc2626",   # red - Critical
    2: "#ea580c",   # orange - Emergency
    3: "#ca8a04",   # yellow - Urgent
    4: "#2563eb",   # blue - Semi-Urgent
    5: "#16a34a",   # green - Non-Urgent
}

ESI_LABELS = {
    1: "ESI 1 Critical",
    2: "ESI 2 Emergency",
    3: "ESI 3 Urgent",
    4: "ESI 4 Semi-Urgent",
    5: "ESI 5 Non-Urgent",
}

STATUS_COLORS = {
    "waiting": "#ca8a04",
    "assigned": "#2563eb",
    "in_triage": "#7c3aed",
    "doctor_review": "#ea580c",
    "completed": "#16a34a",
    "critical": "#dc2626",
}


def _headers(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def _safe_get(api_url: str, path: str, token: str, timeout: int = 20) -> dict | None:
    try:
        resp = requests.get(f"{api_url}{path}", headers=_headers(token), timeout=timeout)
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


def _safe_post(api_url: str, path: str, payload: dict, token: str, timeout: int = 20) -> dict | None:
    try:
        resp = requests.post(
            f"{api_url}{path}", json=payload, headers=_headers(token), timeout=timeout
        )
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


def _esi_badge(esi_level) -> str:
    try:
        level = int(str(esi_level).strip().replace("ESI", "").strip())
    except Exception:
        level = 3
    color = ESI_COLORS.get(level, "#6b7280")
    label = ESI_LABELS.get(level, f"ESI {level}")
    return f"<span style='background:{color};color:white;padding:3px 10px;border-radius:12px;font-size:12px;font-weight:700'>{label}</span>"


def _status_badge(status: str) -> str:
    color = STATUS_COLORS.get(str(status).lower(), "#6b7280")
    return f"<span style='background:{color};color:white;padding:3px 10px;border-radius:12px;font-size:12px;font-weight:700'>{status.replace('_', ' ').title()}</span>"


def _format_wait(estimated_wait_time) -> str:
    if estimated_wait_time is None:
        return "—"
    try:
        mins = int(estimated_wait_time)
        if mins == 0:
            return "Immediate"
        return f"{mins} min"
    except Exception:
        return str(estimated_wait_time)


def render_waiting_queue(api_url: str, token: str, role: str, username: str) -> None:
    st.title("Emergency Waiting Queue")
    st.caption("Real-time emergency department queue with AI-prioritized triage assignments.")

    can_assign = role in ["admin", "doctor", "super_admin"]

    # ---- Auto Assign All button ----
    if can_assign:
        col_btn1, col_btn2, _ = st.columns([1, 1, 4])
        with col_btn1:
            if st.button("Auto Assign All Waiting", type="primary", use_container_width=True):
                result = _safe_post(api_url, "/api/assignments/auto-assign-all", {}, token)
                if result:
                    assigned = result.get("assigned_count", 0)
                    skipped = result.get("skipped_count", 0)
                    msg = result.get("message", "")
                    if assigned > 0:
                        st.success(f"Assigned {assigned} patients. {msg}")
                    elif "No active nurses" in msg:
                        st.warning("No active nurses available. Run the seed script or add nurses first.")
                    else:
                        st.info(f"No waiting patients to assign. {msg}")
                    st.rerun()
        with col_btn2:
            if st.button("Refresh Queue", use_container_width=True):
                st.rerun()
    else:
        if st.button("Refresh Queue"):
            st.rerun()

    st.divider()

    # ---- Load queue data ----
    data = _safe_get(api_url, "/api/assignments/waiting-queue", token)
    if data is None:
        return

    queue = data.get("waiting_queue") or data.get("queue") or []

    if not queue:
        st.info("No waiting patients found. Run a Live Prediction to add patients to the queue.")
        return

    # ---- Summary metrics ----
    total = len(queue)
    waiting = sum(1 for p in queue if str(p.get("assignment_status", "")).lower() == "waiting")
    assigned = total - waiting
    critical = sum(1 for p in queue if p.get("critical_status"))

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total in Queue", total)
    m2.metric("Waiting", waiting, delta=f"{waiting} unassigned" if waiting else None,
              delta_color="inverse" if waiting else "normal")
    m3.metric("Assigned", assigned)
    m4.metric("Critical", critical)

    st.divider()

    # ---- Load nurses for assignment dropdown ----
    nurses_data = _safe_get(api_url, "/nurses", token) if can_assign else None
    nurses = (nurses_data or {}).get("nurses", []) if nurses_data else []
    active_nurses = [n for n in nurses if n.get("available_status") is not False]
    nurse_options = {
        f"{n.get('name')} ({n.get('experience_level', 'normal')}) — {n.get('active_patient_count', 0)} pts": n.get("id")
        for n in active_nurses
    }

    if can_assign and not nurse_options:
        st.warning(
            "No active nurses found. Run `python scripts/seed_demo_data.py` to create demo nurses, "
            "or add nurses via Nurse Management."
        )

    # ---- Queue table ----
    for patient in queue:
        prediction_id = patient.get("prediction_id") or patient.get("patient_id")
        patient_name = patient.get("patient_name") or f"Patient #{prediction_id}"
        esi_raw = patient.get("esi_level") or patient.get("esi_severity") or "3"
        assignment_status = str(patient.get("assignment_status") or "waiting").lower()
        assigned_nurse = patient.get("assigned_nurse_name") or "Unassigned"
        wait_time = _format_wait(patient.get("wait_time") or patient.get("estimated_wait_time"))
        icu_risk = "Yes" if patient.get("icu_risk") else "No"
        critical_flag = patient.get("critical_status", False)
        created_at = patient.get("created_at") or patient.get("arrival_time") or "—"

        # card border color by ESI
        try:
            esi_num = int(str(esi_raw).replace("ESI", "").strip())
        except Exception:
            esi_num = 3
        border_color = ESI_COLORS.get(esi_num, "#6b7280")

        with st.container():
            st.markdown(
                f"""
                <div style="border-left:5px solid {border_color};padding:12px 18px;
                    border-radius:10px;background:#f8fafc;margin-bottom:8px;">
                  <span style="font-weight:700;font-size:16px;">{patient_name}</span>
                  &nbsp;&nbsp;
                  {_esi_badge(esi_raw)}
                  &nbsp;&nbsp;
                  {_status_badge(assignment_status)}
                  &nbsp;&nbsp;
                  {"<span style='background:#dc2626;color:white;padding:2px 8px;border-radius:8px;font-size:11px'>CRITICAL</span>" if critical_flag else ""}
                  <br/>
                  <small style="color:#64748b;">
                    ICU Risk: <b>{icu_risk}</b> &nbsp;|&nbsp;
                    Wait: <b>{wait_time}</b> &nbsp;|&nbsp;
                    Nurse: <b>{assigned_nurse}</b> &nbsp;|&nbsp;
                    Arrived: {str(created_at)[:19]}
                  </small>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if can_assign and assignment_status not in ("completed",):
                with st.expander(f"Actions for {patient_name}"):
                    action_col1, action_col2, action_col3 = st.columns(3)

                    with action_col1:
                        st.markdown("**Auto Assign**")
                        if st.button(f"Auto Assign", key=f"auto_{prediction_id}"):
                            result = _safe_post(
                                api_url,
                                "/api/assignments/auto-assign",
                                {"prediction_id": prediction_id},
                                token,
                            )
                            if result:
                                if result.get("status") == "no_nurse_available":
                                    st.warning("No active nurses available.")
                                else:
                                    a = result.get("assignment", {})
                                    nurse_info = a.get("nurse") or {}
                                    st.success(f"Assigned to {nurse_info.get('name', 'nurse')}")
                                    st.rerun()

                    with action_col2:
                        if nurse_options:
                            st.markdown("**Manual Assign / Reassign**")
                            selected_nurse_label = st.selectbox(
                                "Choose nurse",
                                list(nurse_options.keys()),
                                key=f"nurse_select_{prediction_id}",
                                label_visibility="collapsed",
                            )
                            selected_nurse_id = nurse_options[selected_nurse_label]
                            assign_label = "Reassign" if assignment_status == "assigned" else "Assign"
                            if st.button(assign_label, key=f"assign_{prediction_id}"):
                                endpoint = (
                                    "/api/assignments/reassign-nurse"
                                    if assignment_status in ("assigned", "in_triage", "doctor_review")
                                    else "/api/assignments/assign-nurse"
                                )
                                result = _safe_post(
                                    api_url,
                                    endpoint,
                                    {"prediction_id": prediction_id, "nurse_id": selected_nurse_id},
                                    token,
                                )
                                if result:
                                    st.success("Nurse assigned.")
                                    st.rerun()

                    with action_col3:
                        st.markdown("**Update Status**")
                        new_status = st.selectbox(
                            "New status",
                            ["in_triage", "doctor_review", "completed"],
                            key=f"status_select_{prediction_id}",
                            label_visibility="collapsed",
                        )
                        if st.button("Update", key=f"update_status_{prediction_id}"):
                            result = _safe_post(
                                api_url,
                                "/api/assignments/update-status",
                                {"prediction_id": prediction_id, "status": new_status},
                                token,
                            )
                            if result:
                                st.success(f"Status updated to {new_status}.")
                                st.rerun()

            elif role == "nurse" and assignment_status in ("assigned", "in_triage"):
                with st.expander(f"Nurse actions for {patient_name}"):
                    n_col1, n_col2 = st.columns(2)
                    with n_col1:
                        if assignment_status == "assigned":
                            if st.button("Start Triage", key=f"triage_{prediction_id}", type="primary"):
                                result = _safe_post(
                                    api_url,
                                    "/api/assignments/update-status",
                                    {"prediction_id": prediction_id, "status": "in_triage"},
                                    token,
                                )
                                if result:
                                    st.success("Triage started.")
                                    st.rerun()
                    with n_col2:
                        if assignment_status == "in_triage":
                            if st.button("Send to Doctor", key=f"doctor_{prediction_id}"):
                                result = _safe_post(
                                    api_url,
                                    "/api/assignments/update-status",
                                    {"prediction_id": prediction_id, "status": "doctor_review"},
                                    token,
                                )
                                if result:
                                    st.success("Patient sent to doctor review.")
                                    st.rerun()

    st.divider()
    st.caption(
        "Queue sorted by ESI severity (critical first), then wait time. "
        "Nurses see only their assigned patients."
    )
