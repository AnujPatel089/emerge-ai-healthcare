"""
Admin assignment dashboard for EmergeAI Healthcare.

Shows nurse workload summary, waiting queue overview, and provides
controls for auto-assignment and manual nurse assignment.

Role access: admin, super_admin (full), doctor (view + assign).
"""

from __future__ import annotations

import requests
import streamlit as st


ADMIN_ASSIGNMENT_ACTIONS = ["Assign Nurse", "Reassign Nurse", "Monitor Queue", "Monitor Workload"]

AVAILABILITY_COLORS = {
    "Available": "#16a34a",
    "Busy": "#ca8a04",
    "Critical Load": "#dc2626",
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


def render_nurse_workload(api_url: str, token: str, role: str) -> None:
    st.title("Nurse Workload Monitor")
    st.caption(
        "Real-time nurse workload overview. Green = Available (0-3 pts), "
        "Yellow = Busy (4-6 pts), Red = Critical Load (7+)."
    )

    col_refresh, col_autoassign, _ = st.columns([1, 2, 4])
    with col_refresh:
        if st.button("Refresh", use_container_width=True):
            st.rerun()
    with col_autoassign:
        if role in ["admin", "super_admin", "doctor"]:
            if st.button("Auto Assign All Waiting Patients", type="primary", use_container_width=True):
                result = _safe_post(api_url, "/api/assignments/auto-assign-all", {}, token)
                if result:
                    assigned = result.get("assigned_count", 0)
                    msg = result.get("message", "")
                    if assigned > 0:
                        st.success(f"Assigned {assigned} patients. {msg}")
                    elif "No active nurses" in msg:
                        st.warning(
                            "No active nurses available. "
                            "Run `python scripts/seed_demo_data.py` to create demo nurses."
                        )
                    else:
                        st.info(f"No waiting patients to assign. {msg}")
                    st.rerun()

    st.divider()

    data = _safe_get(api_url, "/api/assignments/nurse-workload", token)
    if data is None:
        return

    nurses = data.get("nurses") or data.get("workload") or []

    if not nurses:
        st.warning(
            "No nurse records found. Run `python scripts/seed_demo_data.py` "
            "to create demo nurses, or add nurses via Nurse Management."
        )
        return

    # Summary metrics
    total = len(nurses)
    available = sum(1 for n in nurses if n.get("availability_status") == "Available")
    busy = sum(1 for n in nurses if n.get("availability_status") == "Busy")
    critical_load = sum(1 for n in nurses if n.get("availability_status") == "Critical Load")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Nurses", total)
    m2.metric("Available", available)
    m3.metric("Busy", busy)
    m4.metric("Critical Load", critical_load, delta_color="inverse" if critical_load else "normal")

    st.divider()

    if not available and total > 0:
        st.warning(
            "No available nurses at this time. All nurses are at capacity. "
            "New patients will remain in waiting status."
        )

    # Nurse workload cards
    cols = st.columns(min(3, total) or 1)
    for i, nurse in enumerate(nurses):
        col = cols[i % len(cols)]
        with col:
            avail_status = nurse.get("availability_status", "Available")
            color = AVAILABILITY_COLORS.get(avail_status, "#6b7280")
            active = nurse.get("current_assigned_patient_count") or nurse.get("active_patient_count") or 0
            critical_pts = nurse.get("critical_patients") or 0
            in_triage_pts = nurse.get("in_triage_patients") or 0
            doctor_review_pts = nurse.get("doctor_review_patients") or 0
            exp = nurse.get("experience_level") or "normal"
            name = nurse.get("name") or "Unknown Nurse"
            email = nurse.get("email") or ""

            st.markdown(
                f"""
                <div style="border:2px solid {color};border-radius:14px;padding:16px;
                    background:#f8fafc;margin-bottom:12px;">
                  <div style="font-weight:700;font-size:16px;">{name}</div>
                  <div style="color:#64748b;font-size:12px;">{email}</div>
                  <div style="margin:8px 0;">
                    <span style="background:{color};color:white;padding:2px 10px;
                        border-radius:8px;font-size:12px;font-weight:700;">
                      {avail_status}
                    </span>
                    &nbsp;
                    <span style="background:#e0f2fe;color:#0369a1;padding:2px 8px;
                        border-radius:8px;font-size:11px;">
                      {exp.title()}
                    </span>
                  </div>
                  <div style="font-size:13px;color:#334155;line-height:1.8;">
                    <b>Active patients:</b> {active}<br/>
                    <b>In triage:</b> {in_triage_pts}<br/>
                    <b>Doctor review:</b> {doctor_review_pts}<br/>
                    <b>Critical/High:</b> {critical_pts}
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.divider()

    # Manual assignment panel
    if role in ["admin", "super_admin", "doctor"]:
        st.subheader("Manual Nurse Assignment")

        queue_data = _safe_get(api_url, "/api/assignments/waiting-queue", token)
        queue = (queue_data or {}).get("waiting_queue") or (queue_data or {}).get("queue") or []
        waiting_patients = [p for p in queue if str(p.get("assignment_status", "")).lower() == "waiting"]

        nurses_for_assign = [
            n for n in nurses if n.get("availability_status") in ("Available", "Busy")
        ]
        nurse_map = {
            f"{n.get('name')} ({n.get('availability_status')}, {n.get('current_assigned_patient_count', 0)} pts)": n.get("id")
            for n in nurses_for_assign
        }

        if not waiting_patients:
            st.info("No patients waiting for assignment.")
        elif not nurse_map:
            st.warning(
                "No available nurses to assign. "
                "Run `python scripts/seed_demo_data.py` to create demo nurses."
            )
        else:
            with st.form("manual_assign_form"):
                patient_options = {
                    f"Patient #{p.get('prediction_id')} | {p.get('esi_severity') or 'ESI ?'} | {p.get('priority', '')}": p.get("prediction_id")
                    for p in waiting_patients
                }
                selected_patient_label = st.selectbox(
                    "Select Waiting Patient", list(patient_options.keys())
                )
                selected_nurse_label = st.selectbox(
                    "Select Nurse", list(nurse_map.keys())
                )
                assign_notes = st.text_input("Notes (optional)")
                submit = st.form_submit_button("Assign Nurse", type="primary", use_container_width=True)

            if submit:
                prediction_id = patient_options[selected_patient_label]
                nurse_id = nurse_map[selected_nurse_label]
                result = _safe_post(
                    api_url,
                    "/api/assignments/assign-nurse",
                    {
                        "prediction_id": prediction_id,
                        "nurse_id": nurse_id,
                        "notes": assign_notes or "Manual assignment from admin dashboard.",
                    },
                    token,
                )
                if result:
                    a = result.get("assignment", {})
                    nurse_info = a.get("nurse") or {}
                    st.success(f"Patient #{prediction_id} assigned to {nurse_info.get('name', 'nurse')}.")
                    st.rerun()


def render_admin_dashboard(api_url: str, token: str, role: str, username: str) -> None:
    """Main entry point for the admin assignment dashboard."""
    tab1, tab2 = st.tabs(["Nurse Workload", "Waiting Queue Overview"])

    with tab1:
        render_nurse_workload(api_url, token, role)

    with tab2:
        from frontend.waiting_queue import render_waiting_queue
        render_waiting_queue(api_url, token, role, username)
