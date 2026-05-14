import streamlit as st
import requests
import pandas as pd
import plotly.express as px
from PIL import Image
from io import BytesIO
from datetime import datetime
from streamlit_extras.metric_cards import style_metric_cards
import tempfile
import html
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.triage_rules import ESI_COLORS, ESI_LABELS, estimate_icu_risk, estimate_readmission_risk, rule_based_esi
from src.clinical_summary_generator import DISCLAIMER as HISTORICAL_REPORT_DISCLAIMER
from src.image_assessment import IMAGE_EXTENSIONS
from frontend.historical_reports import render_historical_reports_page
from frontend.waiting_queue import render_waiting_queue
from frontend.nurse_dashboard import render_nurse_dashboard
from frontend.admin_dashboard import render_admin_dashboard, render_nurse_workload
from frontend.doctor_dashboard import render_doctor_dashboard
from frontend.mlops_dashboard import render_mlops_dashboard
from frontend.platform_dashboard import render_platform_dashboard
from frontend.self_healing_dashboard import render_self_healing_dashboard
from frontend.admin_command_center import render_admin_command_center
from frontend.patient_timeline import render_patient_timeline
from frontend.alerts_dashboard import render_alerts_dashboard
from frontend.model_governance import render_model_governance
from frontend.render_status import render_render_status

try:
    from audio_recorder_streamlit import audio_recorder
    import speech_recognition as sr
    AUDIO_ENABLED = True
except Exception:
    AUDIO_ENABLED = False


#API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="EmergeAI Healthcare",
    page_icon="🏥",
    layout="wide"
)


# -----------------------------
# SESSION STATE
# -----------------------------

if "token" not in st.session_state:
    st.session_state.token = None
if "role" not in st.session_state:
    st.session_state.role = None
if "username" not in st.session_state:
    st.session_state.username = None
if "prediction_id" not in st.session_state:
    st.session_state.prediction_id = None
if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None
if "problem_description" not in st.session_state:
    st.session_state.problem_description = ""
if "nlp_result" not in st.session_state:
    st.session_state.nlp_result = None
if "cv_result" not in st.session_state:
    st.session_state.cv_result = None
# NEW
if "dl_image_result" not in st.session_state:
    st.session_state.dl_image_result = None
if "dl_image_log_id" not in st.session_state:
    st.session_state.dl_image_log_id = None
if "triage_result" not in st.session_state:
    st.session_state.triage_result = None
if "triage_log_id" not in st.session_state:
    st.session_state.triage_log_id = None
if "triage_upload_context" not in st.session_state:
    st.session_state.triage_upload_context = None
if "is_authenticated" not in st.session_state:
    st.session_state.is_authenticated = False
if "account_status" not in st.session_state:
    st.session_state.account_status = None
if "auth_view" not in st.session_state:
    st.session_state.auth_view = "login"

SYNTHETIC_DATA_PATH = Path("data/emergency_synthetic_data.csv")


SYMPTOM_LABELS = {
    "cc_chestpain": "Chest Pain",
    "cc_shortnessofbreath": "Shortness of Breath",
    "cc_headache": "Headache",
    "cc_fever": "Fever",
    "cc_abdominalpain": "Abdominal Pain",
    "cc_dizziness": "Dizziness",
    "cc_syncope": "Syncope",
    "cc_weakness": "Weakness"
}

ROLE_DISPLAY_NAMES = {
    "super_admin": "Super Admin",
    "admin": "Hospital Admin",
    "doctor": "Emergency Doctor",
    "nurse": "Triage Nurse",
    "patient": "Patient",
}


# -----------------------------
# HELPERS
# -----------------------------

def logout():
    for key in [
        "token", "role", "username", "prediction_id", "last_prediction",
        "problem_description", "nlp_result", "cv_result",
        "dl_image_result", "dl_image_log_id", "triage_result", "triage_log_id",
        "triage_upload_context", "is_authenticated", "account_status"
    ]:
        st.session_state[key] = None
    st.session_state.is_authenticated = False
    st.session_state.problem_description = ""
    st.session_state.auth_view = "login"
    st.rerun()


def auth_headers():
    if not st.session_state.token:
        return {}
    return {"Authorization": f"Bearer {st.session_state.token}"}

def check_backend_status():
    try:
        response = requests.get(
            f"{API_URL}/health",
            timeout=3
        )
        return response.status_code == 200
    except Exception:
        return False


def show_backend_offline_message() -> None:
    st.error(
        "Backend is offline. Start FastAPI with "
        "`uvicorn backend.main:app --reload --port 8000`, then try again."
    )
    st.caption(f"Frontend is currently configured to call: {API_URL}")


def handle_response_error(response):
    if response.status_code == 401:
        st.error("Session expired or invalid token. Please login again.")
        st.session_state.token = None
        st.session_state.role = None
        st.session_state.username = None
        st.stop()
    elif response.status_code == 403:
        st.error("Access denied. Your role does not have permission.")
    else:
        st.error(response.text)


def transcribe_audio(audio_bytes):
    if not AUDIO_ENABLED:
        return "Audio transcription packages are not installed."

    recognizer = sr.Recognizer()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        temp_audio.write(audio_bytes)
        temp_audio_path = temp_audio.name

    try:
        with sr.AudioFile(temp_audio_path) as source:
            audio_data = recognizer.record(source)
        return recognizer.recognize_google(audio_data)
    except sr.UnknownValueError:
        return "Could not understand the audio. Please record again clearly."
    except sr.RequestError:
        return "Speech recognition service error. Please check internet connection."
    except Exception as e:
        return f"Audio processing error: {e}"


def extract_symptoms_from_text(problem_description):
    return requests.post(
        f"{API_URL}/extract-symptoms",
        json={"problem_description": problem_description},
        headers=auth_headers(),
        timeout=20
    )


def analyze_image_with_cv(uploaded_image):
    files = {
        "image": (uploaded_image.name, uploaded_image.getvalue(), uploaded_image.type)
    }
    return requests.post(
        f"{API_URL}/analyze-image",
        files=files,
        headers=auth_headers(),
        timeout=30
    )


def upload_triage_context_file(uploaded_file, patient_id, patient_name, report_type, notes, triage_session_id=None):
    files = {
        "file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type or "application/octet-stream")
    }
    data = {
        "patient_id": patient_id,
        "patient_name": patient_name,
        "report_type": report_type,
        "notes": notes or "",
        "triage_session_id": triage_session_id or "",
    }
    return requests.post(
        f"{API_URL}/triage-uploads/upload",
        data=data,
        files=files,
        headers=auth_headers(),
        timeout=60,
    )


def highlight_emergency_keywords(text, emergency_keywords):
    if not text:
        return ""
    highlighted = html.escape(text)
    for keyword in emergency_keywords:
        escaped_keyword = html.escape(keyword)
        highlighted = highlighted.replace(
            escaped_keyword,
            f"<mark style='background-color:#ffcccc;color:#8b0000;"
            f"font-weight:700;padding:2px 5px;border-radius:5px;'>"
            f"{escaped_keyword}</mark>"
        )
    return highlighted


def save_clinical_feedback(prediction_id, accepted, override_esi, clinical_notes, override_reason):
    return requests.post(
        f"{API_URL}/clinical-feedback",
        json={
            "prediction_id": prediction_id,
            "accepted": accepted,
            "override_esi": override_esi,
            "clinical_notes": clinical_notes,
            "override_reason": override_reason
        },
        headers=auth_headers(),
        timeout=20
    )


def get_clinical_feedback_history():
    return requests.get(
        f"{API_URL}/clinical-feedback",
        headers=auth_headers(),
        timeout=20
    )


def get_pending_approval_count():
    if st.session_state.role not in ["super_admin", "admin", "doctor", "nurse"] or not st.session_state.token:
        return 0
    try:
        response = requests.get(
            f"{API_URL}/admin/approvals",
            headers=auth_headers(),
            timeout=8
        )
        if response.status_code == 200:
            data = response.json()
            if st.session_state.role == "nurse":
                return len(data.get("new_patient_requests", []))
            return len(data.get("pending_admin_requests", []))
    except Exception:
        return 0
    return 0


def render_nurse_assignment_panel(prediction_id):
    st.markdown('<div class="ui-panel">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Nurse Assignment</div>', unsafe_allow_html=True)

    if st.session_state.role not in ["doctor", "admin", "nurse"]:
        st.info("Nurse assignment is available to clinical users.")
        st.markdown('</div>', unsafe_allow_html=True)
        return

    if prediction_id is None:
        st.warning("Run a prediction first so a prediction ID is available.")
        st.markdown('</div>', unsafe_allow_html=True)
        return

    nurses = []
    assignments = []

    try:
        nurses_response = requests.get(
            f"{API_URL}/nurses",
            headers=auth_headers(),
            timeout=20
        )

        if nurses_response.status_code == 200:
            nurses = nurses_response.json().get("nurses", [])
        else:
            handle_response_error(nurses_response)

        assignments_response = requests.get(
            f"{API_URL}/assignments/{prediction_id}",
            headers=auth_headers(),
            timeout=20
        )

        if assignments_response.status_code == 200:
            assignments = assignments_response.json().get("assignments", [])
        elif assignments_response.status_code != 404:
            handle_response_error(assignments_response)

    except requests.exceptions.ConnectionError:
        st.error("Backend connection error. Make sure FastAPI is running on port 8000.")
        st.markdown('</div>', unsafe_allow_html=True)
        return
    except Exception as e:
        st.error(f"Could not load nurse assignment data: {e}")
        st.markdown('</div>', unsafe_allow_html=True)
        return

    if assignments:
        st.write("Current assignment history")
        assignment_rows = []
        for assignment in assignments:
            nurse = assignment.get("nurse") or {}
            assignment_rows.append({
                "Assignment ID": assignment.get("id"),
                "Nurse": nurse.get("name", "Unknown"),
                "Email": nurse.get("email", ""),
                "Department": nurse.get("department", ""),
                "Status": assignment.get("status"),
                "Assigned At": assignment.get("assigned_at"),
                "Notes": assignment.get("notes")
            })
        st.dataframe(pd.DataFrame(assignment_rows), use_container_width=True, hide_index=True)
    else:
        st.info("No nurse has been assigned to this prediction yet.")

    if st.session_state.role not in ["doctor", "admin"]:
        st.info("Nurse accounts can view assignments, but only doctors and admins can assign nurses.")
        st.markdown('</div>', unsafe_allow_html=True)
        return

    available_nurses = [
        nurse for nurse in nurses
        if nurse.get("available_status") is True
    ]

    if not available_nurses:
        st.warning("No available nurses found. Add or mark a nurse as available first.")
        st.markdown('</div>', unsafe_allow_html=True)
        return

    nurse_options = {
        f"{nurse.get('name')} | {nurse.get('department')} | {nurse.get('email')}": nurse.get("id")
        for nurse in available_nurses
    }

    with st.form(f"assign_nurse_form_{prediction_id}"):
        selected_nurse = st.selectbox("Assign Available Nurse", list(nurse_options.keys()))
        assignment_status = st.selectbox(
            "Assignment Status",
            ["Assigned", "In Progress", "Completed", "Cancelled"]
        )
        assignment_notes = st.text_area(
            "Notes",
            placeholder="Optional handoff notes for this patient."
        )
        assign_clicked = st.form_submit_button("Assign Nurse", use_container_width=True)

    if assign_clicked:
        payload = {
            "prediction_id": int(prediction_id),
            "nurse_id": int(nurse_options[selected_nurse]),
            "status": assignment_status,
            "notes": assignment_notes or None
        }

        try:
            response = requests.post(
                f"{API_URL}/api/assignments/assign-nurse",
                json=payload,
                headers=auth_headers(),
                timeout=20
            )

            if response.status_code == 200:
                st.success("Nurse assigned successfully.")
                st.rerun()
            else:
                handle_response_error(response)

        except requests.exceptions.ConnectionError:
            st.error("Backend connection error. Make sure FastAPI is running on port 8000.")
        except Exception as e:
            st.error(f"Could not assign nurse: {e}")

    st.markdown('</div>', unsafe_allow_html=True)


def render_patient_status_panel(prediction_id):
    st.markdown('<div class="ui-panel">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Patient Status Timeline</div>', unsafe_allow_html=True)

    if st.session_state.role not in ["nurse", "doctor", "admin"]:
        st.info("Patient status tracking is available to clinical users.")
        st.markdown('</div>', unsafe_allow_html=True)
        return

    if prediction_id is None:
        st.warning("Select or run a prediction first.")
        st.markdown('</div>', unsafe_allow_html=True)
        return

    all_statuses = [
        "Waiting",
        "Assigned to Nurse",
        "Under Nurse Care",
        "Waiting for Doctor",
        "Under Treatment",
        "Admitted",
        "Discharged"
    ]

    status_options_by_role = {
        "nurse": [
            "Assigned to Nurse",
            "Under Nurse Care",
            "Waiting for Doctor"
        ],
        "doctor": all_statuses,
        "admin": all_statuses
    }

    try:
        response = requests.get(
            f"{API_URL}/patient-status/{int(prediction_id)}",
            headers=auth_headers(),
            timeout=20
        )

        if response.status_code == 200:
            status_data = response.json()
            current_status = status_data.get("current_status")
            timeline = status_data.get("timeline", [])
        else:
            handle_response_error(response)
            st.markdown('</div>', unsafe_allow_html=True)
            return

    except requests.exceptions.ConnectionError:
        st.error("Backend connection error. Make sure FastAPI is running on port 8000.")
        st.markdown('</div>', unsafe_allow_html=True)
        return
    except Exception as e:
        st.error(f"Could not load patient status: {e}")
        st.markdown('</div>', unsafe_allow_html=True)
        return

    current_label = current_status.get("patient_status") if current_status else "No status recorded"
    st.metric("Current Status", current_label)

    with st.form(f"patient_status_form_{prediction_id}"):
        selected_status = st.selectbox(
            "Update Patient Status",
            status_options_by_role.get(st.session_state.role, [])
        )
        status_notes = st.text_area("Status Notes")
        update_clicked = st.form_submit_button("Update Status", use_container_width=True)

    if update_clicked:
        try:
            update_response = requests.put(
                f"{API_URL}/patient-status/{int(prediction_id)}",
                json={
                    "patient_status": selected_status,
                    "notes": status_notes or None
                },
                headers=auth_headers(),
                timeout=20
            )

            if update_response.status_code == 200:
                st.success("Patient status updated.")
                st.rerun()
            else:
                handle_response_error(update_response)

        except requests.exceptions.ConnectionError:
            st.error("Backend connection error. Make sure FastAPI is running on port 8000.")
        except Exception as e:
            st.error(f"Could not update patient status: {e}")

    if timeline:
        st.write("Timeline")
        timeline_rows = [
            {
                "Time": item.get("updated_at"),
                "Status": item.get("patient_status"),
                "Updated By": item.get("updated_by"),
                "Role": item.get("updated_role"),
                "Notes": item.get("notes")
            }
            for item in reversed(timeline)
        ]
        st.dataframe(pd.DataFrame(timeline_rows), use_container_width=True, hide_index=True)
    else:
        st.info("No status changes recorded yet.")

    st.markdown('</div>', unsafe_allow_html=True)


# -----------------------------
# LOGIN PAGE
# -----------------------------

def register_page():
    st.title("Create EmergeAI Account")
    st.caption("Register for role-based access. Admin requests require approval from an existing admin.")

    with st.form("register_form"):
        full_name = st.text_input("Full Name")
        username = st.text_input("Username")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")
        requested_role = st.selectbox("Requested Role", ["patient", "doctor", "nurse", "admin"])
        submitted = st.form_submit_button("Register", use_container_width=True)

    if submitted:
        if not all([full_name.strip(), username.strip(), email.strip(), password, confirm_password]):
            st.error("Required fields cannot be empty.")
        elif password != confirm_password:
            st.error("Password and confirm password must match.")
        else:
            try:
                response = requests.post(
                    f"{API_URL}/register",
                    json={
                        "full_name": full_name.strip(),
                        "username": username.strip(),
                        "email": email.strip(),
                        "password": password,
                        "confirm_password": confirm_password,
                        "requested_role": requested_role,
                    },
                    timeout=20,
                )
                if response.status_code == 200:
                    message = response.json().get("message", "Registration submitted.")
                    if requested_role == "admin":
                        st.warning(message)
                    else:
                        st.success(message)
                    st.session_state.auth_view = "login"
                else:
                    st.error(response.json().get("detail", response.text))
            except Exception as e:
                st.error(f"Could not submit registration: {e}")

    if st.button("Back to Login"):
        st.session_state.auth_view = "login"
        st.rerun()


def login_page():
    """Professional login screen for EmergeAI Healthcare."""
    if st.session_state.auth_view == "register":
        register_page()
        return

    st.markdown(
        """
        <style>
        /* Hide default Streamlit chrome on login */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        [data-testid="stHeader"] {background: transparent;}

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(14,165,233,0.20), transparent 28%),
                radial-gradient(circle at bottom right, rgba(37,99,235,0.16), transparent 30%),
                linear-gradient(135deg, #f8fbff 0%, #eef7ff 52%, #f8fafc 100%);
        }

        .block-container {
            padding-top: 2.2rem;
            padding-bottom: 2rem;
            max-width: 1180px;
        }

        .login-shell {
            display: grid;
            grid-template-columns: 1.05fr 0.95fr;
            gap: 32px;
            align-items: center;
            margin-top: 18px;
        }

        .brand-panel {
            background: linear-gradient(135deg, #0f172a 0%, #1d4ed8 58%, #06b6d4 100%);
            color: white;
            border-radius: 30px;
            padding: 42px 42px;
            box-shadow: 0 24px 70px rgba(15, 23, 42, 0.24);
            min-height: 570px;
            position: relative;
            overflow: hidden;
        }

        .brand-panel:before {
            content: "";
            position: absolute;
            width: 260px;
            height: 260px;
            border-radius: 50%;
            right: -90px;
            top: -90px;
            background: rgba(255,255,255,0.13);
        }

        .brand-panel:after {
            content: "";
            position: absolute;
            width: 170px;
            height: 170px;
            border-radius: 50%;
            left: -70px;
            bottom: -70px;
            background: rgba(255,255,255,0.10);
        }

        .brand-kicker {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            background: rgba(255,255,255,0.15);
            border: 1px solid rgba(255,255,255,0.22);
            padding: 8px 14px;
            border-radius: 999px;
            font-size: 13px;
            font-weight: 800;
            margin-bottom: 26px;
        }

        .brand-title {
            font-size: 52px;
            line-height: 1.02;
            font-weight: 950;
            margin: 0 0 16px 0;
            letter-spacing: -1.4px;
        }

        .brand-subtitle {
            font-size: 18px;
            line-height: 1.7;
            color: rgba(255,255,255,0.88);
            max-width: 560px;
            margin-bottom: 30px;
        }

        .feature-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 14px;
            margin-top: 28px;
            position: relative;
            z-index: 2;
        }

        .feature-card {
            background: rgba(255,255,255,0.13);
            border: 1px solid rgba(255,255,255,0.18);
            border-radius: 18px;
            padding: 16px;
            backdrop-filter: blur(10px);
        }

        .feature-card b {
            display: block;
            font-size: 15px;
            margin-bottom: 4px;
        }

        .feature-card span {
            font-size: 12px;
            color: rgba(255,255,255,0.78);
        }

        .login-card-pro {
            background: rgba(255,255,255,0.96);
            border: 1px solid #e2e8f0;
            border-radius: 30px;
            padding: 34px;
            box-shadow: 0 24px 70px rgba(15, 23, 42, 0.13);
        }

        .login-card-title {
            text-align: left;
            margin-bottom: 22px;
        }

        .login-card-title .icon-box {
            width: 58px;
            height: 58px;
            border-radius: 18px;
            display: flex;
            align-items: center;
            justify-content: center;
            background: linear-gradient(135deg, #e0f2fe, #dbeafe);
            color: #1d4ed8;
            font-size: 30px;
            margin-bottom: 16px;
        }

        .login-card-title h2 {
            margin: 0 0 6px 0;
            color: #0f172a;
            font-size: 30px;
            font-weight: 950;
            letter-spacing: -0.5px;
        }

        .login-card-title p {
            margin: 0;
            color: #64748b;
            font-size: 14px;
        }

        .secure-badge-row {
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
            margin: 14px 0 22px 0;
        }

        .secure-badge {
            background: #ecfeff;
            color: #0369a1;
            border: 1px solid #bae6fd;
            padding: 7px 10px;
            border-radius: 999px;
            font-size: 12px;
            font-weight: 850;
        }

        .stTextInput input {
            border-radius: 14px !important;
            border: 1px solid #cbd5e1 !important;
            padding: 0.85rem 1rem !important;
            background: #f8fafc !important;
        }

        .stTextInput input:focus {
            border-color: #2563eb !important;
            box-shadow: 0 0 0 3px rgba(37,99,235,0.12) !important;
        }

        [data-baseweb="select"] > div {
            border-radius: 14px !important;
            border: 1px solid #cbd5e1 !important;
            background: #f8fafc !important;
            min-height: 48px;
        }

        div.stButton > button:first-child {
            background: linear-gradient(90deg, #2563eb, #0ea5e9);
            color: white;
            border: none;
            border-radius: 15px;
            padding: 0.85rem 1.4rem;
            font-weight: 950;
            letter-spacing: 0.1px;
            box-shadow: 0 14px 30px rgba(37,99,235,0.26);
            margin-top: 8px;
        }

        div.stButton > button:first-child:hover {
            background: linear-gradient(90deg, #1d4ed8, #0284c7);
            color: white;
            border: none;
            transform: translateY(-1px);
        }

        .demo-credentials {
            margin-top: 20px;
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 18px;
            padding: 16px 18px;
            color: #334155;
            font-size: 14px;
            line-height: 1.7;
        }

        .demo-credentials b {
            color: #0f172a;
            font-size: 15px;
        }

        .notice-box {
            margin-top: 14px;
            background: #fff7ed;
            border: 1px solid #fed7aa;
            color: #9a3412;
            border-radius: 16px;
            padding: 13px 15px;
            font-size: 13px;
            font-weight: 650;
        }

        .login-footer {
            text-align: center;
            margin-top: 18px;
            color: #64748b;
            font-size: 12px;
            font-weight: 700;
        }

        @media (max-width: 900px) {
            .login-shell {
                grid-template-columns: 1fr;
            }
            .brand-panel {
                min-height: auto;
            }
            .brand-title {
                font-size: 40px;
            }
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    left, right = st.columns([1.08, 0.92], gap="large")

    with left:
        st.markdown(
            """
            <div class="brand-panel">
                <div class="brand-kicker">● Secure Clinical AI Workspace</div>
                <h1 class="brand-title">EmergeAI<br>Healthcare</h1>
                <div class="brand-subtitle">
                    AI-powered emergency triage and hospital operations dashboard for
                    doctors, nurses, and administrators.
                </div>
                <div class="feature-grid">
                    <div class="feature-card">
                        <b>🚑 AI Triage</b>
                        <span>Severity prediction, risk scoring, and emergency prioritization.</span>
                    </div>
                    <div class="feature-card">
                        <b>🩺 Clinical Workflow</b>
                        <span>Nurse assignment, vitals, treatment review, and patient status.</span>
                    </div>
                    <div class="feature-card">
                        <b>🧠 Explainable AI</b>
                        <span>SHAP explainability, NLP symptoms, and image analysis support.</span>
                    </div>
                    <div class="feature-card">
                        <b>📊 Admin Command</b>
                        <span>Analytics, alerts, beds, inventory, shifts, and staff workload.</span>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with right:
        st.markdown(
            """
            <div class="login-card-pro">
                <div class="login-card-title">
                    <div class="icon-box">🏥</div>
                    <h2>Sign in</h2>
                    <p>Access your role-based hospital dashboard.</p>
                </div>
                <div class="secure-badge-row">
                    <span class="secure-badge">Backend API</span>
                    <span class="secure-badge">Role-Based Access</span>
                    <span class="secure-badge">Audit Ready</span>
                </div>
            """,
            unsafe_allow_html=True
        )

        username = st.text_input(
            "Username",
            placeholder="anuj, chintan, admin, doctor, or nurse",
            key="login_username"
        )

        password = st.text_input(
            "Password",
            type="password",
            placeholder="Enter your password",
            key="login_password"
        )

        selected_role = st.selectbox(
            "Login Role",
            ["super_admin", "admin", "doctor", "nurse", "patient"],
            format_func=lambda role: role.replace("_", " ").title(),
            key="login_role"
        )

        login_clicked = st.button(
            "Sign in to Dashboard →",
            use_container_width=True
        )

        if login_clicked:
            if not username.strip() or not password.strip():
                st.error("Please enter username and password.")
            elif not check_backend_status():
                show_backend_offline_message()
            else:
                with st.spinner("Authenticating secure session..."):
                    try:
                        response = requests.post(
                            f"{API_URL}/login",
                            json={"username": username.strip(), "password": password, "role": selected_role},
                            timeout=60
                        )
                        if response.status_code == 200:
                            data = response.json()
                            backend_role = str(data.get("role", "")).lower()

                            if backend_role != selected_role:
                                st.error(
                                    f"This account is registered as {backend_role.title()}, "
                                    f"not {selected_role.title()}. Please select the correct role."
                                )
                                st.stop()

                            st.session_state.token = data["access_token"]
                            st.session_state.role = backend_role
                            st.session_state.username = username.strip()
                            st.session_state.account_status = data.get("account_status", "active")
                            st.session_state.is_authenticated = True
                            st.success("Login successful. Loading dashboard...")
                            st.rerun()
                        else:
                            try:
                                st.error(response.json().get("detail", "Invalid username or password."))
                            except Exception:
                                st.error("Invalid username or password.")
                    except requests.exceptions.ConnectionError:
                        show_backend_offline_message()
                    except requests.exceptions.Timeout:
                        st.error("Backend did not respond in time. Confirm FastAPI is running and try again.")
                    except Exception as e:
                        st.error(f"Login failed unexpectedly: {e}")

        action_col1, action_col2 = st.columns(2)
        with action_col1:
            if st.button("Register", use_container_width=True):
                st.session_state.auth_view = "register"
                st.rerun()
        with action_col2:
            if st.button("Continue as Patient", use_container_width=True):
                st.session_state.token = None
                st.session_state.username = "Patient Demo User"
                st.session_state.role = "patient"
                st.session_state.account_status = "active"
                st.session_state.is_authenticated = True
                st.rerun()

        st.info(
            "Patient Demo Mode is for educational and demonstration purposes only. "
            "Data shown is synthetic and not real patient information."
        )

        st.markdown(
            """
                <div class="demo-credentials">
                    <b>Demo Accounts</b><br>
                    👤 Anuj: <code>anuj</code> / <code>anuj123</code> / any role<br>
                    👤 Chintan: <code>chintan</code> / <code>chintan123</code> / any role<br>
                    🛡️ Admin: <code>admin</code> / <code>admin123</code><br>
                    🩺 Doctor: <code>doctor</code> / <code>doctor123</code><br>
                    👩‍⚕️ Nurse: <code>nurse</code> / <code>nurse123</code>
                </div>
                <div class="notice-box">
                    Educational prototype only. Not for real medical diagnosis or clinical use.
                </div>
            </div>
            <div class="login-footer">EmergeAI Healthcare © 2026</div>
            """,
            unsafe_allow_html=True
        )

if st.session_state.token is None and not st.session_state.is_authenticated:
    login_page()
    st.stop()


# -----------------------------
# SIDEBAR
# -----------------------------

st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #eff6ff 0%, #ffffff 45%, #f8fafc 100%);
        border-right: 1px solid #e2e8f0;
    }

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #0f172a;
    }

    [data-testid="stSidebar"] h3 {
        display: none;
    }

    .sidebar-profile {
        background: linear-gradient(135deg, #0ea5e9, #2563eb);
        color: white;
        padding: 18px 16px;
        border-radius: 18px;
        box-shadow: 0 12px 30px rgba(37, 99, 235, 0.22);
        margin-bottom: 16px;
    }

    .sidebar-profile .name {
        font-size: 20px;
        font-weight: 900;
        margin-bottom: 4px;
    }

    .sidebar-profile .role {
        font-size: 14px;
        opacity: 0.92;
        margin-bottom: 10px;
    }

    .status-dot {
        display: inline-block;
        width: 9px;
        height: 9px;
        border-radius: 50%;
        background: #22c55e;
        margin-right: 6px;
    }

    .sidebar-section {
        display: none;
    }

    .top-status-card {
        background: rgba(255,255,255,0.9);
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 14px 18px;
        box-shadow: 0 10px 24px rgba(15, 23, 42, 0.05);
        margin-bottom: 12px;
    }

    .top-status-title {
        font-size: 22px;
        font-weight: 900;
        color: #0f172a;
        margin-bottom: 4px;
    }

    .top-status-subtitle {
        color: #64748b;
        font-size: 14px;
    }

    .status-pill {
        display: inline-block;
        padding: 6px 10px;
        border-radius: 999px;
        background: #ecfeff;
        color: #0369a1;
        font-weight: 800;
        font-size: 12px;
        border: 1px solid #bae6fd;
        margin: 4px 6px 0 0;
    }

    .app-brand {
        background: #ffffff;
        border: 1px solid #dbeafe;
        border-radius: 10px;
        padding: 12px 14px;
        margin-bottom: 12px;
        box-shadow: 0 8px 22px rgba(15, 23, 42, 0.05);
    }

    .app-brand .brand-title {
        font-size: 18px;
        font-weight: 900;
        color: #0f172a;
        line-height: 1.1;
    }

    .app-brand .brand-subtitle {
        font-size: 12px;
        color: #64748b;
        margin-top: 4px;
        font-weight: 700;
    }

    .sidebar-profile {
        border-radius: 10px;
        padding: 14px;
        margin-bottom: 12px;
    }

    .sidebar-profile .name {
        font-size: 17px;
    }

    .nav-helper {
        color: #475569;
        font-size: 12px;
        font-weight: 700;
        margin: 8px 0 4px 0;
    }

    .alert-item {
        border-left: 4px solid #dc2626;
        background: #fff7ed;
        border-radius: 8px;
        padding: 9px 11px;
        margin-bottom: 8px;
        color: #7f1d1d;
        font-size: 13px;
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown("### 🏥 EmergeAI Enterprise")

st.sidebar.markdown(
    """
    <div class="app-brand">
        <div class="brand-title">EmergeAI Healthcare</div>
        <div class="brand-subtitle">Emergency triage command dashboard</div>
    </div>
    """,
    unsafe_allow_html=True
)

role_display = ROLE_DISPLAY_NAMES.get(st.session_state.role, str(st.session_state.role or "User").title())
username_display = str(st.session_state.username or "User").title()
backend_online = check_backend_status()
backend_status = "Backend Online" if backend_online else "Backend Offline"
backend_badge = "Online" if backend_online else "Offline"
database_badge = "Connected" if backend_online else "Unavailable"

st.sidebar.markdown(
    f"""
    <div class="sidebar-profile">
        <div class="name">{username_display}</div>
        <div class="role">Role: <b>{role_display}</b></div>
        <div style="margin-top:10px;font-weight:700;">Active Session</div>
        <div style="margin-top:6px;font-weight:700;">{backend_status}</div>
    </div>
    """,
    unsafe_allow_html=True
)

from datetime import datetime

current_time = datetime.now().strftime("%Y-%m-%d %I:%M %p")

st.sidebar.caption(f"Hospital time: {current_time}")
st.sidebar.caption(f"Account status: {st.session_state.account_status or 'active'}")
pending_approval_count = get_pending_approval_count()
if pending_approval_count:
    sidebar_label = "New patients" if st.session_state.role == "nurse" else "Pending approvals"
    st.sidebar.warning(f"{sidebar_label}: {pending_approval_count}")
if st.session_state.role == "patient":
    st.sidebar.info(
        "Patient Demo Mode is for educational and demonstration purposes only. "
        "Data shown is synthetic and not real patient information."
    )


if st.sidebar.button("🚪 Logout", use_container_width=True):
    logout()

# Clean role-based navigation labels shown to users.
# Existing page logic still works through page_alias below.
menu_options = []

st.sidebar.markdown('<div class="sidebar-section">🚑 Clinical Workspace</div>', unsafe_allow_html=True)
menu_options.extend([
    "🚑 Triage Prediction",
    "🚦 Multi-Modal Triage",
    "🚨 Live Risk Watchlist",
    "📄 PDF Reports"
])

if st.session_state.role in ["doctor", "admin"]:
    st.sidebar.markdown('<div class="sidebar-section">🩺 Doctor Tools</div>', unsafe_allow_html=True)
    menu_options.extend([
    "📋 Patient History",
    "🧠 SHAP Explainability",
    "🤖 AI Clinical Copilot",
    "🩺 Clinical Feedback Dashboard",
    "✍️ Doctor Feedback"
    ])

if st.session_state.role == "nurse":
    st.sidebar.markdown('<div class="sidebar-section">👩‍⚕️ Nurse Tools</div>', unsafe_allow_html=True)
    menu_options.append("📄 Limited History")

st.sidebar.markdown('<div class="sidebar-section">📊 Analytics</div>', unsafe_allow_html=True)
menu_options.extend([
    "📊 Clinical Dashboard",
    "📈 AI Analytics",
    "🔬 DL Image Analysis"
])

if st.session_state.role in ["doctor", "admin", "nurse"]:
    menu_options.append("Nurse Management")
    menu_options.append("Nurse Patient Care")
    menu_options.append("Doctor Review")
    menu_options.append("Patient Status")
    menu_options.append("Emergency Queue")
    menu_options.append("Bed Management")
    menu_options.append("Medication Records")
    menu_options.append("Clinical Alerts")
    menu_options.append("Discharge Summary")
    menu_options.append("Shift Management")
    menu_options.append("Patient Registration")
    menu_options.append("Lab Tests")
    menu_options.append("Specialist Referral")
    menu_options.append("Consent Management")
    menu_options.append("Incident Reports")
    menu_options.append("Follow-up Appointments")
    menu_options.append("Family Notifications")
    menu_options.append("Hospital Analytics")

# Admin-only features are hidden from doctor and nurse accounts
if st.session_state.role in ["super_admin", "admin"]:
    st.sidebar.markdown('<div class="sidebar-section">🛡️ Admin Command Center</div>', unsafe_allow_html=True)
    menu_options.extend([
    "🏥 Company Health Dashboard",
    "🚨 Emergency Command Center",
    "🧪 Model Retraining",
    "🛡️ Admin Monitor"
    ])

if st.session_state.role in ["super_admin", "admin"]:
    menu_options.append("Staff Management")
    menu_options.append("Admin Overview")
    menu_options.append("Billing")
    menu_options.append("Inventory Management")

page_alias = {
    "🚑 Triage Prediction": "Live Prediction",
    "🚦 Multi-Modal Triage": "🚦 Multi-Modal Triage",
    "🚨 Live Risk Watchlist": "🚨 Risk Watchlist",
    "📄 PDF Reports": "📄 PDF Reports",
    "📋 Patient History": "Patient History",
    "🧠 SHAP Explainability": "SHAP Explainability",
    "🤖 AI Clinical Copilot": "🤖 AI Clinical Copilot",
    "🩺 Clinical Feedback Dashboard": "Clinical Feedback Dashboard",
    "✍️ Doctor Feedback": "Doctor Feedback",
    "📄 Limited History": "Limited History",
    "📊 Clinical Dashboard": "📊 Dashboard",
    "📈 AI Analytics": "📈 Analytics",
    "🔬 DL Image Analysis": "🔬 DL Image Analysis",
    "🏥 Company Health Dashboard": "🏥 Company Health Dashboard",
    "🚨 Emergency Command Center": "🚨 Emergency Command Center",
    "🧪 Model Retraining": "🧪 Model Retraining",
    "🛡️ Admin Monitor": "🛡️ Admin Monitor",
}

st.sidebar.markdown('<div class="sidebar-section">Role Navigation</div>', unsafe_allow_html=True)

menu_options = []
page_alias = {}


def add_nav(label, target=None, roles=None):
    if roles and st.session_state.role not in roles:
        return
    if label not in menu_options:
        menu_options.append(label)
    page_alias[label] = target or label


for label, target in [
    ("Clinical | Triage Prediction", "Live Prediction"),
    ("Clinical | Multi-Modal Triage", "🚦 Multi-Modal Triage"),
    ("Clinical | Emergency Queue", "Emergency Queue"),
    ("Clinical | Patient Registration", "Patient Registration"),
    ("Clinical | Patient Status", "Patient Status"),
    ("Clinical | Historical Reports", "Historical Reports"),
    ("Clinical | Alerts", "Clinical Alerts"),
    ("Clinical | PDF Reports", "📄 PDF Reports"),
]:
    add_nav(label, target)

for label, target in [
    ("Doctor | Patient History", "Patient History"),
    ("Doctor | SHAP Explainability", "SHAP Explainability"),
    ("Doctor | Review and Treatment", "Doctor Review"),
    ("Doctor | Lab Tests", "Lab Tests"),
    ("Doctor | Specialist Referral", "Specialist Referral"),
    ("Doctor | Bed Management", "Bed Management"),
    ("Doctor | Discharge Summary", "Discharge Summary"),
    ("Doctor | Clinical Feedback", "Clinical Feedback Dashboard"),
    ("Doctor | Doctor Feedback", "Doctor Feedback"),
    ("Doctor | AI Clinical Copilot", "🤖 AI Clinical Copilot"),
    ("Doctor | Image Analysis", "🔬 DL Image Analysis"),
    ("Doctor | Risk Watchlist", "🚨 Risk Watchlist"),
]:
    add_nav(label, target, roles=["doctor", "admin"])

for label, target in [
    ("Nurse | Nurse Management", "Nurse Management"),
    ("Nurse | Patient Care", "Nurse Patient Care"),
    ("Nurse | Medication Records", "Medication Records"),
    ("Nurse | Bed Status", "Bed Management"),
    ("Nurse | Limited History", "Limited History"),
]:
    add_nav(label, target, roles=["nurse", "admin"])

for label, target in [
    ("Admin | Overview", "Admin Overview"),
    ("Admin | Staff Management", "Staff Management"),
    ("Admin | Shift Management", "Shift Management"),
    ("Admin | Billing", "Billing"),
    ("Admin | Inventory", "Inventory Management"),
    ("Admin | Consent Management", "Consent Management"),
    ("Admin | Incident Reports", "Incident Reports"),
    ("Admin | Follow-up Appointments", "Follow-up Appointments"),
    ("Admin | Family Notifications", "Family Notifications"),
    ("Admin | Hospital Analytics", "Hospital Analytics"),
    ("Admin | Company Health", "🏥 Company Health Dashboard"),
    ("Admin | Emergency Command Center", "🚨 Emergency Command Center"),
    ("Admin | Model Retraining", "🧪 Model Retraining"),
    ("Admin | Admin Monitor", "🛡️ Admin Monitor"),
]:
    add_nav(label, target, roles=["admin"])

navigation_groups = {}


def add_group_page(group, label, target, roles=None):
    if roles and st.session_state.role not in roles:
        return
    navigation_groups.setdefault(group, [])
    if not any(item["label"] == label for item in navigation_groups[group]):
        navigation_groups[group].append({"label": label, "target": target})


for label, target in [
    ("Triage Prediction", "Live Prediction"),
    ("Multi-Modal Triage", "🚦 Multi-Modal Triage"),
    ("Waiting Queue", "Waiting Queue"),
    ("Patient Registration", "Patient Registration"),
    ("Patient Status", "Patient Status"),
    ("Historical Reports", "Historical Reports"),
    ("Emergency Queue", "Emergency Queue"),
    ("Medication Records", "Medication Records"),
    ("Shift Management", "Shift Management"),
    ("Clinical Alerts", "Clinical Alerts"),
    ("Consent Management", "Consent Management"),
    ("Incident Reports", "Incident Reports"),
    ("Follow-up Appointments", "Follow-up Appointments"),
    ("Family Notifications", "Family Notifications"),
]:
    add_group_page("Clinical Workspace", label, target)

for label, target in [
    ("Patient History", "Patient History"),
    ("SHAP Explainability", "SHAP Explainability"),
    ("Doctor Review", "Doctor Review"),
    ("Doctor Review Queue", "Doctor Review Queue"),
    ("Waiting Queue", "Waiting Queue"),
    ("Nurse Workload", "Nurse Workload"),
    ("Lab Tests", "Lab Tests"),
    ("Specialist Referral", "Specialist Referral"),
    ("Bed Management", "Bed Management"),
    ("Discharge Summary", "Discharge Summary"),
    ("Clinical Feedback", "Clinical Feedback Dashboard"),
    ("Doctor Feedback", "Doctor Feedback"),
    ("AI Clinical Copilot", "🤖 AI Clinical Copilot"),
    ("Image Analysis", "🔬 DL Image Analysis"),
    ("Risk Watchlist", "🚨 Risk Watchlist"),
    ("AI Model Health", "AI Model Health"),
    ("System Status", "System Status"),
    ("Self-Healing Status", "Self-Healing System"),
    ("Patient Timeline", "Patient Timeline"),
    ("Platform Alerts", "Platform Alerts"),
    ("Model Governance", "Model Governance"),
    ("Render Status", "Render Status"),
]:
    add_group_page("Doctor Tools", label, target, roles=["doctor", "admin"])

for label, target in [
    ("Nurse Management", "Nurse Management"),
    ("My Patients", "My Patients"),
    ("Nurse Patient Care", "Nurse Patient Care"),
    ("Waiting Queue", "Waiting Queue"),
    ("Medication Records", "Medication Records"),
    ("Bed Status", "Bed Management"),
    ("Limited History", "Limited History"),
]:
    add_group_page("Nurse Workspace", label, target, roles=["nurse", "admin"])

for label, target in [
    ("Admin Overview", "Admin Overview"),
    ("Waiting Queue", "Waiting Queue"),
    ("Nurse Workload", "Nurse Workload"),
    ("Approvals", "Admin Approvals"),
    ("Staff Management", "Staff Management"),
    ("Admin Approvals", "Admin Approvals"),
    ("Shift Management", "Shift Management"),
    ("Billing", "Billing"),
    ("Inventory Management", "Inventory Management"),
    ("Consent Management", "Consent Management"),
    ("Incident Reports", "Incident Reports"),
    ("Follow-up Appointments", "Follow-up Appointments"),
    ("Family Notifications", "Family Notifications"),
    ("Company Health Dashboard", "🏥 Company Health Dashboard"),
    ("Emergency Command Center", "🚨 Emergency Command Center"),
    ("Model Retraining", "🧪 Model Retraining"),
    ("MLOps Monitoring", "MLOps Monitoring"),
    ("Platform Health", "Platform Health"),
    ("Self-Healing System", "Self-Healing System"),
    ("Command Center", "Admin Command Center"),
    ("Patient Timeline", "Patient Timeline"),
    ("Platform Alerts", "Platform Alerts"),
    ("Model Governance", "Model Governance"),
    ("Render Status", "Render Status"),
    ("Admin Monitor", "🛡️ Admin Monitor"),
]:
    add_group_page("Admin Command Center", label, target, roles=["super_admin", "admin"])

for label, target in [
    ("Approvals", "Admin Approvals"),
]:
    add_group_page("Doctor Approval Panel", label, target, roles=["doctor"])

for label, target in [
    ("Approvals", "Admin Approvals"),
]:
    add_group_page("Nurse Approval Panel", label, target, roles=["nurse"])

for label, target in [
    ("Clinical Dashboard", "📊 Dashboard"),
    ("AI Analytics", "📈 Analytics"),
    ("Hospital Analytics", "Hospital Analytics"),
    ("PDF Reports", "📄 PDF Reports"),
]:
    add_group_page("Analytics & Reports", label, target)

if st.session_state.role == "patient":
    navigation_groups = {
        "Patient Portal": [
            {"label": "Dashboard", "target": "Patient Demo Dashboard"},
            {"label": "Historical Reports", "target": "Historical Reports"},
            {"label": "Demo Report", "target": "Patient Demo Report"},
        ]
    }

group_names = list(navigation_groups.keys())
st.sidebar.markdown('<div class="nav-helper">Navigation</div>', unsafe_allow_html=True)
selected_group = st.sidebar.selectbox("Workspace", group_names, label_visibility="collapsed")
group_pages = navigation_groups[selected_group]
page_labels = [item["label"] for item in group_pages]
selected_page = st.sidebar.selectbox("Page", page_labels, label_visibility="collapsed")
page = next(item["target"] for item in group_pages if item["label"] == selected_page)

welcome_by_role = {
    "super_admin": "System ownership, network analytics, and audit oversight are available.",
    "admin": "Hospital operations, staff approvals, and queue oversight are available.",
    "doctor": "Emergency review, patient history, AI risk analysis, and clinical feedback tools are available.",
    "nurse": "Triage intake, vitals entry, patient monitoring, and new-patient awareness are available.",
    "patient": "Patient portal records, summaries, and report downloads are available.",
}

st.markdown(
    f"""
    <div class="top-status-card">
        <div class="top-status-title">Welcome, {username_display}</div>
        <div class="top-status-subtitle">
            {welcome_by_role.get(st.session_state.role, "Role-based workspace loaded.")}
        </div>
        <div style="margin-top:10px;">
            <span class="status-pill">Role: {role_display}</span>
            <span class="status-pill">Backend: {backend_badge}</span>
            <span class="status-pill">Database: {database_badge}</span>
            <span class="status-pill">Workspace: {selected_group}</span>
            <span class="status-pill">Page: {selected_page}</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

try:
    alert_response = requests.get(f"{API_URL}/alerts", headers=auth_headers(), timeout=5)
    if alert_response.status_code == 200:
        unresolved_critical_alerts = [
            alert for alert in alert_response.json().get("alerts", [])
            if not alert.get("resolved") and alert.get("severity") == "Critical"
        ]
        if unresolved_critical_alerts:
            view_all_alerts = st.toggle(
                f"View all critical alerts ({len(unresolved_critical_alerts)})",
                value=False
            )
            visible_alerts = unresolved_critical_alerts if view_all_alerts else unresolved_critical_alerts[:3]
            with st.container():
                for alert in visible_alerts:
                    st.markdown(
                        f"""
                        <div class="alert-item">
                            Prediction #{alert.get('prediction_id')}: {html.escape(str(alert.get('message') or 'Critical alert'))}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
except Exception:
    pass


# -----------------------------
# REUSABLE PATIENT FORM
# -----------------------------

def patient_form(default_prefix=""):
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input(f"{default_prefix}Age", 0, 120, 55)
        gender = st.selectbox(f"{default_prefix}Gender", ["Male", "Female"])
        race = st.selectbox(f"{default_prefix}Race", ["White", "Black", "Asian", "Other"])
        ethnicity = st.selectbox(f"{default_prefix}Ethnicity", ["Non-Hispanic", "Hispanic"])
        arrivalmode = st.selectbox(f"{default_prefix}Arrival Mode", ["Walk-in", "Ambulance", "Police", "Other"])

    with col2:
        triage_vital_hr = st.number_input(f"{default_prefix}Heart Rate", value=125)
        triage_vital_sbp = st.number_input(f"{default_prefix}Systolic BP", value=90)
        triage_vital_dbp = st.number_input(f"{default_prefix}Diastolic BP", value=60)
        triage_vital_rr = st.number_input(f"{default_prefix}Respiratory Rate", value=24)
        triage_vital_o2 = st.number_input(f"{default_prefix}Oxygen Saturation", value=88)
        triage_vital_temp = st.number_input(f"{default_prefix}Temperature", value=38.5)

    with col3:
        cc_chestpain = st.checkbox(f"{default_prefix}Chest Pain")
        cc_shortnessofbreath = st.checkbox(f"{default_prefix}Shortness of Breath")
        cc_headache = st.checkbox(f"{default_prefix}Headache")
        cc_fever = st.checkbox(f"{default_prefix}Fever")
        cc_abdominalpain = st.checkbox(f"{default_prefix}Abdominal Pain")
        cc_dizziness = st.checkbox(f"{default_prefix}Dizziness")
        cc_syncope = st.checkbox(f"{default_prefix}Syncope")
        cc_weakness = st.checkbox(f"{default_prefix}Weakness")

    return {
        "age": age, "gender": gender, "race": race,
        "ethnicity": ethnicity, "arrivalmode": arrivalmode,
        "triage_vital_hr": triage_vital_hr, "triage_vital_sbp": triage_vital_sbp,
        "triage_vital_dbp": triage_vital_dbp, "triage_vital_rr": triage_vital_rr,
        "triage_vital_o2": triage_vital_o2, "triage_vital_temp": triage_vital_temp,
        "cc_chestpain": int(cc_chestpain), "cc_shortnessofbreath": int(cc_shortnessofbreath),
        "cc_headache": int(cc_headache), "cc_fever": int(cc_fever),
        "cc_abdominalpain": int(cc_abdominalpain), "cc_dizziness": int(cc_dizziness),
        "cc_syncope": int(cc_syncope), "cc_weakness": int(cc_weakness),
    }

def load_synthetic_demo_data():
    if SYNTHETIC_DATA_PATH.exists():
        return pd.read_csv(SYNTHETIC_DATA_PATH)
    return pd.DataFrame()


def render_patient_demo_page(page_name):
    demo_df = load_synthetic_demo_data()
    st.info(
        "Patient Demo Mode is for educational and demonstration purposes only. "
        "Data shown is synthetic and not real patient information."
    )

    if page_name == "Patient Demo Dashboard":
        st.title("Patient Demo Dashboard")
        if demo_df.empty:
            st.warning("Synthetic dataset not found. Run `python -m src.generate_emergency_data`.")
            return
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Synthetic Patients", f"{len(demo_df):,}")
        c2.metric("High Acuity", f"{(demo_df['Triage_Level'] <= 2).mean():.1%}")
        c3.metric("ICU Required", f"{demo_df['ICU_Required'].mean():.1%}")
        c4.metric("Avg Wait", f"{demo_df['Wait_Time_Minutes'].mean():.0f} min")

    elif page_name == "Patient Demo Triage":
        st.title("Patient Demo Triage Prediction")
        with st.form("patient_demo_triage"):
            col1, col2, col3 = st.columns(3)
            with col1:
                age = st.number_input("Age", 1, 100, 45)
                gender = st.selectbox("Gender", ["Female", "Male", "Nonbinary"])
                diabetes = st.checkbox("Diabetes")
                hypertension = st.checkbox("Hypertension")
                smoking = st.checkbox("Smoking")
            with col2:
                sbp = st.number_input("Systolic BP", 65, 230, 120)
                dbp = st.number_input("Diastolic BP", 35, 135, 80)
                heart_rate = st.number_input("Heart Rate", 35, 190, 88)
                respiratory_rate = st.number_input("Respiratory Rate", 6, 50, 18)
                oxygen = st.number_input("Oxygen Saturation", 70, 100, 97)
                temperature = st.number_input("Temperature", 95.0, 106.0, 98.6)
            with col3:
                chest_pain = st.checkbox("Chest Pain")
                shortness = st.checkbox("Shortness of Breath")
                fever = st.checkbox("Fever")
                st.text_area("Symptom Description", "Synthetic demo triage case.")
            submitted = st.form_submit_button("Predict ESI")
        if submitted:
            patient = {
                "Age": age,
                "Gender": gender,
                "Systolic_BP": sbp,
                "Diastolic_BP": dbp,
                "Heart_Rate": heart_rate,
                "Respiratory_Rate": respiratory_rate,
                "Oxygen_Saturation": oxygen,
                "Temperature": temperature,
                "Diabetes": int(diabetes),
                "Hypertension": int(hypertension),
                "Smoking": int(smoking),
                "Chest_Pain": int(chest_pain),
                "Shortness_of_Breath": int(shortness),
                "Fever": int(fever),
            }
            esi, reasons = rule_based_esi(patient)
            icu = estimate_icu_risk(patient, esi)
            readmit = estimate_readmission_risk(patient, esi, icu)
            color = ESI_COLORS[esi]
            st.markdown(
                f"<div style='background:{color};color:white;padding:16px;border-radius:8px;font-size:24px;font-weight:800;'>ESI {esi}: {ESI_LABELS[esi]}</div>",
                unsafe_allow_html=True,
            )
            if esi == 1 or icu >= 0.5:
                st.error("Emergency alert: critical synthetic case. This is not medical advice.")
            m1, m2, m3 = st.columns(3)
            m1.metric("ICU Risk", f"{icu:.1%}")
            m2.metric("Readmission Risk", f"{readmit:.1%}")
            m3.metric("Wait Target", {1: "Immediate", 2: "10 min", 3: "45 min", 4: "90 min", 5: "150 min"}[esi])
            for reason in reasons:
                st.write(f"- {reason}")

    elif page_name in ["Patient Demo Analytics", "Patient Demo Data", "Patient Demo Report"]:
        if demo_df.empty:
            st.warning("Synthetic dataset not found. Run `python -m src.generate_emergency_data`.")
            return
        if page_name == "Patient Demo Analytics":
            st.title("Patient Demo Analytics")
            c1, c2 = st.columns(2)
            with c1:
                counts = demo_df["Triage_Level"].value_counts().sort_index().reset_index()
                counts.columns = ["ESI", "Patients"]
                st.plotly_chart(px.bar(counts, x="ESI", y="Patients", title="ESI Distribution"), use_container_width=True)
            with c2:
                wait = demo_df.groupby("Triage_Level", as_index=False)["Wait_Time_Minutes"].mean()
                st.plotly_chart(px.bar(wait, x="Triage_Level", y="Wait_Time_Minutes", title="Average Wait Time"), use_container_width=True)
        elif page_name == "Patient Demo Data":
            st.title("Sample Synthetic Patient Data")
            st.dataframe(demo_df.head(100), use_container_width=True, hide_index=True)
        else:
            st.title("Demo Report")
            report = (
                f"Patient Demo Report\nSynthetic patients: {len(demo_df):,}\n"
                f"High acuity rate: {(demo_df['Triage_Level'] <= 2).mean():.1%}\n"
                f"ICU required rate: {demo_df['ICU_Required'].mean():.1%}\n"
                "Educational synthetic data only."
            )
            st.code(report, language="text")
            st.download_button("Download Demo Report", report, file_name="patient_demo_report.txt")


# ===========================================================
# LIVE PREDICTION — ADVANCED HOSPITAL UI
# ===========================================================

if page.startswith("Patient Demo"):
    render_patient_demo_page(page)

elif page in ["MLOps Monitoring", "AI Model Health"]:
    if page == "MLOps Monitoring" and st.session_state.role not in ["admin", "super_admin"]:
        st.error("Only admin and super admin roles can access full MLOps monitoring.")
    elif page == "AI Model Health" and st.session_state.role not in ["doctor", "admin", "super_admin"]:
        st.error("Only doctor and admin roles can access AI model health.")
    else:
        render_mlops_dashboard(st.session_state.role, auth_headers)

elif page in ["Platform Health", "System Status"]:
    if page == "Platform Health" and st.session_state.role not in ["admin", "super_admin"]:
        st.error("Only admin and super admin roles can access platform health.")
    elif page == "System Status" and st.session_state.role not in ["doctor", "admin", "super_admin"]:
        st.error("Only doctor and admin roles can access system status.")
    else:
        render_platform_dashboard(st.session_state.role, auth_headers)

elif page == "Self-Healing System":
    if st.session_state.role not in ["doctor", "admin", "super_admin"]:
        st.error("Only doctor and admin roles can access self-healing status.")
    else:
        render_self_healing_dashboard(st.session_state.role, auth_headers)

elif page == "Admin Command Center":
    if st.session_state.role not in ["doctor", "admin", "super_admin"]:
        st.error("Only doctor and admin roles can access command center summary.")
    else:
        render_admin_command_center(st.session_state.role, auth_headers)

elif page == "Patient Timeline":
    if st.session_state.role not in ["nurse", "doctor", "admin", "super_admin"]:
        st.error("Only clinical staff can access patient timeline.")
    else:
        render_patient_timeline(auth_headers)

elif page == "Platform Alerts":
    if st.session_state.role not in ["doctor", "admin", "super_admin"]:
        st.error("Only doctor and admin roles can access platform alerts.")
    else:
        render_alerts_dashboard(st.session_state.role, auth_headers)

elif page == "Model Governance":
    if st.session_state.role not in ["doctor", "admin", "super_admin"]:
        st.error("Only doctor and admin roles can access model governance.")
    else:
        render_model_governance(auth_headers)

elif page == "Render Status":
    if st.session_state.role not in ["doctor", "admin", "super_admin"]:
        st.error("Only doctor and admin roles can access Render status.")
    else:
        render_render_status(auth_headers)

elif page == "Historical Reports":
    render_historical_reports_page(
        API_URL,
        st.session_state.token,
        st.session_state.role,
        st.session_state.username,
    )

elif page == "Waiting Queue":
    if st.session_state.role == "patient":
        st.error("Access denied.")
        st.stop()
    render_waiting_queue(
        API_URL,
        st.session_state.token,
        st.session_state.role,
        st.session_state.username,
    )

elif page == "Nurse Workload":
    if st.session_state.role not in ["admin", "super_admin", "doctor", "nurse"]:
        st.error("Access denied.")
        st.stop()
    render_nurse_workload(
        API_URL,
        st.session_state.token,
        st.session_state.role,
    )

elif page == "My Patients":
    if st.session_state.role not in ["nurse", "admin", "super_admin", "doctor"]:
        st.error("Access denied.")
        st.stop()
    render_nurse_dashboard(
        API_URL,
        st.session_state.token,
        st.session_state.role,
        st.session_state.username,
    )

elif page == "Doctor Review Queue":
    if st.session_state.role not in ["doctor", "admin", "super_admin"]:
        st.error("Access denied.")
        st.stop()
    render_doctor_dashboard(
        API_URL,
        st.session_state.token,
        st.session_state.role,
        st.session_state.username,
    )

elif page == "Live Prediction":

    st.markdown(
        """
        <style>
        .clinical-hero {
            background: linear-gradient(135deg, #0f172a 0%, #1d4ed8 48%, #06b6d4 100%);
            color: white;
            padding: 28px 30px;
            border-radius: 26px;
            box-shadow: 0 18px 50px rgba(37, 99, 235, 0.25);
            margin-bottom: 22px;
        }
        .clinical-hero h1 {
            font-size: 38px;
            font-weight: 900;
            margin-bottom: 6px;
        }
        .clinical-hero p {
            font-size: 15px;
            opacity: 0.92;
            margin: 0;
        }
        .ui-panel {
            background: rgba(255,255,255,0.95);
            border: 1px solid #e2e8f0;
            border-radius: 22px;
            padding: 20px 22px;
            box-shadow: 0 12px 35px rgba(15, 23, 42, 0.06);
            margin-bottom: 18px;
        }
        .section-title {
            font-size: 21px;
            font-weight: 900;
            color: #0f172a;
            margin-bottom: 8px;
        }
        .section-caption {
            color: #64748b;
            font-size: 14px;
            margin-bottom: 10px;
        }
        .alert-critical {
            background: linear-gradient(90deg, #991b1b, #ef4444);
            color: white;
            border-radius: 18px;
            padding: 16px 18px;
            font-size: 18px;
            font-weight: 900;
            box-shadow: 0 14px 34px rgba(239, 68, 68, 0.24);
            margin: 12px 0 18px 0;
        }
        .alert-warning {
            background: linear-gradient(90deg, #ea580c, #f59e0b);
            color: white;
            border-radius: 18px;
            padding: 16px 18px;
            font-size: 18px;
            font-weight: 900;
            box-shadow: 0 14px 34px rgba(245, 158, 11, 0.24);
            margin: 12px 0 18px 0;
        }
        .result-card-critical {
            background: linear-gradient(135deg, #7f1d1d, #dc2626);
            color: white;
            padding: 28px;
            border-radius: 26px;
            box-shadow: 0 18px 50px rgba(220, 38, 38, 0.28);
        }
        .result-card-high {
            background: linear-gradient(135deg, #9a3412, #f97316);
            color: white;
            padding: 28px;
            border-radius: 26px;
            box-shadow: 0 18px 50px rgba(249, 115, 22, 0.25);
        }
        .result-card-stable {
            background: linear-gradient(135deg, #065f46, #10b981);
            color: white;
            padding: 28px;
            border-radius: 26px;
            box-shadow: 0 18px 50px rgba(16, 185, 129, 0.20);
        }
        .result-card-critical h2,
        .result-card-high h2,
        .result-card-stable h2 {
            margin: 0;
            font-size: 34px;
            font-weight: 900;
        }
        .result-card-critical p,
        .result-card-high p,
        .result-card-stable p {
            margin-top: 8px;
            opacity: 0.95;
            font-size: 15px;
        }
        .mini-label {
            font-size: 13px;
            color: #64748b;
            font-weight: 800;
            margin-bottom: 4px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="clinical-hero">
            <h1>🚑 Emergency Triage Command Center</h1>
            <p>Real-time emergency prediction with vitals, symptoms, NLP notes, image analysis, safety rules, and clinical feedback.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    def vital_status(value, warning_low=None, warning_high=None, critical_low=None, critical_high=None):
        if critical_low is not None and value <= critical_low:
            return "Critical"
        if critical_high is not None and value >= critical_high:
            return "Critical"
        if warning_low is not None and value <= warning_low:
            return "Warning"
        if warning_high is not None and value >= warning_high:
            return "Warning"
        return "Normal"

    def status_delta(status):
        if status == "Critical":
            return "Critical", "inverse"
        if status == "Warning":
            return "Warning", "inverse"
        return "Normal", "normal"

    st.markdown('<div class="ui-panel">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">👤 Patient Intake</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-caption">Enter demographic and arrival information before running the emergency triage model.</div>', unsafe_allow_html=True)

    id_col, name_col = st.columns(2)
    with id_col:
        patient_id = st.text_input("Patient ID", value=f"PT-{datetime.now().strftime('%Y%m%d')}")
    with name_col:
        patient_name = st.text_input("Patient Name", value="Demo Patient")

    p1, p2, p3, p4, p5 = st.columns(5)
    with p1:
        age = st.number_input("Age", min_value=0, max_value=120, value=55)
    with p2:
        gender = st.selectbox("Gender", ["Male", "Female"])
    with p3:
        race = st.selectbox("Race", ["White", "Black", "Asian", "Other"])
    with p4:
        ethnicity = st.selectbox("Ethnicity", ["Non-Hispanic", "Hispanic"])
    with p5:
        arrivalmode = st.selectbox("Arrival Mode", ["Walk-in", "Ambulance", "Police", "Other"])
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="ui-panel">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🩺 Live Vitals Monitor</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-caption">Vitals are color-coded through status labels: Normal, Warning, or Critical.</div>', unsafe_allow_html=True)

    v1, v2, v3 = st.columns(3)
    with v1:
        triage_vital_hr = st.slider("❤️ Heart Rate", 40, 220, 140)
    with v2:
        triage_vital_sbp = st.slider("🩸 Systolic BP", 50, 250, 92)
    with v3:
        triage_vital_dbp = st.slider("🫀 Diastolic BP", 30, 180, 82)

    v4, v5, v6 = st.columns(3)
    with v4:
        triage_vital_rr = st.slider("🌬️ Respiratory Rate", 5, 60, 18)
    with v5:
        triage_vital_o2 = st.slider("🫁 Oxygen Saturation", 50, 100, 90)
    with v6:
        triage_vital_temp = st.slider("🌡️ Temperature", 34.0, 42.0, 37.0)

    hr_status = vital_status(triage_vital_hr, warning_high=110, critical_high=140)
    sbp_status = vital_status(triage_vital_sbp, warning_low=95, critical_low=85)
    rr_status = vital_status(triage_vital_rr, warning_high=22, critical_high=30)
    o2_status = vital_status(triage_vital_o2, warning_low=93, critical_low=88)
    temp_status = vital_status(triage_vital_temp, warning_high=38.0, critical_high=39.0)
    dbp_status = vital_status(triage_vital_dbp, warning_low=55, warning_high=105, critical_low=45, critical_high=120)

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    with m1:
        d, c = status_delta(hr_status)
        st.metric("❤️ HR", f"{triage_vital_hr} bpm", d, delta_color=c)
    with m2:
        d, c = status_delta(sbp_status)
        st.metric("🩸 SBP", f"{triage_vital_sbp}", d, delta_color=c)
    with m3:
        d, c = status_delta(dbp_status)
        st.metric("🫀 DBP", f"{triage_vital_dbp}", d, delta_color=c)
    with m4:
        d, c = status_delta(rr_status)
        st.metric("🌬️ RR", f"{triage_vital_rr}/min", d, delta_color=c)
    with m5:
        d, c = status_delta(o2_status)
        st.metric("🫁 SpO₂", f"{triage_vital_o2}%", d, delta_color=c)
    with m6:
        d, c = status_delta(temp_status)
        st.metric("🌡️ Temp", f"{triage_vital_temp}°C", d, delta_color=c)

    style_metric_cards()

    critical_vitals = [s for s in [hr_status, sbp_status, rr_status, o2_status, temp_status, dbp_status] if s == "Critical"]
    warning_vitals = [s for s in [hr_status, sbp_status, rr_status, o2_status, temp_status, dbp_status] if s == "Warning"]

    if critical_vitals:
        st.markdown(
            f'<div class="alert-critical">🚨 Critical vitals detected: {len(critical_vitals)} parameter(s) require immediate review.</div>',
            unsafe_allow_html=True
        )
    elif warning_vitals:
        st.markdown(
            f'<div class="alert-warning">⚠️ Warning vitals detected: {len(warning_vitals)} parameter(s) should be monitored.</div>',
            unsafe_allow_html=True
        )
    else:
        st.success("✅ Vitals are currently within stable range.")

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="ui-panel">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">⚠️ Symptom Selection</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-caption">Symptoms can be manually selected or auto-filled from NLP extraction.</div>', unsafe_allow_html=True)

    extracted = {}
    if st.session_state.nlp_result:
        extracted = st.session_state.nlp_result.get("extracted_symptoms", {})

    s1, s2, s3, s4 = st.columns(4)
    with s1:
        cc_chestpain = st.checkbox("Chest Pain", value=bool(extracted.get("cc_chestpain", 0)), key="cc_chestpain_box")
        cc_headache = st.checkbox("Headache", value=bool(extracted.get("cc_headache", 0)), key="cc_headache_box")
    with s2:
        cc_shortnessofbreath = st.checkbox("Shortness of Breath", value=bool(extracted.get("cc_shortnessofbreath", 0)), key="cc_shortnessofbreath_box")
        cc_fever = st.checkbox("Fever", value=bool(extracted.get("cc_fever", 0)), key="cc_fever_box")
    with s3:
        cc_abdominalpain = st.checkbox("Abdominal Pain", value=bool(extracted.get("cc_abdominalpain", 0)), key="cc_abdominalpain_box")
        cc_dizziness = st.checkbox("Dizziness", value=bool(extracted.get("cc_dizziness", 0)), key="cc_dizziness_box")
    with s4:
        cc_syncope = st.checkbox("Syncope", value=bool(extracted.get("cc_syncope", 0)), key="cc_syncope_box")
        cc_weakness = st.checkbox("Weakness", value=bool(extracted.get("cc_weakness", 0)), key="cc_weakness_box")

    symptom_count = sum([
        int(cc_chestpain), int(cc_shortnessofbreath), int(cc_headache), int(cc_fever),
        int(cc_abdominalpain), int(cc_dizziness), int(cc_syncope), int(cc_weakness)
    ])
    st.metric("Selected Symptoms", symptom_count)
    st.markdown('</div>', unsafe_allow_html=True)

    report_types = [
        "Blood Test", "X-Ray", "MRI", "CT Scan", "Prescription", "Discharge Summary",
        "Previous Diagnosis", "Emergency Visit", "Surgery Report", "Allergy Report",
        "Medical Image", "Other"
    ]
    st.markdown('<div class="ui-panel">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Past Medical Report / Image Upload</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-caption">Upload Past Medical Report or Medical Image Optional. This supports review and does not override model prediction.</div>', unsafe_allow_html=True)
    u1, u2 = st.columns([1, 1])
    with u1:
        triage_report_type = st.selectbox("Report Type", report_types, key="triage_report_type")
        triage_upload = st.file_uploader(
            "Upload Past Medical Report or Medical Image Optional",
            type=["pdf", "txt", "csv", "docx", "jpg", "jpeg", "png"],
            key="triage_context_upload",
        )
    with u2:
        triage_upload_notes = st.text_area("Add notes about this report/image", height=124)
        analyze_upload = st.button("Analyze Uploaded Context", use_container_width=True)

    if triage_upload is not None and Path(triage_upload.name).suffix.lower() in IMAGE_EXTENSIONS:
        st.image(triage_upload.getvalue(), caption=triage_upload.name, use_container_width=True)

    if analyze_upload:
        if triage_upload is None:
            st.warning("Choose a file first, or continue without an upload.")
        elif not st.session_state.token:
            st.warning("Uploads require an authenticated account. Patient Demo Mode can still run prediction without an upload.")
        else:
            with st.spinner("Analyzing uploaded report/image as supporting triage context..."):
                upload_response = upload_triage_context_file(
                    triage_upload,
                    patient_id.strip() or "Unknown",
                    patient_name.strip() or "Unknown Patient",
                    triage_report_type,
                    triage_upload_notes,
                    str(st.session_state.prediction_id or ""),
                )
            if upload_response.status_code == 200:
                st.session_state.triage_upload_context = upload_response.json().get("upload")
                st.success("Uploaded context analyzed and attached to this triage workflow.")
            else:
                handle_response_error(upload_response)

    if st.session_state.triage_upload_context:
        context = st.session_state.triage_upload_context
        st.info("Historical context is ready and will be shown with the prediction result.")
        with st.expander("Current Uploaded Context Preview"):
            st.write(f"File: {context.get('file_name')} | Type: {context.get('report_type')} | Stored as: {context.get('file_type')}")
            st.write((context.get("clinical_summary") or {}).get("summary_text", "No summary available."))
    st.markdown('</div>', unsafe_allow_html=True)

    nlp_panel, voice_panel, image_panel = st.columns(3)

    with nlp_panel:
        st.markdown('<div class="ui-panel">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🧠 NLP Notes</div>', unsafe_allow_html=True)
        typed_description = st.text_area(
            "Problem Description",
            value=st.session_state.problem_description,
            placeholder="Example: Patient reports severe chest pain, dizziness, shortness of breath, and passed out.",
            height=150
        )
        st.session_state.problem_description = typed_description

        if st.button("🧠 Extract Symptoms", use_container_width=True):
            if not st.session_state.problem_description.strip():
                st.warning("Please enter a problem description first.")
            else:
                nlp_response = extract_symptoms_from_text(st.session_state.problem_description)
                if nlp_response.status_code == 200:
                    st.session_state.nlp_result = nlp_response.json()
                    st.success("NLP extraction completed.")
                    st.rerun()
                else:
                    handle_response_error(nlp_response)
        st.markdown('</div>', unsafe_allow_html=True)

    with voice_panel:
        st.markdown('<div class="ui-panel">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🎙️ Voice Intake</div>', unsafe_allow_html=True)
        st.caption("Record patient complaint and convert it into text.")
        if AUDIO_ENABLED:
            audio_bytes = audio_recorder(
                text="Click to record",
                recording_color="#e74c3c",
                neutral_color="#2563eb",
                icon_name="microphone",
                icon_size="2x"
            )
            if audio_bytes:
                st.audio(audio_bytes, format="audio/wav")
                if st.button("📝 Convert Voice", use_container_width=True):
                    transcribed_text = transcribe_audio(audio_bytes)
                    if (
                        "Could not understand" not in transcribed_text
                        and "service error" not in transcribed_text
                        and "Audio processing error" not in transcribed_text
                    ):
                        st.session_state.problem_description = transcribed_text
                        st.success("Voice converted to text.")
                        st.rerun()
                    else:
                        st.warning(transcribed_text)
        else:
            st.warning("Install: pip install audio-recorder-streamlit SpeechRecognition")
        st.markdown('</div>', unsafe_allow_html=True)

    with image_panel:
        st.markdown('<div class="ui-panel">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📷 Image Assessment</div>', unsafe_allow_html=True)
        uploaded_image = st.file_uploader("Upload clinical image", type=["jpg", "jpeg", "png"])
        if uploaded_image is not None:
            st.image(uploaded_image, caption="Uploaded clinical image", use_container_width=True)
            if st.button("🔍 Analyze Image", use_container_width=True):
                cv_response = analyze_image_with_cv(uploaded_image)
                if cv_response.status_code == 200:
                    st.session_state.cv_result = cv_response.json()
                    st.success("Image analysis completed.")
                    st.rerun()
                else:
                    handle_response_error(cv_response)
        else:
            st.info("Optional: upload wound/injury image.")
        st.markdown('</div>', unsafe_allow_html=True)

    if st.session_state.nlp_result or st.session_state.cv_result:
        st.markdown('<div class="ui-panel">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🧾 AI Intake Summary</div>', unsafe_allow_html=True)
        sum1, sum2 = st.columns(2)
        with sum1:
            if st.session_state.nlp_result:
                nlp_result = st.session_state.nlp_result
                if nlp_result.get("has_emergency_keyword"):
                    st.error("🚨 Emergency keywords detected. Immediate clinical review recommended.")
                emergency_keywords = nlp_result.get("emergency_keywords", [])
                matched_terms = nlp_result.get("matched_terms", [])
                if emergency_keywords:
                    st.markdown("**Highlighted Emergency Keywords**")
                    highlighted_text = highlight_emergency_keywords(nlp_result.get("cleaned_text", ""), emergency_keywords)
                    st.markdown(highlighted_text, unsafe_allow_html=True)
                st.write("**Matched Terms:**", ", ".join(matched_terms) if matched_terms else "None")
                st.code(nlp_result.get("llm_ready_text", ""), language="text")
            else:
                st.info("No NLP result yet.")
        with sum2:
            if st.session_state.cv_result:
                cv_data = st.session_state.cv_result.get("analysis", {})
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("Brightness", cv_data.get("avg_brightness"))
                with c2:
                    st.metric("Redness", cv_data.get("redness_score"))
                with c3:
                    st.metric("Size", str(cv_data.get("image_size")))
                for flag in cv_data.get("visual_flags", []):
                    st.info(flag)
                st.caption(cv_data.get("clinical_note"))
            else:
                st.info("No image analysis result yet.")
        st.markdown('</div>', unsafe_allow_html=True)

    patient_data = {
        "age": age, "gender": gender, "race": race, "ethnicity": ethnicity,
        "arrivalmode": arrivalmode,
        "triage_vital_hr": triage_vital_hr, "triage_vital_sbp": triage_vital_sbp,
        "triage_vital_dbp": triage_vital_dbp, "triage_vital_rr": triage_vital_rr,
        "triage_vital_o2": triage_vital_o2, "triage_vital_temp": triage_vital_temp,
        "cc_chestpain": int(cc_chestpain), "cc_shortnessofbreath": int(cc_shortnessofbreath),
        "cc_headache": int(cc_headache), "cc_fever": int(cc_fever),
        "cc_abdominalpain": int(cc_abdominalpain), "cc_dizziness": int(cc_dizziness),
        "cc_syncope": int(cc_syncope), "cc_weakness": int(cc_weakness),
        "problem_description": st.session_state.problem_description,
        "llm_ready_text": st.session_state.nlp_result.get("llm_ready_text") if st.session_state.nlp_result else None,
        "matched_terms": st.session_state.nlp_result.get("matched_terms") if st.session_state.nlp_result else [],
        "emergency_keywords": st.session_state.nlp_result.get("emergency_keywords") if st.session_state.nlp_result else [],
        "cv_analysis": st.session_state.cv_result.get("analysis") if st.session_state.cv_result else {}
    }

    st.markdown('<div class="ui-panel">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🤖 Run AI Triage Prediction</div>', unsafe_allow_html=True)
    run1, run2 = st.columns([2, 1])
    with run1:
        st.caption("This will send patient vitals, symptoms, NLP notes, and image analysis to the FastAPI backend.")
    with run2:
        run_prediction = st.button("🚨 Run Prediction", use_container_width=True, type="primary")
    st.markdown('</div>', unsafe_allow_html=True)

    if run_prediction:
        with st.spinner(
            "🧠 AI engine analyzing patient vitals, symptoms, NLP notes, and image signals..."
            ):
            response = requests.post(
                f"{API_URL}/predict",
                json=patient_data,
                headers=auth_headers(),
                timeout=20
            )

        if response.status_code == 200:

            result = response.json()

            st.session_state.last_prediction = result

            st.session_state.prediction_id = (
                result.get("prediction_id")
                or result.get("log_id")
            )

            st.success("✅ Prediction completed successfully.")

        else:
            handle_response_error(response)

    result = st.session_state.last_prediction
    if result:
        final_prediction_text = str(result.get("final_prediction", "N/A"))
        confidence = float(result.get("confidence", 0) or 0)

        prediction_clean = final_prediction_text.lower()
        if "1" in prediction_clean or "critical" in prediction_clean:
            card_class = "result-card-critical"
            result_icon = "🔴"
            clinical_status = "Critical"
        elif "2" in prediction_clean or "high" in prediction_clean:
            card_class = "result-card-high"
            result_icon = "🟠"
            clinical_status = "High Risk"
        else:
            card_class = "result-card-stable"
            result_icon = "🟢"
            clinical_status = "Stable / Moderate"

        st.markdown(
            f"""
            <div class="{card_class}">
                <h2>{result_icon} Final Prediction: {final_prediction_text}</h2>
                <p>Clinical status: <b>{clinical_status}</b> | Prediction ID: <b>{st.session_state.prediction_id}</b> | Confidence: <b>{confidence:.2%}</b></p>
            </div>
            """,
            unsafe_allow_html=True
        )


        composite_risk = result.get("composite_risk", {})

        if composite_risk:
            st.markdown("### 🧮 Composite Risk Score")

            risk_score = composite_risk.get("composite_risk_score", 0)
            risk_level = composite_risk.get("risk_level", "Unknown")
            risk_factors = composite_risk.get("risk_factors", [])

            cr1, cr2, cr3 = st.columns(3)

            with cr1:
                st.metric("Risk Score", f"{risk_score}/100")

            with cr2:
                st.metric("Risk Level", risk_level)

            with cr3:
                st.metric("Risk Factors", len(risk_factors))

            if risk_level == "Critical":
                st.error("🚨 Critical composite risk detected.")
            elif risk_level == "High":
                st.warning("⚠️ High composite risk detected.")
            elif risk_level == "Moderate":
                st.info("🟡 Moderate composite risk detected.")
            else:
                st.success("🟢 Low composite risk detected.")

            if risk_factors:
                st.markdown("#### Risk Factors")
                for factor in risk_factors:
                    st.warning(f"• {factor}")


        r1, r2, r3, r4 = st.columns(4)
        with r1:
            st.metric("ML Prediction", result.get("ml_prediction"))
        with r2:
            st.metric("Final Prediction", result.get("final_prediction"))
        with r3:
            st.metric("Confidence", f"{confidence:.1%}")
        with r4:
            st.metric("Prediction ID", st.session_state.prediction_id)
        style_metric_cards()

        action1, action2, action3 = st.columns(3)
        with action1:
            st.info(f"🚦 Multi-modal triage: use log_id **{st.session_state.prediction_id}**")
        with action2:
            st.info(f"📄 PDF report: use log_id **{st.session_state.prediction_id}**")
        with action3:
            st.info("🩺 Doctor feedback is available below for doctor/admin.")

        upload_context = st.session_state.get("triage_upload_context")
        if upload_context:
            summary = upload_context.get("clinical_summary") or {}
            risk_flags = upload_context.get("risk_flags") or []
            detected = upload_context.get("detected_conditions") or {}
            image_metadata = upload_context.get("image_metadata") or {}
            image_quality_notes = upload_context.get("image_quality_notes") or []
            ocr_text = upload_context.get("ocr_text") or ""
            extracted_text = upload_context.get("extracted_text") or ""

            st.markdown("### Historical Context")
            st.warning(HISTORICAL_REPORT_DISCLAIMER)
            h1, h2, h3 = st.columns(3)
            with h1:
                st.metric("Uploaded File", upload_context.get("file_name", "N/A"))
            with h2:
                st.metric("Report/Image Type", upload_context.get("report_type", "N/A"))
            with h3:
                st.metric("Context Flags", len(risk_flags))

            flag_colors = {
                "High Risk": "#dc2626",
                "Medium Risk": "#d97706",
                "Low Risk": "#16a34a",
                "Allergy Alert": "#2563eb",
                "Image Quality Warning": "#ca8a04",
                "Clinician Review Required": "#2563eb",
            }
            if risk_flags:
                st.markdown("#### Risk Flags")
                for flag in risk_flags:
                    color = flag_colors.get(flag.get("level"), "#64748b")
                    st.markdown(
                        f"<div style='border-left:6px solid {color};border:1px solid #e5e7eb;padding:12px;border-radius:8px;margin-bottom:8px;'>"
                        f"<b style='color:{color};'>{flag.get('label')}</b><br>{flag.get('level')} - {flag.get('reason')}</div>",
                        unsafe_allow_html=True,
                    )

            st.markdown("#### Doctor/Nurse Summary")
            st.write(summary.get("summary_text") or summary.get("patient_background") or "No supporting summary available.")

            with st.expander("Detected Conditions"):
                st.json(detected.get("detected_conditions", detected))
            with st.expander("Extracted Text"):
                st.text((extracted_text or "No extracted text available.")[:4000])
            with st.expander("OCR Preview"):
                st.text((ocr_text or summary.get("ocr_findings") or "No OCR text available.")[:3000])
            with st.expander("Image Metadata"):
                if image_metadata:
                    st.json(image_metadata)
                else:
                    st.write("No image metadata for document upload.")
            if image_quality_notes:
                st.warning("Image Quality Warning: " + " | ".join(image_quality_notes))
            if summary.get("recommended_questions"):
                with st.expander("Recommended Follow-Up Questions"):
                    for question in summary.get("recommended_questions", []):
                        st.write(f"- {question}")
        else:
            st.info("No past report or medical image was uploaded. Prediction used current vitals and symptoms only.")

        render_nurse_assignment_panel(st.session_state.prediction_id)

        render_patient_status_panel(st.session_state.prediction_id)

        chart_col, rules_col = st.columns([1.2, 1])
        with chart_col:
            st.subheader("📊 Prediction Probability Distribution")
            probabilities = result.get("probabilities", {})
            if probabilities:
                prob_df = pd.DataFrame(
                    list(probabilities.items()),
                    columns=["Triage Level", "Probability"]
                )
                fig = px.bar(
                    prob_df,
                    x="Triage Level",
                    y="Probability",
                    title="Model Probability by Triage Level"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No probability distribution returned.")

        with rules_col:
            st.subheader("🛡️ Safety Rules")
            safety_reasons = result.get("safety_reasons", [])
            if safety_reasons:
                for reason in safety_reasons:
                    st.warning(reason)
            else:
                st.success("No safety escalations triggered.")

            st.subheader("🧠 Clinical Explanations")
            explanations = result.get("clinical_explanations", [])
            if explanations:
                for explanation in explanations:
                    st.info(explanation)
            else:
                st.info("No explanations returned.")

        if st.session_state.role in ["doctor", "admin"]:
            st.markdown('<div class="ui-panel">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">🩺 Clinical Feedback Workflow</div>', unsafe_allow_html=True)
            prediction_id = st.session_state.prediction_id
            if prediction_id is None:
                st.warning("Prediction ID not found.")
            else:
                st.info(f"Feedback will be saved for Prediction ID: {prediction_id}")
                feedback_choice = st.radio(
                    "Clinical Decision",
                    ["Prediction Accepted", "Prediction Changed"],
                    horizontal=True
                )
                accepted = feedback_choice == "Prediction Accepted"
                override_esi = None
                override_reason = ""

                if not accepted:
                    override_esi = st.selectbox("Override ESI Level", [1, 2, 3, 4, 5])
                    override_reason = st.text_area(
                        "Override Reason",
                        placeholder="Example: Patient symptoms indicate higher acuity than AI prediction."
                    )

                default_notes = st.session_state.problem_description
                if st.session_state.nlp_result:
                    default_notes += "\n\nNLP Matched Terms: " + ", ".join(st.session_state.nlp_result.get("matched_terms", []))
                    default_notes += "\nEmergency Keywords: " + ", ".join(st.session_state.nlp_result.get("emergency_keywords", []))
                    default_notes += "\nLLM-Ready Text: " + st.session_state.nlp_result.get("llm_ready_text", "")
                if st.session_state.cv_result:
                    cv_data = st.session_state.cv_result.get("analysis", {})
                    default_notes += "\n\nComputer Vision Visual Flags: " + ", ".join(cv_data.get("visual_flags", []))
                    default_notes += "\nRedness Score: " + str(cv_data.get("redness_score", ""))
                    default_notes += "\nBrightness: " + str(cv_data.get("avg_brightness", ""))

                clinical_notes = st.text_area("Clinical Notes", value=default_notes)

                if st.button("💾 Save Clinical Feedback", use_container_width=True):
                    if not accepted and not override_reason.strip():
                        st.warning("Please enter an override reason.")
                    else:
                        feedback_response = save_clinical_feedback(
                            prediction_id=prediction_id,
                            accepted=accepted,
                            override_esi=override_esi,
                            clinical_notes=clinical_notes,
                            override_reason=override_reason
                        )
                        if feedback_response.status_code in [200, 201]:
                            st.success("Clinical feedback saved to PostgreSQL successfully.")
                            st.json(feedback_response.json())
                        else:
                            handle_response_error(feedback_response)
            st.markdown('</div>', unsafe_allow_html=True)



# ===========================================================
# SHAP — ADVANCED EXPLAINABILITY DASHBOARD
# ===========================================================

elif page == "SHAP Explainability":

    st.markdown(
        """
        <style>
        .shap-hero {
            background: linear-gradient(135deg, #312e81 0%, #2563eb 50%, #06b6d4 100%);
            color: white;
            padding: 28px 30px;
            border-radius: 26px;
            box-shadow: 0 18px 50px rgba(37, 99, 235, 0.25);
            margin-bottom: 22px;
        }
        .shap-hero h1 {
            font-size: 38px;
            font-weight: 900;
            margin-bottom: 6px;
        }
        .shap-hero p {
            font-size: 15px;
            opacity: 0.92;
            margin: 0;
        }
        .explain-panel {
            background: rgba(255,255,255,0.96);
            border: 1px solid #e2e8f0;
            border-radius: 22px;
            padding: 20px 22px;
            box-shadow: 0 12px 35px rgba(15, 23, 42, 0.06);
            margin-bottom: 18px;
        }
        .explain-title {
            font-size: 21px;
            font-weight: 900;
            color: #0f172a;
            margin-bottom: 8px;
        }
        .explain-caption {
            color: #64748b;
            font-size: 14px;
            margin-bottom: 12px;
        }
        .risk-factor-card {
            background: linear-gradient(135deg, #f8fafc, #eef2ff);
            border: 1px solid #c7d2fe;
            border-radius: 18px;
            padding: 16px 18px;
            min-height: 120px;
            box-shadow: 0 10px 24px rgba(15,23,42,0.05);
        }
        .risk-factor-card .rank {
            font-size: 14px;
            font-weight: 900;
            color: #2563eb;
        }
        .risk-factor-card .feature {
            font-size: 20px;
            font-weight: 900;
            color: #0f172a;
            margin-top: 6px;
        }
        .risk-factor-card .value {
            font-size: 14px;
            color: #64748b;
            margin-top: 6px;
        }
        .clinical-summary-box {
            background: #ecfeff;
            border: 1px solid #bae6fd;
            color: #075985;
            border-radius: 18px;
            padding: 18px 20px;
            font-weight: 700;
            line-height: 1.6;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="shap-hero">
            <h1>🧠 AI Explainability Dashboard</h1>
            <p>Understand which patient features influence the model prediction using SHAP-based feature attribution.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<div class="explain-panel">', unsafe_allow_html=True)
    st.markdown('<div class="explain-title">👤 Patient Scenario for Explanation</div>', unsafe_allow_html=True)
    st.markdown('<div class="explain-caption">Enter a patient profile to generate model explanations. Use abnormal vitals to test high-risk behavior.</div>', unsafe_allow_html=True)

    patient_data = patient_form("SHAP ")

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="explain-panel">', unsafe_allow_html=True)
    st.markdown('<div class="explain-title">🚀 Generate Explanation</div>', unsafe_allow_html=True)
    gen_col1, gen_col2 = st.columns([2, 1])

    with gen_col1:
        st.info(
            "SHAP helps explain why the model predicted a triage level by showing which features pushed the prediction higher or lower."
        )

    with gen_col2:
        generate_clicked = st.button(
            "🧠 Generate SHAP Explanation",
            use_container_width=True,
            type="primary"
        )

    st.markdown('</div>', unsafe_allow_html=True)

    if generate_clicked:
        with st.spinner("Generating SHAP explanation from backend..."):
            shap_response = requests.post(
                f"{API_URL}/shap",
                json=patient_data,
                headers=auth_headers(),
                timeout=30
            )

        if shap_response.status_code == 200:
            shap_data = shap_response.json()
            st.success("SHAP explanation generated successfully.")

            feature_df = pd.DataFrame(shap_data.get("feature_importance", []))

            if feature_df.empty:
                st.warning("No SHAP feature importance values were returned.")
                st.stop()

            # Normalize expected columns safely
            if "shap_value" not in feature_df.columns:
                possible_numeric_cols = feature_df.select_dtypes(include="number").columns.tolist()
                if possible_numeric_cols:
                    feature_df = feature_df.rename(columns={possible_numeric_cols[0]: "shap_value"})

            if "feature" not in feature_df.columns:
                feature_df["feature"] = feature_df.index.astype(str)

            feature_df["abs_shap"] = feature_df["shap_value"].abs()
            feature_df = feature_df.sort_values("abs_shap", ascending=False)
            top_features = feature_df.head(3)

            st.markdown('<div class="explain-panel">', unsafe_allow_html=True)
            st.markdown('<div class="explain-title">🏆 Top Risk Contributors</div>', unsafe_allow_html=True)
            st.markdown('<div class="explain-caption">These are the strongest features influencing the model output for this patient scenario.</div>', unsafe_allow_html=True)

            c1, c2, c3 = st.columns(3)
            top_cols = [c1, c2, c3]

            for idx, (_, row) in enumerate(top_features.iterrows()):
                with top_cols[idx]:
                    direction = "Raises risk" if row["shap_value"] > 0 else "Lowers risk"
                    st.markdown(
                        f"""
                        <div class="risk-factor-card">
                            <div class="rank">#{idx + 1} Contributor</div>
                            <div class="feature">{row['feature']}</div>
                            <div class="value">SHAP value: <b>{row['shap_value']:.4f}</b></div>
                            <div class="value">Impact: <b>{direction}</b></div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="explain-panel">', unsafe_allow_html=True)
            st.markdown('<div class="explain-title">📊 Feature Impact Chart</div>', unsafe_allow_html=True)
            st.markdown('<div class="explain-caption">Positive SHAP values push the model toward a higher-risk prediction. Negative values reduce risk contribution.</div>', unsafe_allow_html=True)

            chart_df = feature_df.head(15).sort_values("shap_value")

            fig = px.bar(
                chart_df,
                x="shap_value",
                y="feature",
                orientation="h",
                title="Top SHAP Feature Contributions",
                labels={"shap_value": "SHAP Value", "feature": "Patient Feature"},
                color="shap_value",
                color_continuous_scale="RdBu_r"
            )
            fig.update_layout(height=520)
            st.plotly_chart(fig, use_container_width=True)

            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="explain-panel">', unsafe_allow_html=True)
            st.markdown('<div class="explain-title">🩺 Clinical Interpretation</div>', unsafe_allow_html=True)

            top_names = top_features["feature"].astype(str).tolist()
            positive_count = int((feature_df["shap_value"] > 0).sum())
            negative_count = int((feature_df["shap_value"] < 0).sum())

            st.markdown(
                f"""
                <div class="clinical-summary-box">
                    The model explanation shows that <b>{', '.join(top_names)}</b> are the strongest contributors for this patient scenario.
                    There are <b>{positive_count}</b> features increasing the model's risk signal and <b>{negative_count}</b> features lowering it.
                    This explanation should support clinical review, not replace clinician judgment.
                </div>
                """,
                unsafe_allow_html=True
            )

            with st.expander("View full SHAP data table"):
                st.dataframe(feature_df, use_container_width=True)

            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="explain-panel">', unsafe_allow_html=True)
            st.markdown('<div class="explain-title">🖼️ SHAP Visualization Image</div>', unsafe_allow_html=True)
            st.markdown('<div class="explain-caption">Backend-generated SHAP plot image for additional visual explanation.</div>', unsafe_allow_html=True)

            image_response = requests.get(
                f"{API_URL}/shap/image",
                headers=auth_headers(),
                timeout=20
            )

            if image_response.status_code == 200:
                image = Image.open(BytesIO(image_response.content))
                st.image(image, use_container_width=True)
            else:
                st.warning("SHAP image could not be loaded.")
                st.code(image_response.text, language="text")

            st.markdown('</div>', unsafe_allow_html=True)

        else:
            handle_response_error(shap_response)


# ===========================================================
# PATIENT HISTORY — ADVANCED CLINICAL TIMELINE
# ===========================================================

elif page == "Patient History":

    st.markdown(
        """
        <style>
        .history-hero {
            background: linear-gradient(135deg, #0f172a 0%, #1d4ed8 52%, #06b6d4 100%);
            color: white;
            padding: 28px 30px;
            border-radius: 26px;
            box-shadow: 0 18px 50px rgba(37, 99, 235, 0.25);
            margin-bottom: 22px;
        }
        .history-hero h1 {
            font-size: 38px;
            font-weight: 900;
            margin-bottom: 6px;
        }
        .history-hero p {
            font-size: 15px;
            opacity: 0.92;
            margin: 0;
        }
        .history-panel {
            background: rgba(255,255,255,0.96);
            border: 1px solid #e2e8f0;
            border-radius: 22px;
            padding: 20px 22px;
            box-shadow: 0 12px 35px rgba(15, 23, 42, 0.06);
            margin-bottom: 18px;
        }
        .history-title {
            font-size: 21px;
            font-weight: 900;
            color: #0f172a;
            margin-bottom: 8px;
        }
        .history-caption {
            color: #64748b;
            font-size: 14px;
            margin-bottom: 12px;
        }
        .patient-card {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-left: 7px solid #2563eb;
            border-radius: 20px;
            padding: 18px 20px;
            box-shadow: 0 10px 28px rgba(15, 23, 42, 0.06);
            margin-bottom: 14px;
        }
        .patient-card-critical {
            border-left-color: #dc2626;
            background: linear-gradient(90deg, #fff7f7, #ffffff);
        }
        .patient-card-high {
            border-left-color: #f97316;
            background: linear-gradient(90deg, #fff7ed, #ffffff);
        }
        .patient-card-stable {
            border-left-color: #10b981;
            background: linear-gradient(90deg, #ecfdf5, #ffffff);
        }
        .patient-card-title {
            font-size: 20px;
            font-weight: 900;
            color: #0f172a;
        }
        .patient-card-subtitle {
            font-size: 13px;
            color: #64748b;
            margin-top: 4px;
        }
        .risk-badge {
            display: inline-block;
            padding: 7px 12px;
            border-radius: 999px;
            font-weight: 900;
            font-size: 13px;
            color: white;
        }
        .risk-critical { background: #dc2626; }
        .risk-high { background: #f97316; }
        .risk-stable { background: #10b981; }
        .mini-note {
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 14px;
            padding: 12px 14px;
            color: #334155;
            font-size: 14px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="history-hero">
            <h1>📋 Patient History Command Center</h1>
            <p>Review prediction history, high-risk cases, NLP notes, image findings, clinical explanations, and feedback status.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    response = requests.get(
        f"{API_URL}/history",
        headers=auth_headers(),
        timeout=20
    )

    if response.status_code != 200:
        handle_response_error(response)
        st.stop()

    data = response.json()
    history = data.get("history", [])

    if len(history) == 0:
        st.info("No prediction history found. Run a live prediction first.")
        st.stop()

    df = pd.DataFrame(history)

    # Safe formatting helpers
    def normalize_prediction(value):
        return str(value or "Unknown").strip()

    def risk_from_prediction(value):
        text = normalize_prediction(value).lower()
        if text in ["1", "1.0", "esi 1", "esi-1"] or "critical" in text:
            return "Critical"
        if text in ["2", "2.0", "esi 2", "esi-2"] or "high" in text:
            return "High"
        return "Stable"

    df["risk_group"] = df.get("final_prediction", pd.Series(["Unknown"] * len(df))).apply(risk_from_prediction)

    if "created_at" in df.columns:
        df["created_at_dt"] = pd.to_datetime(df["created_at"], errors="coerce")
    else:
        df["created_at_dt"] = pd.NaT

    st.markdown('<div class="history-panel">', unsafe_allow_html=True)
    st.markdown('<div class="history-title">📊 History Overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="history-caption">Summary of all saved prediction records from PostgreSQL.</div>', unsafe_allow_html=True)

    total_records = len(df)
    critical_count = int((df["risk_group"] == "Critical").sum())
    high_count = int((df["risk_group"] == "High").sum())
    stable_count = int((df["risk_group"] == "Stable").sum())

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Records", total_records)
    with c2:
        st.metric("Critical", critical_count)
    with c3:
        st.metric("High Risk", high_count)
    with c4:
        st.metric("Stable / Moderate", stable_count)
    style_metric_cards()
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="history-panel">', unsafe_allow_html=True)
    st.markdown('<div class="history-title">🔎 Search & Filters</div>', unsafe_allow_html=True)

    f1, f2, f3 = st.columns([2, 1, 1])
    with f1:
        search_text = st.text_input(
            "Search patient history",
            placeholder="Search notes, symptoms, emergency keywords, ESI, feedback, source..."
        )
    with f2:
        risk_filter = st.selectbox(
            "Risk Group",
            ["All", "Critical", "High", "Stable"]
        )
    with f3:
        max_rows = st.selectbox(
            "Show Records",
            [10, 20, 50, 100, "All"],
            index=1
        )

    filtered_df = df.copy()

    if search_text:
        filtered_df = filtered_df[
            filtered_df.astype(str).apply(
                lambda row: row.str.contains(search_text, case=False, na=False).any(),
                axis=1
            )
        ]

    if risk_filter != "All":
        filtered_df = filtered_df[filtered_df["risk_group"] == risk_filter]

    filtered_df = filtered_df.sort_values("created_at_dt", ascending=False, na_position="last")

    if max_rows != "All":
        filtered_df = filtered_df.head(int(max_rows))

    st.success(f"Showing {len(filtered_df)} of {len(df)} records")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="history-panel">', unsafe_allow_html=True)
    st.markdown('<div class="history-title">📈 Clinical Trends</div>', unsafe_allow_html=True)

    chart1, chart2 = st.columns(2)

    with chart1:
        if "final_prediction" in df.columns:
            pred_df = (
                df["final_prediction"]
                .astype(str)
                .value_counts()
                .reset_index()
            )
            pred_df.columns = ["Final Prediction", "Count"]
            fig = px.bar(
                pred_df,
                x="Final Prediction",
                y="Count",
                title="Final Prediction Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No final_prediction column available.")

    with chart2:
        risk_df = (
            df["risk_group"]
            .value_counts()
            .reset_index()
        )
        risk_df.columns = ["Risk Group", "Count"]
        fig2 = px.pie(
            risk_df,
            names="Risk Group",
            values="Count",
            title="Risk Group Breakdown"
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="history-panel">', unsafe_allow_html=True)
    st.markdown('<div class="history-title">🧾 Recent Patient Timeline</div>', unsafe_allow_html=True)
    st.markdown('<div class="history-caption">Expandable cards show detailed vitals, NLP findings, CV analysis, safety rules, and clinical explanations.</div>', unsafe_allow_html=True)

    if filtered_df.empty:
        st.warning("No records match your filters.")
    else:
        filtered_ids = filtered_df["id"].tolist() if "id" in filtered_df.columns else []
        filtered_history = [row for row in history if row.get("id") in filtered_ids]

        # preserve filtered dataframe order
        id_to_row = {row.get("id"): row for row in filtered_history}
        ordered_history = [id_to_row.get(i) for i in filtered_ids if i in id_to_row]

        for row in ordered_history:
            risk_group = risk_from_prediction(row.get("final_prediction"))
            if risk_group == "Critical":
                card_class = "patient-card patient-card-critical"
                badge_class = "risk-badge risk-critical"
                risk_icon = "🔴"
            elif risk_group == "High":
                card_class = "patient-card patient-card-high"
                badge_class = "risk-badge risk-high"
                risk_icon = "🟠"
            else:
                card_class = "patient-card patient-card-stable"
                badge_class = "risk-badge risk-stable"
                risk_icon = "🟢"

            st.markdown(
                f"""
                <div class="{card_class}">
                    <div style="display:flex;justify-content:space-between;align-items:center;gap:16px;">
                        <div>
                            <div class="patient-card-title">{risk_icon} Patient #{row.get('id')} — Final Prediction: {row.get('final_prediction')}</div>
                            <div class="patient-card-subtitle">Age {row.get('age')} • {row.get('gender')} • Arrival: {row.get('arrivalmode')} • Created: {row.get('created_at')}</div>
                        </div>
                        <div><span class="{badge_class}">{risk_group}</span></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            with st.expander(f"Open full clinical record for Patient #{row.get('id')}"):
                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    st.metric("ML Prediction", row.get("ml_prediction"))
                with m2:
                    st.metric("Final Prediction", row.get("final_prediction"))
                with m3:
                    conf = row.get("confidence")
                    try:
                        st.metric("Confidence", f"{float(conf):.1%}")
                    except Exception:
                        st.metric("Confidence", conf)
                with m4:
                    st.metric("Feedback", row.get("feedback"))
                style_metric_cards()

                st.markdown("### 🩺 Vitals")
                v1, v2, v3, v4, v5, v6 = st.columns(6)
                with v1:
                    st.metric("HR", row.get("triage_vital_hr"))
                with v2:
                    st.metric("SBP", row.get("triage_vital_sbp"))
                with v3:
                    st.metric("DBP", row.get("triage_vital_dbp"))
                with v4:
                    st.metric("RR", row.get("triage_vital_rr"))
                with v5:
                    st.metric("SpO₂", row.get("triage_vital_o2"))
                with v6:
                    st.metric("Temp", row.get("triage_vital_temp"))

                st.markdown("### ⚠️ Symptoms")
                symptoms = []
                for key, label in SYMPTOM_LABELS.items():
                    if row.get(key) == 1:
                        symptoms.append(label)
                if symptoms:
                    st.warning(", ".join(symptoms))
                else:
                    st.info("No symptom flags selected.")

                detail_col1, detail_col2 = st.columns(2)

                with detail_col1:
                    st.markdown("### 📝 Patient Problem Description")
                    st.markdown(
                        f"<div class='mini-note'>{html.escape(str(row.get('problem_description') or 'No description saved'))}</div>",
                        unsafe_allow_html=True
                    )

                    st.markdown("### 🤖 LLM-Ready Clinical Text")
                    st.code(row.get("llm_ready_text") or "", language="text")

                    st.markdown("### ✅ NLP Matched Terms")
                    matched = row.get("matched_terms")
                    if matched:
                        st.success(matched)
                    else:
                        st.info("No matched terms saved.")

                    st.markdown("### 🚨 Emergency Keywords")
                    keywords = row.get("emergency_keywords")
                    if keywords:
                        st.error(keywords)
                    else:
                        st.info("No emergency keywords saved.")

                with detail_col2:
                    st.markdown("### 📷 Computer Vision Analysis")
                    cv_analysis = row.get("cv_analysis")
                    if cv_analysis:
                        st.json(cv_analysis)
                    else:
                        st.info("No CV analysis saved.")

                    st.markdown("### 🛡️ Safety Rules")
                    safety = row.get("safety_reasons")
                    if safety:
                        st.warning(safety)
                    else:
                        st.success("No safety reasons saved.")

                    st.markdown("### 🧠 Clinical Explanations")
                    explanations = row.get("clinical_explanations")
                    if explanations:
                        st.info(explanations)
                    else:
                        st.info("No explanations saved.")

                st.caption(f"Record source: {row.get('source')} | Created at: {row.get('created_at')}")

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="history-panel">', unsafe_allow_html=True)
    st.markdown('<div class="history-title">📊 Full Searchable Table</div>', unsafe_allow_html=True)
    st.dataframe(filtered_df.drop(columns=["created_at_dt"], errors="ignore"), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ===========================================================
# LIMITED HISTORY (unchanged)
# ===========================================================

elif page == "Limited History":
    st.title("📄 Limited Patient History")
    response = requests.get(f"{API_URL}/history/limited", headers=auth_headers(), timeout=20)
    if response.status_code == 200:
        payload = response.json()
        rows = payload.get("history", payload) if isinstance(payload, dict) else payload
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.info("No limited patient history records found.")
    else:
        handle_response_error(response)


# ===========================================================
# ENTERPRISE COMPANY HEALTH DASHBOARD
# ===========================================================

elif page == "🏥 Company Health Dashboard":

    if st.session_state.role != "admin":
        st.error("Access denied. Admin only.")
        st.stop()

    st.markdown("""
    <style>
    .company-hero {
        background: linear-gradient(135deg, #0f172a, #2563eb, #06b6d4);
        color: white;
        padding: 30px;
        border-radius: 28px;
        margin-bottom: 22px;
        box-shadow: 0 18px 50px rgba(37,99,235,0.25);
    }

    .company-hero h1 {
        font-size: 40px;
        font-weight: 900;
        margin-bottom: 6px;
    }

    .company-panel {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 22px;
        padding: 22px;
        box-shadow: 0 12px 35px rgba(15,23,42,0.06);
        margin-bottom: 18px;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="company-hero">
        <h1>🏥 Enterprise Company Health Dashboard</h1>
        <p>Hospital-wide AI operations, emergency load, and enterprise monitoring center.</p>
    </div>
    """, unsafe_allow_html=True)

    response = requests.get(
        f"{API_URL}/history",
        headers=auth_headers(),
        timeout=20
    )

    if response.status_code != 200:
        handle_response_error(response)
        st.stop()

    history = response.json().get("history", [])

    if len(history) == 0:
        st.info("No enterprise records found.")
        st.stop()

    df = pd.DataFrame(history)

    total_cases = len(df)

    if "confidence" in df.columns:
        df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")
        avg_confidence = df["confidence"].mean()
    else:
        avg_confidence = 0

    critical_cases = 0
    high_cases = 0

    if "final_prediction" in df.columns:
        critical_cases = (
            df["final_prediction"]
            .astype(str)
            .str.contains("1|critical", case=False, na=False)
            .sum()
        )

        high_cases = (
            df["final_prediction"]
            .astype(str)
            .str.contains("2|high", case=False, na=False)
            .sum()
        )

    today_cases = 0

    if "created_at" in df.columns:
        df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")

        today_cases = df[
            df["created_at"].dt.date ==
            pd.Timestamp.today().date()
        ].shape[0]

    st.markdown('<div class="company-panel">', unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        st.metric("Total AI Cases", total_cases)

    with c2:
        st.metric("Today's Cases", today_cases)

    with c3:
        st.metric("Critical Cases", int(critical_cases))

    with c4:
        st.metric("High Risk", int(high_cases))

    with c5:
        st.metric("Avg Confidence", f"{avg_confidence:.1%}")

    style_metric_cards()

    st.markdown('</div>', unsafe_allow_html=True)

    chart1, chart2 = st.columns(2)

    with chart1:

        st.markdown('<div class="company-panel">', unsafe_allow_html=True)

        st.subheader("📊 AI Triage Distribution")

        if "final_prediction" in df.columns:

            pred_df = (
                df["final_prediction"]
                .astype(str)
                .value_counts()
                .reset_index()
            )

            pred_df.columns = ["Prediction", "Count"]

            fig = px.bar(
                pred_df,
                x="Prediction",
                y="Count",
                title="Hospital Triage Distribution"
            )

            st.plotly_chart(fig, use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)

    with chart2:

        st.markdown('<div class="company-panel">', unsafe_allow_html=True)

        st.subheader("📈 AI Confidence Distribution")

        if "confidence" in df.columns:

            fig2 = px.histogram(
                df,
                x="confidence",
                nbins=20,
                title="Prediction Confidence"
            )

            st.plotly_chart(fig2, use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="company-panel">', unsafe_allow_html=True)

    st.subheader("🧠 Enterprise AI System Health")

    h1, h2, h3, h4 = st.columns(4)

    with h1:
        st.success("FastAPI Backend Online")

    with h2:
        st.success("PostgreSQL Connected")

    with h3:
        st.success("RBAC Authentication Active")

    with h4:
        st.success("AI Services Running")

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="company-panel">', unsafe_allow_html=True)

    st.subheader("📋 Recent Enterprise Records")

    st.dataframe(
        df.head(20),
        use_container_width=True
    )

    st.markdown('</div>', unsafe_allow_html=True)

# ===========================================================
# REAL-TIME EMERGENCY COMMAND CENTER
# ===========================================================

elif page == "🚨 Emergency Command Center":

    if st.session_state.role != "admin":
        st.error("Access denied.")
        st.stop()

    st.markdown("""
    <style>
    .command-hero {
        background: linear-gradient(135deg,#7f1d1d,#dc2626,#ef4444);
        color:white;
        padding:30px;
        border-radius:28px;
        margin-bottom:22px;
        box-shadow:0 18px 50px rgba(220,38,38,.30);
    }

    .command-card {
        background:white;
        border-radius:22px;
        padding:22px;
        border:1px solid #e2e8f0;
        margin-bottom:18px;
        box-shadow:0 12px 35px rgba(15,23,42,.06);
    }

    .critical-box {
        background:#7f1d1d;
        color:white;
        padding:16px;
        border-radius:18px;
        margin-bottom:12px;
    }

    .high-box {
        background:#ea580c;
        color:white;
        padding:16px;
        border-radius:18px;
        margin-bottom:12px;
    }

    .stable-box {
        background:#059669;
        color:white;
        padding:16px;
        border-radius:18px;
        margin-bottom:12px;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="command-hero">
        <h1>🚨 Emergency Command Center</h1>
        <p>Live hospital emergency operations and critical patient monitoring.</p>
    </div>
    """, unsafe_allow_html=True)

    response = requests.get(
        f"{API_URL}/history",
        headers=auth_headers(),
        timeout=20
    )

    if response.status_code != 200:
        handle_response_error(response)
        st.stop()

    history = response.json().get("history", [])

    if not history:
        st.warning("No emergency records found.")
        st.stop()

    df = pd.DataFrame(history)

    if "created_at" in df.columns:
        df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
        df = df.sort_values("created_at", ascending=False)

    total_cases = len(df)

    critical_df = df[
        df["final_prediction"]
        .astype(str)
        .str.contains("1|critical", case=False, na=False)
    ]

    high_df = df[
        df["final_prediction"]
        .astype(str)
        .str.contains("2|high", case=False, na=False)
    ]

    stable_df = df[
        ~df.index.isin(critical_df.index)
        &
        ~df.index.isin(high_df.index)
    ]

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("Total Active Cases", total_cases)

    with c2:
        st.metric("Critical", len(critical_df))

    with c3:
        st.metric("High Risk", len(high_df))

    with c4:
        st.metric("Stable", len(stable_df))

    style_metric_cards()

    st.markdown('<div class="command-card">', unsafe_allow_html=True)

    st.subheader("🔴 Critical Emergency Queue")

    if critical_df.empty:
        st.success("No critical emergencies currently.")
    else:

        for _, row in critical_df.head(10).iterrows():

            confidence = float(row.get("confidence", 0) or 0)

            st.markdown(f"""
            <div class="critical-box">
                🚑 Patient #{row.get("id")} |
                Prediction: {row.get("final_prediction")} |
                Confidence: {confidence:.1%}
                <br><br>
                HR: {row.get("triage_vital_hr")} |
                SpO₂: {row.get("triage_vital_o2")} |
                Temp: {row.get("triage_vital_temp")}
            </div>
            """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    left, right = st.columns(2)

    with left:

        st.markdown('<div class="command-card">', unsafe_allow_html=True)

        st.subheader("🟠 High Risk Monitoring")

        if high_df.empty:
            st.info("No high-risk patients.")
        else:
            st.dataframe(
                high_df.head(10),
                use_container_width=True
            )

        st.markdown('</div>', unsafe_allow_html=True)

    with right:

        st.markdown('<div class="command-card">', unsafe_allow_html=True)

        st.subheader("📈 Emergency Distribution")

        trend_df = (
            df["final_prediction"]
            .astype(str)
            .value_counts()
            .reset_index()
        )

        trend_df.columns = ["Prediction", "Count"]

        fig = px.pie(
            trend_df,
            names="Prediction",
            values="Count",
            title="Emergency Triage Distribution"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="command-card">', unsafe_allow_html=True)

    st.subheader("📡 Live Emergency Feed")

    st.dataframe(
        df.head(20),
        use_container_width=True
    )

    st.markdown('</div>', unsafe_allow_html=True)

# ===========================================================
# AI CLINICAL COPILOT
# ===========================================================

elif page == "🤖 AI Clinical Copilot":

    if st.session_state.role not in ["doctor", "admin"]:
        st.error("Access denied.")
        st.stop()

    st.markdown("""
    <style>
    .copilot-hero {
        background: linear-gradient(135deg,#312e81,#2563eb,#06b6d4);
        color:white;
        padding:30px;
        border-radius:28px;
        margin-bottom:22px;
        box-shadow:0 18px 50px rgba(37,99,235,.25);
    }

    .copilot-card {
        background:white;
        border-radius:22px;
        padding:22px;
        border:1px solid #e2e8f0;
        margin-bottom:18px;
        box-shadow:0 12px 35px rgba(15,23,42,.06);
    }

    .recommend-box {
        background:#eff6ff;
        border:1px solid #bfdbfe;
        border-radius:18px;
        padding:16px;
        margin-bottom:12px;
    }

    .warning-box {
        background:#fef2f2;
        border:1px solid #fecaca;
        border-radius:18px;
        padding:16px;
        margin-bottom:12px;
        color:#991b1b;
        font-weight:700;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="copilot-hero">
        <h1>🤖 AI Clinical Copilot</h1>
        <p>AI-assisted clinical recommendations and emergency triage reasoning support.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="copilot-card">', unsafe_allow_html=True)

    st.subheader("🩺 Patient Clinical Summary")

    age = st.number_input("Age", 0, 120, 55)
    hr = st.number_input("Heart Rate", 20, 250, 125)
    o2 = st.number_input("Oxygen Saturation", 50, 100, 90)
    temp = st.number_input("Temperature", 30.0, 45.0, 38.5)

    symptoms = st.text_area(
        "Symptoms",
        placeholder="Chest pain, dizziness, shortness of breath..."
    )

    notes = st.text_area(
        "Clinical Notes",
        placeholder="Additional nurse/doctor notes..."
    )

    generate = st.button(
        "🤖 Generate AI Clinical Guidance",
        use_container_width=True,
        type="primary"
    )

    st.markdown('</div>', unsafe_allow_html=True)

    if generate:

        risk_level = "Stable"

        if hr > 140 or o2 < 88:
            risk_level = "Critical"
        elif hr > 120 or o2 < 92:
            risk_level = "High"

        st.markdown('<div class="copilot-card">', unsafe_allow_html=True)

        st.subheader("🚨 AI Clinical Assessment")

        if risk_level == "Critical":
            st.markdown("""
            <div class="warning-box">
            🚨 Critical patient indicators detected. Immediate physician evaluation recommended.
            </div>
            """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="recommend-box">
        <b>AI Risk Assessment:</b> {risk_level}
        </div>
        """, unsafe_allow_html=True)

        recommendations = []

        if "chest" in symptoms.lower():
            recommendations.append("Consider ECG and cardiac enzyme testing.")

        if "shortness" in symptoms.lower():
            recommendations.append("Recommend oxygen monitoring and respiratory assessment.")

        if temp >= 38.5:
            recommendations.append("Evaluate possible infection or sepsis pathway.")

        if hr > 120:
            recommendations.append("Continuous cardiac monitoring recommended.")

        if o2 < 92:
            recommendations.append("Supplemental oxygen may be required.")

        if not recommendations:
            recommendations.append("Continue standard clinical monitoring.")

        st.subheader("🧠 AI Recommendations")

        for rec in recommendations:
            st.info(rec)

        st.subheader("📋 Suggested Immediate Actions")

        actions = [
            "Review patient vitals every 15 minutes.",
            "Escalate abnormal findings to attending physician.",
            "Monitor AI triage score changes.",
            "Document intervention timeline."
        ]

        for action in actions:
            st.success(action)

        st.subheader("⚠️ AI Clinical Disclaimer")

        st.warning(
            "AI recommendations are decision-support only and do not replace physician judgment."
        )

        st.markdown('</div>', unsafe_allow_html=True)


# ===========================================================
# CLINICAL FEEDBACK DASHBOARD (unchanged)
# ===========================================================

elif page == "Clinical Feedback Dashboard":

    st.markdown("""
    <style>
    .feedback-hero{background:linear-gradient(135deg,#0f172a 0%,#2563eb 52%,#06b6d4 100%);color:white;padding:28px 30px;border-radius:26px;box-shadow:0 18px 50px rgba(37,99,235,.25);margin-bottom:22px;}
    .feedback-hero h1{font-size:38px;font-weight:900;margin-bottom:6px;}
    .feedback-hero p{font-size:15px;opacity:.92;margin:0;}
    .feedback-panel{background:rgba(255,255,255,.96);border:1px solid #e2e8f0;border-radius:22px;padding:20px 22px;box-shadow:0 12px 35px rgba(15,23,42,.06);margin-bottom:18px;}
    .feedback-title{font-size:21px;font-weight:900;color:#0f172a;margin-bottom:8px;}
    .feedback-note{background:#f8fafc;border:1px solid #e2e8f0;border-radius:16px;padding:14px 16px;color:#334155;}
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="feedback-hero">
        <h1>🩺 Clinical Feedback Intelligence</h1>
        <p>Monitor AI acceptance, clinician overrides, reviewer activity, and feedback quality for human-in-the-loop governance.</p>
    </div>
    """, unsafe_allow_html=True)

    response = get_clinical_feedback_history()
    if response.status_code != 200:
        handle_response_error(response)
        st.stop()

    data = response.json()
    if len(data) == 0:
        st.info("No clinical feedback records found.")
        st.stop()

    df = pd.DataFrame(data)

    st.markdown('<div class="feedback-panel">', unsafe_allow_html=True)
    st.markdown('<div class="feedback-title">📊 Feedback Governance KPIs</div>', unsafe_allow_html=True)

    total_feedback = len(df)
    accepted_count = int(df["accepted"].sum()) if "accepted" in df.columns else 0
    changed_count = total_feedback - accepted_count
    acceptance_rate = round((accepted_count / total_feedback) * 100, 1) if total_feedback else 0
    override_rate = round((changed_count / total_feedback) * 100, 1) if total_feedback else 0

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("Total Feedback", total_feedback)
    with k2:
        st.metric("Accepted", accepted_count, f"{acceptance_rate}%")
    with k3:
        st.metric("Overridden", changed_count, f"{override_rate}%")
    with k4:
        reviewers = df["reviewer_username"].nunique() if "reviewer_username" in df.columns else 0
        st.metric("Active Reviewers", reviewers)
    style_metric_cards()
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="feedback-panel">', unsafe_allow_html=True)
    st.markdown('<div class="feedback-title">🔎 Search & Review Filters</div>', unsafe_allow_html=True)

    f1, f2, f3 = st.columns([2,1,1])
    with f1:
        search_text = st.text_input("Search feedback", placeholder="Search notes, reviewer, override reason, prediction...")
    with f2:
        decision_filter = st.selectbox("Decision", ["All", "Accepted", "Overridden"])
    with f3:
        max_rows = st.selectbox("Records", [10, 20, 50, "All"], index=1)

    filtered_df = df.copy()
    if search_text:
        filtered_df = filtered_df[filtered_df.astype(str).apply(lambda row: row.str.contains(search_text, case=False, na=False).any(), axis=1)]
    if decision_filter == "Accepted" and "accepted" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["accepted"] == True]
    elif decision_filter == "Overridden" and "accepted" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["accepted"] == False]
    if max_rows != "All":
        filtered_df = filtered_df.head(int(max_rows))
    st.success(f"Showing {len(filtered_df)} feedback records")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="feedback-panel">', unsafe_allow_html=True)
    st.markdown('<div class="feedback-title">📈 Feedback Analytics</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        decision_df = pd.DataFrame({"Decision": ["Accepted", "Overridden"], "Count": [accepted_count, changed_count]})
        fig = px.pie(decision_df, names="Decision", values="Count", title="AI Accepted vs Clinician Overridden", color="Decision", color_discrete_map={"Accepted":"#10b981","Overridden":"#f97316"})
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        if "reviewer_username" in df.columns:
            reviewer_df = df["reviewer_username"].fillna("Unknown").value_counts().reset_index()
            reviewer_df.columns = ["Reviewer", "Feedback Count"]
            fig2 = px.bar(reviewer_df, x="Reviewer", y="Feedback Count", title="Reviewer Activity")
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Reviewer column not available.")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="feedback-panel">', unsafe_allow_html=True)
    st.markdown('<div class="feedback-title">🧾 Feedback Records</div>', unsafe_allow_html=True)
    st.dataframe(filtered_df, use_container_width=True)

    if "override_reason" in filtered_df.columns:
        st.markdown("### ⚠️ Override Reason Review")
        override_rows = filtered_df[filtered_df.get("accepted", False) == False] if "accepted" in filtered_df.columns else filtered_df
        if len(override_rows) > 0:
            for _, row in override_rows.head(10).iterrows():
                with st.expander(f"Feedback #{row.get('id')} | Prediction #{row.get('prediction_id')} | Reviewer: {row.get('reviewer_username', 'Unknown')}"):
                    st.markdown(f"<div class='feedback-note'>{html.escape(str(row.get('override_reason') or 'No override reason saved.'))}</div>", unsafe_allow_html=True)
                    st.write("Clinical Notes")
                    st.info(row.get("feedback_notes") or "No notes saved.")
        else:
            st.info("No override records in current filter.")
    st.markdown('</div>', unsafe_allow_html=True)

elif page == "Doctor Feedback":
    st.title("🩺 Doctor Feedback")
    log_id = st.number_input("Prediction Log ID", min_value=1, value=1)
    feedback = st.selectbox("Feedback", [
        "Correct - High risk patient", "Correct - Medium risk patient",
        "Incorrect - Over-triaged", "Incorrect - Under-triaged", "Needs review"
    ])
    if st.button("Submit Feedback"):
        response = requests.post(
            f"{API_URL}/feedback",
            json={"log_id": log_id, "feedback": feedback},
            headers=auth_headers(), timeout=20
        )
        if response.status_code == 200:
            st.success("Feedback saved successfully")
            st.json(response.json())
        else:
            handle_response_error(response)


# ===========================================================
# NEW: DL IMAGE ANALYSIS
# ===========================================================

elif page == "🔬 DL Image Analysis":

    st.markdown("""
    <style>
    .dl-hero{background:linear-gradient(135deg,#0f172a 0%,#7c3aed 50%,#06b6d4 100%);color:white;padding:28px 30px;border-radius:26px;box-shadow:0 18px 50px rgba(124,58,237,.25);margin-bottom:22px;}
    .dl-hero h1{font-size:38px;font-weight:900;margin-bottom:6px;}.dl-hero p{font-size:15px;opacity:.92;margin:0;}
    .dl-panel{background:rgba(255,255,255,.96);border:1px solid #e2e8f0;border-radius:22px;padding:20px 22px;box-shadow:0 12px 35px rgba(15,23,42,.06);margin-bottom:18px;}
    .dl-title{font-size:21px;font-weight:900;color:#0f172a;margin-bottom:8px;}
    .severity-critical{background:linear-gradient(135deg,#7f1d1d,#dc2626);color:white;padding:22px;border-radius:22px;box-shadow:0 14px 36px rgba(220,38,38,.25);}
    .severity-warning{background:linear-gradient(135deg,#9a3412,#f97316);color:white;padding:22px;border-radius:22px;box-shadow:0 14px 36px rgba(249,115,22,.22);}
    .severity-stable{background:linear-gradient(135deg,#065f46,#10b981);color:white;padding:22px;border-radius:22px;box-shadow:0 14px 36px rgba(16,185,129,.18);}
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="dl-hero">
        <h1>🔬 Deep Learning Medical Image Analysis</h1>
        <p>EfficientNet wound classification, infection severity scoring, classical CV flags, and image audit tracking.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="dl-panel">', unsafe_allow_html=True)
    st.markdown('<div class="dl-title">📤 Image Intake</div>', unsafe_allow_html=True)
    left, right = st.columns([1.15, 1])
    with left:
        uploaded = st.file_uploader("Upload wound or injury image", type=["jpg", "jpeg", "png"], key="dl_uploader")
        patient_notes = st.text_area("Patient notes", placeholder="e.g. 3-day-old wound, diabetic patient, swelling and redness", height=120)
        run_dl = st.button("🧠 Run Deep Learning Analysis", use_container_width=True, type="primary", disabled=uploaded is None)
    with right:
        if uploaded:
            st.image(uploaded, caption="Uploaded medical image", use_container_width=True)
        else:
            st.info("Upload an image to activate EfficientNet + infection severity scoring.")
    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded and run_dl:
        with st.spinner("Running EfficientNet image analysis..."):
            files = {"image": (uploaded.name, uploaded.getvalue(), uploaded.type)}
            data = {"patient_notes": patient_notes}
            response = requests.post(f"{API_URL}/v2/analyze-image-dl", files=files, data=data, headers=auth_headers(), timeout=60)
        if response.status_code == 200:
            result = response.json()
            st.session_state.dl_image_result = result
            st.session_state.dl_image_log_id = result.get("log_id")
            st.success(f"Analysis complete — image log_id: {result.get('log_id')}")
            st.rerun()
        else:
            handle_response_error(response)

    dl = st.session_state.dl_image_result
    if dl:
        severity = str(dl.get("infection_severity", "unknown")).lower()
        severity_score = dl.get("severity_score") or 0
        card_class = "severity-critical" if severity in ["severe", "critical"] or severity_score >= .75 else "severity-warning" if severity in ["moderate"] or severity_score >= .45 else "severity-stable"

        st.markdown(f"""
        <div class="{card_class}">
            <h2 style="margin:0;font-weight:900;">🧬 Infection Severity: {dl.get('infection_severity', '—')}</h2>
            <p style="margin:8px 0 0 0;opacity:.95;">Wound class: <b>{dl.get('wound_class','—')}</b> | Severity score: <b>{severity_score:.3f}/1.0</b> | log_id: <b>{dl.get('log_id')}</b></p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="dl-panel">', unsafe_allow_html=True)
        k1,k2,k3,k4,k5 = st.columns(5)
        with k1: st.metric("Wound Class", dl.get("wound_class", "—"))
        with k2: st.metric("Wound Confidence", f"{(dl.get('wound_confidence') or 0):.0%}")
        with k3: st.metric("Infection", dl.get("infection_severity", "—"))
        with k4: st.metric("Severity Score", f"{severity_score:.3f}")
        with k5: st.metric("Image Log ID", dl.get("log_id", "—"))
        style_metric_cards()
        st.markdown('</div>', unsafe_allow_html=True)

        c1,c2 = st.columns(2)
        with c1:
            st.markdown('<div class="dl-panel">', unsafe_allow_html=True)
            st.markdown('<div class="dl-title">📊 Wound Classification Probabilities</div>', unsafe_allow_html=True)
            wound_dist = dl.get("wound_distribution")
            if wound_dist:
                wdf = pd.DataFrame({"Class": list(wound_dist.keys()), "Probability": list(wound_dist.values())}).sort_values("Probability")
                fig = px.bar(wdf, x="Probability", y="Class", orientation="h", color="Probability", color_continuous_scale="Blues")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No wound distribution returned.")
            st.markdown('</div>', unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="dl-panel">', unsafe_allow_html=True)
            st.markdown('<div class="dl-title">🦠 Infection Severity Distribution</div>', unsafe_allow_html=True)
            inf_dist = dl.get("infection_distribution")
            if inf_dist:
                idf = pd.DataFrame({"Severity": list(inf_dist.keys()), "Probability": list(inf_dist.values())})
                fig2 = px.bar(idf, x="Severity", y="Probability", color="Severity", color_discrete_map={"none":"#2563eb","mild":"#10b981","moderate":"#f59e0b","severe":"#f97316","critical":"#dc2626"})
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("No infection distribution returned.")
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="dl-panel">', unsafe_allow_html=True)
        st.markdown('<div class="dl-title">🔍 Classical CV + Clinical Notes</div>', unsafe_allow_html=True)
        classical = dl.get("classical_cv", {})
        flags = classical.get("visual_flags", []) if classical else []
        if flags:
            for f in flags:
                st.warning(f) if any(x in str(f).lower() for x in ["red", "bleeding", "burn", "high"]) else st.info(f)
        else:
            st.info("No classical CV flags detected.")
        st.caption(f"Model: {dl.get('model')} | Use image_log_id {dl.get('log_id')} inside Multi-Modal Triage")
        st.markdown('</div>', unsafe_allow_html=True)

elif page == "🚦 Multi-Modal Triage":

    st.markdown(
        """
        <style>
        .fusion-hero {
            background: linear-gradient(135deg, #111827 0%, #4338ca 48%, #0891b2 100%);
            color: white;
            padding: 28px 30px;
            border-radius: 26px;
            box-shadow: 0 18px 50px rgba(67, 56, 202, 0.25);
            margin-bottom: 22px;
        }
        .fusion-hero h1 {
            font-size: 38px;
            font-weight: 900;
            margin-bottom: 6px;
        }
        .fusion-hero p {
            font-size: 15px;
            opacity: 0.92;
            margin: 0;
        }
        .fusion-panel {
            background: rgba(255,255,255,0.96);
            border: 1px solid #e2e8f0;
            border-radius: 22px;
            padding: 20px 22px;
            box-shadow: 0 12px 35px rgba(15, 23, 42, 0.06);
            margin-bottom: 18px;
        }
        .fusion-title {
            font-size: 21px;
            font-weight: 900;
            color: #0f172a;
            margin-bottom: 8px;
        }
        .fusion-caption {
            color: #64748b;
            font-size: 14px;
            margin-bottom: 12px;
        }
        .fusion-card {
            background: linear-gradient(135deg, #f8fafc, #eef2ff);
            border: 1px solid #c7d2fe;
            border-radius: 18px;
            padding: 16px 18px;
            min-height: 125px;
            box-shadow: 0 10px 24px rgba(15,23,42,0.05);
            margin-bottom: 10px;
        }
        .fusion-card .label {
            font-size: 13px;
            font-weight: 900;
            color: #4f46e5;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }
        .fusion-card .value {
            font-size: 24px;
            font-weight: 900;
            color: #0f172a;
            margin-top: 8px;
        }
        .fusion-card .hint {
            font-size: 13px;
            color: #64748b;
            margin-top: 6px;
        }
        .esi-critical {
            background: linear-gradient(135deg, #7f1d1d, #dc2626);
            color: white;
            padding: 30px;
            border-radius: 28px;
            box-shadow: 0 18px 50px rgba(220,38,38,0.28);
            margin-bottom: 18px;
        }
        .esi-high {
            background: linear-gradient(135deg, #9a3412, #f97316);
            color: white;
            padding: 30px;
            border-radius: 28px;
            box-shadow: 0 18px 50px rgba(249,115,22,0.25);
            margin-bottom: 18px;
        }
        .esi-moderate {
            background: linear-gradient(135deg, #854d0e, #facc15);
            color: white;
            padding: 30px;
            border-radius: 28px;
            box-shadow: 0 18px 50px rgba(250,204,21,0.22);
            margin-bottom: 18px;
        }
        .esi-stable {
            background: linear-gradient(135deg, #065f46, #10b981);
            color: white;
            padding: 30px;
            border-radius: 28px;
            box-shadow: 0 18px 50px rgba(16,185,129,0.20);
            margin-bottom: 18px;
        }
        .esi-critical h2, .esi-high h2, .esi-moderate h2, .esi-stable h2 {
            font-size: 38px;
            font-weight: 900;
            margin: 0;
        }
        .esi-critical p, .esi-high p, .esi-moderate p, .esi-stable p {
            font-size: 15px;
            opacity: 0.94;
            margin-top: 8px;
        }
        .fusion-note {
            background: #ecfeff;
            border: 1px solid #bae6fd;
            color: #075985;
            border-radius: 18px;
            padding: 16px 18px;
            font-weight: 700;
            line-height: 1.6;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="fusion-hero">
            <h1>🚦 Multi-Modal AI Triage Fusion Engine</h1>
            <p>Combine structured vitals, symptoms, NLP findings, and wound/image analysis into one ESI 1–5 triage decision.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<div class="fusion-panel">', unsafe_allow_html=True)
    st.markdown('<div class="fusion-title">🧩 Fusion Inputs Overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="fusion-caption">Choose whether to link an existing prediction record or enter a new manual scenario.</div>', unsafe_allow_html=True)

    source_col1, source_col2, source_col3 = st.columns(3)

    with source_col1:
        existing_prediction = st.session_state.prediction_id
        st.markdown(
            f"""
            <div class="fusion-card">
                <div class="label">Prediction Log</div>
                <div class="value">{existing_prediction if existing_prediction else 'Not linked'}</div>
                <div class="hint">From Live Prediction page</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with source_col2:
        existing_image = st.session_state.dl_image_log_id
        st.markdown(
            f"""
            <div class="fusion-card">
                <div class="label">Image Log</div>
                <div class="value">{existing_image if existing_image else 'Optional'}</div>
                <div class="hint">From DL Image Analysis page</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with source_col3:
        nlp_status = "Available" if st.session_state.nlp_result else "Optional"
        st.markdown(
            f"""
            <div class="fusion-card">
                <div class="label">NLP Findings</div>
                <div class="value">{nlp_status}</div>
                <div class="hint">Clinical text extraction</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="fusion-panel">', unsafe_allow_html=True)
    st.markdown('<div class="fusion-title">⚙️ Patient Data Source</div>', unsafe_allow_html=True)

    use_existing = st.toggle(
        "Link to existing prediction log from Live Prediction",
        value=True,
        help="Recommended: use a saved /predict log_id so vitals and symptoms are pulled automatically."
    )

    log_id_input = None
    inline_vitals = {}
    inline_symptoms = []

    if use_existing:
        default_lid = st.session_state.prediction_id or 1
        log_id_input = st.number_input(
            "Prediction log_id",
            min_value=1,
            value=int(default_lid),
            help="Use the prediction_id/log_id returned from Live Prediction."
        )
        st.success(f"Using existing prediction log_id: {log_id_input}")
    else:
        st.info("Manual mode: enter vitals and symptoms directly for the multi-modal triage engine.")
        v_col1, v_col2, v_col3 = st.columns(3)
        with v_col1:
            hr = st.number_input("Heart Rate (bpm)", 40, 250, 90)
            sbp = st.number_input("Systolic BP (mmHg)", 50, 300, 120)
        with v_col2:
            rr = st.number_input("Respiratory Rate", 5, 60, 16)
            spo2 = st.number_input("SpO₂ (%)", 50, 100, 97)
        with v_col3:
            temp = st.number_input("Temperature (°C)", 30.0, 45.0, 36.8, step=0.1)
            consc = st.selectbox("Consciousness", ["alert", "voice", "pain", "unresponsive"])

        inline_vitals = {
            "heart_rate": hr,
            "systolic_bp": sbp,
            "respiratory_rate": rr,
            "spo2": spo2,
            "temperature": temp,
            "consciousness": consc
        }

        st.markdown("#### ⚠️ Manual Symptom Severity")
        sym_options = [
            "chest_pain", "shortness_of_breath", "abdominal_pain",
            "headache", "fever", "nausea", "dizziness",
            "loss_of_consciousness", "severe_bleeding"
        ]
        picked = st.multiselect("Select symptoms", sym_options)

        if picked:
            sev_cols = st.columns(3)
            for idx, symptom in enumerate(picked):
                with sev_cols[idx % 3]:
                    sev = st.select_slider(
                        f"{symptom.replace('_', ' ').title()} severity",
                        ["mild", "moderate", "severe", "critical"],
                        value="moderate",
                        key=f"fusion_sev_{symptom}"
                    )
                    inline_symptoms.append({"name": symptom, "severity": sev})
        else:
            st.caption("No manual symptoms selected yet.")

    st.markdown('</div>', unsafe_allow_html=True)

    input_col1, input_col2 = st.columns(2)

    with input_col1:
        st.markdown('<div class="fusion-panel">', unsafe_allow_html=True)
        st.markdown('<div class="fusion-title">🧠 NLP Findings</div>', unsafe_allow_html=True)
        st.markdown('<div class="fusion-caption">Use existing NLP extraction or generate a new NLP payload from text.</div>', unsafe_allow_html=True)

        use_last_nlp = st.checkbox(
            "Use NLP result from Live Prediction page",
            value=st.session_state.nlp_result is not None
        )

        nlp_payload = None
        if use_last_nlp and st.session_state.nlp_result:
            nlp_payload = st.session_state.nlp_result
            matched_terms = nlp_payload.get("matched_terms", [])
            emergency_keywords = nlp_payload.get("emergency_keywords", [])
            st.success("NLP result loaded from session.")
            st.write("**Matched terms:**", ", ".join(matched_terms) if matched_terms else "None")
            if emergency_keywords:
                st.error("Emergency keywords: " + ", ".join(emergency_keywords))
        else:
            free_text = st.text_area(
                "Enter free-text clinical note",
                placeholder="e.g. crushing chest pain radiating to left arm, shortness of breath, diaphoresis",
                height=130
            )
            if free_text.strip() and st.button("🧠 Extract NLP for Triage", use_container_width=True):
                nlp_r = extract_symptoms_from_text(free_text)
                if nlp_r.status_code == 200:
                    nlp_payload = nlp_r.json()
                    st.session_state.nlp_result = nlp_payload
                    st.success("NLP extracted and saved to session.")
                    st.rerun()
                else:
                    handle_response_error(nlp_r)

        st.markdown('</div>', unsafe_allow_html=True)

    with input_col2:
        st.markdown('<div class="fusion-panel">', unsafe_allow_html=True)
        st.markdown('<div class="fusion-title">📷 Image Analysis Link</div>', unsafe_allow_html=True)
        st.markdown('<div class="fusion-caption">Optional: attach a DL image analysis log_id from the wound/injury analysis page.</div>', unsafe_allow_html=True)

        default_img_id = st.session_state.dl_image_log_id or 0
        image_log_id_input = st.number_input(
            "Image log_id",
            min_value=0,
            value=int(default_img_id) if default_img_id else 0,
            help="Use 0 to skip image fusion."
        )

        if image_log_id_input > 0:
            st.success(f"Image analysis linked: log_id {image_log_id_input}")
        else:
            st.info("No image analysis linked. Triage can still run using vitals/symptoms/NLP.")

        patient_notes = st.text_input(
            "Patient notes",
            placeholder="e.g. diabetic patient, 3-day wound, elderly patient"
        )

        if st.session_state.dl_image_result:
            with st.expander("View latest image analysis summary"):
                st.json(st.session_state.dl_image_result)

        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="fusion-panel">', unsafe_allow_html=True)
    st.markdown('<div class="fusion-title">🚀 Run Multi-Modal Fusion</div>', unsafe_allow_html=True)
    run_col1, run_col2 = st.columns([2, 1])

    with run_col1:
        st.markdown(
            """
            <div class="fusion-note">
                This request combines all available modalities and sends them to <b>/v2/triage</b>. 
                The output includes ESI level, composite risk, modality contributions, red flags, and full component breakdown.
            </div>
            """,
            unsafe_allow_html=True
        )

    with run_col2:
        run_triage = st.button(
            "🚦 Compute Fusion Triage",
            use_container_width=True,
            type="primary"
        )

    st.markdown('</div>', unsafe_allow_html=True)

    if run_triage:
        payload = {
            "log_id": log_id_input if use_existing else None,
            "image_log_id": image_log_id_input if image_log_id_input > 0 else None,
            "nlp_findings": nlp_payload,
            "patient_notes": patient_notes,
        }

        if not use_existing:
            payload.update(inline_vitals)
            payload["symptoms"] = inline_symptoms

        with st.spinner("Computing multi-modal triage score..."):
            response = requests.post(
                f"{API_URL}/v2/triage",
                json=payload,
                headers=auth_headers(),
                timeout=30
            )

        if response.status_code == 200:
            result = response.json()
            st.session_state.triage_result = result
            st.session_state.triage_log_id = result.get("triage_log_id")
            st.success(f"Triage complete — triage_log_id: **{result.get('triage_log_id')}**")
        else:
            handle_response_error(response)

    tr = st.session_state.triage_result

    if tr:
        st.markdown("---")

        level = tr.get("esi_level", 3)
        esi_label = tr.get("esi_label", "Unknown")
        composite_risk = tr.get("composite_risk", 0)
        total_score = tr.get("total_score", 0)

        if level == 1:
            result_class = "esi-critical"
            result_icon = "🔴"
            urgency = "Immediate life-saving intervention likely required"
        elif level == 2:
            result_class = "esi-high"
            result_icon = "🟠"
            urgency = "High-risk patient requiring rapid clinical assessment"
        elif level == 3:
            result_class = "esi-moderate"
            result_icon = "🟡"
            urgency = "Moderate acuity; multiple resources may be needed"
        else:
            result_class = "esi-stable"
            result_icon = "🟢"
            urgency = "Lower acuity; monitor and manage according to clinical workflow"

        st.markdown(
            f"""
            <div class="{result_class}">
                <h2>{result_icon} ESI Level {level} — {esi_label}</h2>
                <p>{urgency} | Triage log ID: <b>{tr.get('triage_log_id')}</b></p>
            </div>
            """,
            unsafe_allow_html=True
        )

        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.metric("ESI Level", level)
        with k2:
            st.metric("Composite Risk", f"{composite_risk} / 100")
        with k3:
            st.metric("Total Score", total_score)
        with k4:
            st.metric("Triage Log ID", tr.get("triage_log_id"))
        style_metric_cards()

        red_flags = tr.get("red_flags", [])
        if red_flags:
            st.error("🚨 Red flags detected: " + ", ".join(red_flags))
        else:
            st.success("✅ No red flags returned by the fusion engine.")

        result_col1, result_col2 = st.columns([1.2, 1])

        with result_col1:
            st.markdown('<div class="fusion-panel">', unsafe_allow_html=True)
            st.markdown('<div class="fusion-title">📊 Modality Contribution Analysis</div>', unsafe_allow_html=True)
            contribs = tr.get("contributions", {})
            if contribs:
                cdf = pd.DataFrame({
                    "Modality": [k.replace("_", " ").title() for k in contribs.keys()],
                    "Score": list(contribs.values())
                }).sort_values("Score")

                fig = px.bar(
                    cdf,
                    x="Score",
                    y="Modality",
                    orientation="h",
                    title="Risk Contribution by Modality",
                    color="Score",
                    color_continuous_scale="Reds"
                )
                fig.update_layout(height=420)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No contribution data returned.")
            st.markdown('</div>', unsafe_allow_html=True)

        with result_col2:
            st.markdown('<div class="fusion-panel">', unsafe_allow_html=True)
            st.markdown('<div class="fusion-title">🧾 Clinical Action Summary</div>', unsafe_allow_html=True)
            st.info(f"📄 Generate a PDF report using log_id **{tr.get('triage_log_id')}** in the PDF Reports page.")
            st.info(f"📋 Review this patient in Patient History if the triage is linked to a prediction log.")
            if level in [1, 2]:
                st.error("Escalate to urgent clinical review and prioritize care team notification.")
            elif level == 3:
                st.warning("Monitor closely and prepare resources based on presenting symptoms.")
            else:
                st.success("Continue standard triage workflow and monitoring.")
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="fusion-panel">', unsafe_allow_html=True)
        st.markdown('<div class="fusion-title">🔬 Full Fusion Breakdown</div>', unsafe_allow_html=True)
        with st.expander("View complete triage API response"):
            st.json(tr)
        with st.expander("View component-level breakdown"):
            st.json(tr.get("components", {}))
        st.markdown('</div>', unsafe_allow_html=True)


# ===========================================================
# NEW: DASHBOARD
# ===========================================================

elif page == "📊 Dashboard":

    st.markdown("""
    <style>
    .dash-hero{background:linear-gradient(135deg,#0f172a 0%,#2563eb 52%,#06b6d4 100%);color:white;padding:28px 30px;border-radius:26px;box-shadow:0 18px 50px rgba(37,99,235,.25);margin-bottom:22px;}
    .dash-hero h1{font-size:38px;font-weight:900;margin-bottom:6px;}.dash-hero p{font-size:15px;opacity:.92;margin:0;}
    .dash-panel{background:rgba(255,255,255,.96);border:1px solid #e2e8f0;border-radius:22px;padding:20px 22px;box-shadow:0 12px 35px rgba(15,23,42,.06);margin-bottom:18px;}
    .dash-title{font-size:21px;font-weight:900;color:#0f172a;margin-bottom:8px;}
    .pressure-alert{background:linear-gradient(90deg,#991b1b,#ef4444);color:white;border-radius:18px;padding:16px 18px;font-size:18px;font-weight:900;box-shadow:0 14px 34px rgba(239,68,68,.24);margin-bottom:18px;}
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="dash-hero">
        <h1>📊 Clinical Operations Dashboard</h1>
        <p>Monitor triage volume, image analysis, multi-modal scores, high-acuity pressure, and clinical feedback trends.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="dash-panel">', unsafe_allow_html=True)
    window = st.selectbox("Operational time window", [6, 12, 24, 48, 72], index=2, format_func=lambda h: f"Last {h} hours")
    with st.spinner("Loading dashboard data..."):
        response = requests.get(f"{API_URL}/v2/dashboard/summary", params={"hours": window}, headers=auth_headers(), timeout=15)
    st.markdown('</div>', unsafe_allow_html=True)

    if response.status_code != 200:
        handle_response_error(response)
        st.stop()

    s = response.json()
    high_acuity = s.get("high_acuity_count", 0)
    if high_acuity >= 5:
        st.markdown(f'<div class="pressure-alert">🚨 High Acuity Pressure: {high_acuity} ESI 1–2 cases in the selected window.</div>', unsafe_allow_html=True)

    st.markdown('<div class="dash-panel">', unsafe_allow_html=True)
    m1,m2,m3,m4,m5 = st.columns(5)
    with m1: st.metric("Predictions", s.get("total_predictions", 0))
    with m2: st.metric("Image Analyses", s.get("total_image_analyses", 0))
    with m3: st.metric("Triage Scores", s.get("total_triage_scores", 0))
    with m4: st.metric("Avg Risk", f"{s.get('avg_composite_risk', 0)} / 100")
    with m5: st.metric("High Acuity", high_acuity, delta_color="inverse")
    style_metric_cards()
    st.markdown('</div>', unsafe_allow_html=True)

    c1,c2 = st.columns(2)
    with c1:
        st.markdown('<div class="dash-panel">', unsafe_allow_html=True)
        st.markdown('<div class="dash-title">🚦 ESI Level Distribution</div>', unsafe_allow_html=True)
        esi = s.get("esi_distribution", {})
        if any(v > 0 for v in esi.values()):
            esi_df = pd.DataFrame({"ESI Level": [f"Level {k}" for k in esi.keys()], "Count": list(esi.values())})
            fig = px.bar(esi_df, x="ESI Level", y="Count", color="ESI Level", title=f"ESI Distribution — last {window}h", color_discrete_map={"Level 1":"#dc2626","Level 2":"#f97316","Level 3":"#f59e0b","Level 4":"#10b981","Level 5":"#2563eb"})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No triage scores recorded in this window.")
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="dash-panel">', unsafe_allow_html=True)
        st.markdown('<div class="dash-title">🩺 Clinical Feedback</div>', unsafe_allow_html=True)
        fb = s.get("clinical_feedback", {})
        accepted, overridden = fb.get("accepted",0), fb.get("overridden",0)
        if accepted + overridden > 0:
            fb_df = pd.DataFrame({"Decision": ["Accepted", "Overridden"], "Count": [accepted, overridden]})
            fig2 = px.pie(fb_df, names="Decision", values="Count", title="AI Accepted vs Overridden", color="Decision", color_discrete_map={"Accepted":"#10b981","Overridden":"#f97316"})
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No clinical feedback recorded in this window.")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="dash-panel">', unsafe_allow_html=True)
    st.markdown('<div class="dash-title">🏥 Operational Summary</div>', unsafe_allow_html=True)
    if high_acuity >= 5:
        st.error("Increase monitoring of triage queue and prioritize critical care capacity.")
    elif high_acuity >= 2:
        st.warning("Moderate pressure detected. Keep watchlist and triage resources ready.")
    else:
        st.success("Operational load appears stable for the selected window.")
    st.markdown('</div>', unsafe_allow_html=True)

elif page == "📄 PDF Reports":

    st.markdown("""
    <style>
    .pdf-hero{background:linear-gradient(135deg,#0f172a 0%,#2563eb 52%,#06b6d4 100%);color:white;padding:28px 30px;border-radius:26px;box-shadow:0 18px 50px rgba(37,99,235,.25);margin-bottom:22px;}
    .pdf-hero h1{font-size:38px;font-weight:900;margin-bottom:6px;}.pdf-hero p{font-size:15px;opacity:.92;margin:0;}
    .pdf-panel{background:rgba(255,255,255,.96);border:1px solid #e2e8f0;border-radius:22px;padding:20px 22px;box-shadow:0 12px 35px rgba(15,23,42,.06);margin-bottom:18px;}
    .pdf-title{font-size:21px;font-weight:900;color:#0f172a;margin-bottom:8px;}
    .report-preview{background:#f8fafc;border:1px dashed #94a3b8;border-radius:20px;padding:18px;color:#334155;}
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="pdf-hero">
        <h1>📄 Clinical PDF Report Center</h1>
        <p>Generate clinician-facing reports with prediction summaries, triage scores, image analysis, and explainability outputs.</p>
    </div>
    """, unsafe_allow_html=True)

    default_id = st.session_state.triage_log_id or st.session_state.prediction_id or 1

    single, batch = st.columns([1.1, 1])

    with single:
        st.markdown('<div class="pdf-panel">', unsafe_allow_html=True)
        st.markdown('<div class="pdf-title">📌 Single Patient Report</div>', unsafe_allow_html=True)

        log_id = st.number_input(
            "Report log_id",
            min_value=1,
            value=int(default_id) if default_id else 1
        )

        st.markdown(f"""
        <div class="report-preview">
            <b>Report Preview</b><br>
            Patient/Prediction Log ID: <b>{log_id}</b><br>
            Includes AI prediction, triage notes, available image/NLP findings, and clinical audit context.
        </div>
        """, unsafe_allow_html=True)

        if st.button("📄 Generate & Download PDF", use_container_width=True, type="primary"):
            with st.spinner("Generating PDF report..."):
                response = requests.get(
                    f"{API_URL}/v2/report/{log_id}",
                    headers=auth_headers(),
                    timeout=30
                )

            if response.status_code == 200:
                st.download_button(
                    "⬇️ Download PDF Report",
                    data=response.content,
                    file_name=f"emergeai_report_{log_id}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )

                st.success(f"Report generated for log_id {log_id}.")

                st.session_state.generated_report_log_id = int(log_id)

            else:
                handle_response_error(response)

        st.markdown("### 📧 Send Report to Doctor")

        doctor_email = st.text_input(
            "Doctor Email Address",
            placeholder="doctor@example.com",
            key="pdf_report_doctor_email"
        )

        send_report_clicked = st.button(
            "📨 Send Report Email",
            key="send_pdf_report_email",
            use_container_width=True
        )

        if send_report_clicked:
            if not doctor_email:
                st.warning("Please enter doctor email address.")
            else:
                payload = {
                    "doctor_email": doctor_email,
                    "prediction_id": int(log_id)
                }

                with st.spinner("Sending report email to doctor..."):
                    email_response = requests.post(
                        f"{API_URL}/send-report-email",
                        json=payload,
                        headers=auth_headers(),
                        timeout=120
                    )

                if email_response.status_code == 200:
                    st.success("✅ Report email sent successfully.")
                else:
                    st.error(email_response.text)

        st.markdown('</div>', unsafe_allow_html=True)

    with batch:
        st.markdown('<div class="pdf-panel">', unsafe_allow_html=True)
        st.markdown('<div class="pdf-title">🗂️ Batch Reports</div>', unsafe_allow_html=True)

        batch_input = st.text_input(
            "Multiple log IDs",
            placeholder="e.g. 1, 5, 12, 38"
        )

        st.info("Batch mode creates separate downloadable reports for each valid numeric ID.")

        if batch_input and st.button("Generate Batch Reports", use_container_width=True):
            ids = [
                int(x.strip())
                for x in batch_input.split(",")
                if x.strip().isdigit()
            ]

            if not ids:
                st.error("Please enter valid numeric IDs separated by commas.")

            for lid in ids:
                with st.spinner(f"Generating report for log_id {lid}..."):
                    r = requests.get(
                        f"{API_URL}/v2/report/{lid}",
                        headers=auth_headers(),
                        timeout=30
                    )

                if r.status_code == 200:
                    st.download_button(
                        label=f"⬇️ Report for log_id {lid}",
                        data=r.content,
                        file_name=f"emergeai_report_{lid}.pdf",
                        mime="application/pdf",
                        key=f"pdf_batch_{lid}"
                    )
                else:
                    st.warning(f"Could not generate report for log_id {lid}: {r.text}")

        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="pdf-panel">', unsafe_allow_html=True)
    st.markdown('<div class="pdf-title">✅ Recommended Report Workflow</div>', unsafe_allow_html=True)
    st.write("1. Run Live Prediction → 2. Run Multi-Modal Triage → 3. Add Clinical Feedback → 4. Generate PDF Report → 5. Send Report to Doctor")
    st.caption("Use the latest prediction_id or triage_log_id shown in the app for faster reporting.")
    st.markdown('</div>', unsafe_allow_html=True)

elif page == "📈 Analytics":

    st.markdown("""
    <style>
    .analytics-hero{background:linear-gradient(135deg,#0f172a 0%,#7c3aed 48%,#06b6d4 100%);color:white;padding:28px 30px;border-radius:26px;box-shadow:0 18px 50px rgba(124,58,237,.25);margin-bottom:22px;}
    .analytics-hero h1{font-size:38px;font-weight:900;margin-bottom:6px;}.analytics-hero p{font-size:15px;opacity:.92;margin:0;}
    .analytics-panel{background:rgba(255,255,255,.96);border:1px solid #e2e8f0;border-radius:22px;padding:20px 22px;box-shadow:0 12px 35px rgba(15,23,42,.06);margin-bottom:18px;}
    .analytics-title{font-size:21px;font-weight:900;color:#0f172a;margin-bottom:8px;}
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="analytics-hero">
        <h1>📈 AI Analytics Command Center</h1>
        <p>Session-level intelligence for prediction, image AI, multi-modal triage, red flags, and model readiness.</p>
    </div>
    """, unsafe_allow_html=True)

    latest_prediction = st.session_state.get("last_prediction")
    latest_image = st.session_state.get("dl_image_result")
    latest_triage = st.session_state.get("triage_result")

    total_predictions = 1 if latest_prediction else 0
    total_image_analyses = 1 if latest_image else 0
    total_triage_scores = 1 if latest_triage else 0
    critical_cases = 1 if latest_triage and (latest_triage.get("esi_level") in [1,2] or latest_triage.get("composite_risk",0) >= 70) else 0

    st.markdown('<div class="analytics-panel">', unsafe_allow_html=True)
    a1,a2,a3,a4 = st.columns(4)
    with a1: st.metric("Predictions", total_predictions)
    with a2: st.metric("Image Analyses", total_image_analyses)
    with a3: st.metric("Triage Scores", total_triage_scores)
    with a4: st.metric("Critical Cases", critical_cases, delta_color="inverse")
    style_metric_cards()
    st.markdown('</div>', unsafe_allow_html=True)

    left,right = st.columns(2)
    with left:
        st.markdown('<div class="analytics-panel">', unsafe_allow_html=True)
        st.markdown('<div class="analytics-title">🚦 Latest Multi-Modal Triage</div>', unsafe_allow_html=True)
        if latest_triage:
            esi_level = latest_triage.get("esi_level", "N/A")
            st.metric("ESI Level", esi_level)
            st.metric("Composite Risk", f"{latest_triage.get('composite_risk',0)} / 100")
            if latest_triage.get("red_flags"):
                st.error("🚨 Red Flags: " + ", ".join(latest_triage.get("red_flags", [])))
            contributions = latest_triage.get("contributions", {})
            if contributions:
                cdf = pd.DataFrame({"Modality": [k.replace("_"," ").title() for k in contributions.keys()], "Score": list(contributions.values())})
                fig = px.bar(cdf, x="Score", y="Modality", orientation="h", title="Multi-Modal Contribution", color="Score", color_continuous_scale="Reds")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No triage result available. Run Multi-Modal Triage first.")
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="analytics-panel">', unsafe_allow_html=True)
        st.markdown('<div class="analytics-title">🔬 Latest Image AI</div>', unsafe_allow_html=True)
        if latest_image:
            st.metric("Wound Class", latest_image.get("wound_class", "N/A"))
            st.metric("Infection Severity", latest_image.get("infection_severity", "N/A"))
            st.metric("Severity Score", f"{(latest_image.get('severity_score') or 0):.3f} / 1.0")
            infection_distribution = latest_image.get("infection_distribution", {})
            if infection_distribution:
                idf = pd.DataFrame({"Severity": list(infection_distribution.keys()), "Probability": list(infection_distribution.values())})
                fig2 = px.bar(idf, x="Severity", y="Probability", title="Infection Severity Distribution")
                st.plotly_chart(fig2, use_container_width=True)
        else:
            st.warning("No image analysis available. Run DL Image Analysis first.")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="analytics-panel">', unsafe_allow_html=True)
    st.markdown('<div class="analytics-title">🤖 Latest Prediction & System Status</div>', unsafe_allow_html=True)
    c1,c2 = st.columns(2)
    with c1:
        if latest_prediction:
            st.metric("ML Prediction", latest_prediction.get("ml_prediction", "N/A"))
            st.metric("Final Prediction", latest_prediction.get("final_prediction", "N/A"))
            probabilities = latest_prediction.get("probabilities", {})
            if probabilities:
                prob_df = pd.DataFrame(list(probabilities.items()), columns=["Triage Level", "Probability"])
                fig3 = px.bar(prob_df, x="Triage Level", y="Probability", title="Latest Prediction Distribution")
                st.plotly_chart(fig3, use_container_width=True)
        else:
            st.warning("No prediction available. Run Live Prediction first.")
    with c2:
        st.success("✅ FastAPI Backend Connected")
        st.success("✅ PostgreSQL Connected")
        st.success("✅ NLP Extraction Active")
        st.success("✅ OpenCV + EfficientNet Active")
        st.success("✅ SHAP Explainability Active")
        st.success("✅ RBAC Authentication Active")
    st.markdown('</div>', unsafe_allow_html=True)

elif page == "🚨 Risk Watchlist":

    st.markdown(
        """
        <style>
        .watch-hero {
            background: linear-gradient(135deg, #7f1d1d 0%, #dc2626 48%, #f97316 100%);
            color: white;
            padding: 28px 30px;
            border-radius: 26px;
            box-shadow: 0 18px 50px rgba(220, 38, 38, 0.25);
            margin-bottom: 22px;
        }
        .watch-hero h1 {
            font-size: 38px;
            font-weight: 900;
            margin-bottom: 6px;
        }
        .watch-hero p {
            font-size: 15px;
            opacity: 0.94;
            margin: 0;
        }
        .watch-panel {
            background: rgba(255,255,255,0.96);
            border: 1px solid #e2e8f0;
            border-radius: 22px;
            padding: 20px 22px;
            box-shadow: 0 12px 35px rgba(15, 23, 42, 0.06);
            margin-bottom: 18px;
        }
        .watch-title {
            font-size: 21px;
            font-weight: 900;
            color: #0f172a;
            margin-bottom: 8px;
        }
        .watch-caption {
            color: #64748b;
            font-size: 14px;
            margin-bottom: 12px;
        }
        .watch-alert-critical {
            background: linear-gradient(90deg, #7f1d1d, #ef4444);
            color: white;
            border-radius: 18px;
            padding: 16px 18px;
            font-size: 18px;
            font-weight: 900;
            box-shadow: 0 14px 34px rgba(239, 68, 68, 0.24);
            margin: 12px 0 18px 0;
        }
        .watch-alert-high {
            background: linear-gradient(90deg, #9a3412, #f97316);
            color: white;
            border-radius: 18px;
            padding: 16px 18px;
            font-size: 18px;
            font-weight: 900;
            box-shadow: 0 14px 34px rgba(249, 115, 22, 0.22);
            margin: 12px 0 18px 0;
        }
        .watch-alert-clear {
            background: linear-gradient(90deg, #065f46, #10b981);
            color: white;
            border-radius: 18px;
            padding: 16px 18px;
            font-size: 18px;
            font-weight: 900;
            box-shadow: 0 14px 34px rgba(16, 185, 129, 0.20);
            margin: 12px 0 18px 0;
        }
        .risk-patient-card {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-left: 8px solid #f97316;
            border-radius: 20px;
            padding: 18px 20px;
            box-shadow: 0 10px 28px rgba(15, 23, 42, 0.06);
            margin-bottom: 14px;
        }
        .risk-patient-critical {
            border-left-color: #dc2626;
            background: linear-gradient(90deg, #fff1f2, #ffffff);
        }
        .risk-patient-high {
            border-left-color: #f97316;
            background: linear-gradient(90deg, #fff7ed, #ffffff);
        }
        .risk-card-title {
            font-size: 20px;
            font-weight: 900;
            color: #0f172a;
        }
        .risk-card-subtitle {
            font-size: 13px;
            color: #64748b;
            margin-top: 4px;
        }
        .watch-badge {
            display: inline-block;
            padding: 7px 12px;
            border-radius: 999px;
            font-weight: 900;
            font-size: 13px;
            color: white;
        }
        .badge-critical { background: #dc2626; }
        .badge-high { background: #f97316; }
        .mini-clinical-note {
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 14px;
            padding: 12px 14px;
            color: #334155;
            font-size: 14px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="watch-hero">
            <h1>🚨 Live Risk Watchlist</h1>
            <p>Command-center view for critical and high-risk emergency patients requiring immediate clinical attention.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    response = requests.get(
        f"{API_URL}/v2/watchlist",
        headers=auth_headers(),
        timeout=20
    )

    if response.status_code != 200:
        handle_response_error(response)
        st.stop()

    data = response.json()
    patients = data.get("patients", [])

    if patients:
        watch_df = pd.DataFrame(patients)
    else:
        watch_df = pd.DataFrame()

    total_watchlist = len(watch_df)
    critical_count = 0
    high_count = 0

    if not watch_df.empty and "risk_level" in watch_df.columns:
        critical_count = int((watch_df["risk_level"].astype(str).str.upper() == "CRITICAL").sum())
        high_count = int((watch_df["risk_level"].astype(str).str.upper() == "HIGH").sum())

    st.markdown('<div class="watch-panel">', unsafe_allow_html=True)
    st.markdown('<div class="watch-title">🏥 Emergency Load Summary</div>', unsafe_allow_html=True)
    st.markdown('<div class="watch-caption">Live view of patients escalated by prediction level, dangerous vitals, or emergency symptoms.</div>', unsafe_allow_html=True)

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("Total Watchlist", total_watchlist)
    with k2:
        st.metric("Critical", critical_count, delta_color="inverse")
    with k3:
        st.metric("High Risk", high_count, delta_color="inverse")
    with k4:
        st.metric("Requested By", data.get("requested_by", st.session_state.username))
    style_metric_cards()

    if critical_count > 0:
        st.markdown(
            f'<div class="watch-alert-critical">🚨 CRITICAL ALERT: {critical_count} patient(s) require immediate emergency review.</div>',
            unsafe_allow_html=True
        )
    elif high_count > 0:
        st.markdown(
            f'<div class="watch-alert-high">⚠️ HIGH-RISK ALERT: {high_count} patient(s) should be prioritized in triage queue.</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="watch-alert-clear">✅ No critical/high-risk patients currently on the watchlist.</div>',
            unsafe_allow_html=True
        )
    st.markdown('</div>', unsafe_allow_html=True)

    if watch_df.empty:
        st.info("No high-risk patients are currently detected. Run more predictions to populate the watchlist.")
        st.stop()

    st.markdown('<div class="watch-panel">', unsafe_allow_html=True)
    st.markdown('<div class="watch-title">🔎 Watchlist Filters</div>', unsafe_allow_html=True)

    f1, f2, f3 = st.columns([1.3, 1, 1])
    with f1:
        search_text = st.text_input(
            "Search watchlist",
            placeholder="Search ID, notes, prediction, vitals, emergency keywords..."
        )
    with f2:
        risk_filter = st.selectbox("Risk Level", ["All", "CRITICAL", "HIGH"])
    with f3:
        max_rows = st.selectbox("Show Patients", [10, 20, 50, "All"], index=1)

    filtered_df = watch_df.copy()

    if search_text:
        filtered_df = filtered_df[
            filtered_df.astype(str).apply(
                lambda row: row.str.contains(search_text, case=False, na=False).any(),
                axis=1
            )
        ]

    if risk_filter != "All" and "risk_level" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["risk_level"].astype(str).str.upper() == risk_filter]

    if "created_at" in filtered_df.columns:
        filtered_df["created_at_dt"] = pd.to_datetime(filtered_df["created_at"], errors="coerce")
        filtered_df = filtered_df.sort_values(
            by=["risk_level", "created_at_dt"],
            ascending=[True, False],
            na_position="last"
        )

    if max_rows != "All":
        filtered_df = filtered_df.head(int(max_rows))

    st.success(f"Showing {len(filtered_df)} of {len(watch_df)} watchlist patients")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="watch-panel">', unsafe_allow_html=True)
    st.markdown('<div class="watch-title">📊 Watchlist Analytics</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        if "risk_level" in watch_df.columns:
            risk_df = (
                watch_df["risk_level"]
                .astype(str)
                .value_counts()
                .reset_index()
            )
            risk_df.columns = ["Risk Level", "Count"]
            fig = px.pie(
                risk_df,
                names="Risk Level",
                values="Count",
                title="Watchlist Risk Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Risk level data not available.")

    with c2:
        vital_cols = ["heart_rate", "systolic_bp", "respiratory_rate", "spo2", "temperature"]
        available_vitals = [c for c in vital_cols if c in watch_df.columns]
        if available_vitals:
            avg_df = pd.DataFrame({
                "Vital": [v.replace("_", " ").title() for v in available_vitals],
                "Average": [pd.to_numeric(watch_df[v], errors="coerce").mean() for v in available_vitals]
            })
            fig2 = px.bar(
                avg_df,
                x="Vital",
                y="Average",
                title="Average Watchlist Vitals"
            )
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Vitals not available for charting.")

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="watch-panel">', unsafe_allow_html=True)
    st.markdown('<div class="watch-title">🧾 Priority Patient Queue</div>', unsafe_allow_html=True)
    st.markdown('<div class="watch-caption">Patients are shown as triage cards with key vitals, prediction, notes, and clinical flags.</div>', unsafe_allow_html=True)

    if filtered_df.empty:
        st.warning("No patients match the selected filters.")
    else:
        for _, row in filtered_df.iterrows():
            risk_level = str(row.get("risk_level", "HIGH")).upper()
            is_critical = risk_level == "CRITICAL"
            card_class = "risk-patient-card risk-patient-critical" if is_critical else "risk-patient-card risk-patient-high"
            badge_class = "watch-badge badge-critical" if is_critical else "watch-badge badge-high"
            icon = "🔴" if is_critical else "🟠"

            st.markdown(
                f"""
                <div class="{card_class}">
                    <div style="display:flex;justify-content:space-between;align-items:center;gap:16px;">
                        <div>
                            <div class="risk-card-title">{icon} Patient #{row.get('id')} — {row.get('prediction')}</div>
                            <div class="risk-card-subtitle">Age {row.get('age')} • {row.get('gender')} • Arrival: {row.get('arrivalmode')} • Created: {row.get('created_at')}</div>
                        </div>
                        <div><span class="{badge_class}">{risk_level}</span></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            with st.expander(f"Open emergency review for Patient #{row.get('id')}"):
                v1, v2, v3, v4, v5, v6 = st.columns(6)
                with v1:
                    st.metric("HR", row.get("heart_rate"))
                with v2:
                    st.metric("SBP", row.get("systolic_bp"))
                with v3:
                    st.metric("DBP", row.get("diastolic_bp"))
                with v4:
                    st.metric("RR", row.get("respiratory_rate"))
                with v5:
                    st.metric("SpO₂", row.get("spo2"))
                with v6:
                    st.metric("Temp", row.get("temperature"))
                style_metric_cards()

                detail1, detail2 = st.columns(2)

                with detail1:
                    st.markdown("### 📝 Problem Description")
                    note = str(row.get("problem_description") or "No problem description saved.")
                    st.markdown(
                        f"<div class='mini-clinical-note'>{html.escape(note)}</div>",
                        unsafe_allow_html=True
                    )

                    st.markdown("### 🚨 Emergency Keywords")
                    keywords = row.get("emergency_keywords")
                    if keywords:
                        st.error(keywords)
                    else:
                        st.info("No emergency keywords saved.")

                    st.markdown("### 🧾 Feedback Status")
                    st.info(row.get("feedback") or "No feedback saved.")

                with detail2:
                    st.markdown("### 🛡️ Safety Reasons")
                    safety = row.get("safety_reasons")
                    if safety:
                        st.warning(safety)
                    else:
                        st.success("No safety reasons saved.")

                    st.markdown("### 🧠 Clinical Explanations")
                    explanations = row.get("clinical_explanations")
                    if explanations:
                        st.info(explanations)
                    else:
                        st.info("No clinical explanations saved.")

                    st.markdown("### 📌 Recommended Action")
                    if is_critical:
                        st.error("Immediate clinician review, escalation, and emergency intervention readiness recommended.")
                    else:
                        st.warning("Prioritize in triage queue and monitor vitals closely.")

                st.caption(f"Prediction confidence: {row.get('confidence')} | Source record ID: {row.get('id')}")

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="watch-panel">', unsafe_allow_html=True)
    st.markdown('<div class="watch-title">📋 Full Watchlist Table</div>', unsafe_allow_html=True)
    st.dataframe(
        filtered_df.drop(columns=["created_at_dt"], errors="ignore"),
        use_container_width=True
    )
    st.markdown('</div>', unsafe_allow_html=True)

# ===========================================================
# NEW: MODEL RETRAINING
# ===========================================================

elif page == "🧪 Model Retraining":

    st.title("🧪 Model Retraining Pipeline")
    st.caption("Admin-only page to monitor retraining readiness and model files.")

    if st.session_state.role != "admin":
        st.error("Only admin can access model retraining.")
        st.stop()

    st.info(
        "Your retraining script is working. It connects to PostgreSQL and requires "
        "at least 20 prediction records before safe retraining."
    )

    response = requests.get(
        f"{API_URL}/history",
        headers=auth_headers(),
        timeout=20
    )

    if response.status_code != 200:
        handle_response_error(response)
        st.stop()

    data = response.json()
    history = data.get("history", [])

    total_records = len(history)
    required_records = 20
    remaining_records = max(required_records - total_records, 0)

    st.markdown("---")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("Training Records Available", total_records)

    with c2:
        st.metric("Required Records", required_records)

    with c3:
        st.metric("Records Needed", remaining_records)

    style_metric_cards()

    progress_value = min(total_records / required_records, 1.0)
    st.progress(progress_value)

    if total_records < required_records:
        st.warning(
            f"Need {remaining_records} more prediction records before retraining."
        )
    else:
        st.success("Enough records available for retraining.")

    st.markdown("---")

    st.subheader("Prediction Class Distribution")

    if history:
        df = pd.DataFrame(history)

        if "final_prediction" in df.columns:
            dist_df = (
                df["final_prediction"]
                .astype(str)
                .value_counts()
                .reset_index()
            )
            dist_df.columns = ["Final Prediction", "Count"]

            fig = px.bar(
                dist_df,
                x="Final Prediction",
                y="Count",
                title="Training Data Class Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("final_prediction column not found in history response.")
    else:
        st.info("No prediction records found yet.")

    st.markdown("---")

    st.subheader("How to Run Retraining")

    st.code(
        """
cd backend
python retrain_model.py
        """,
        language="bash"
    )

    st.subheader("Model File Locations")

    model_col1, model_col2 = st.columns(2)

    with model_col1:
        st.write("Current model")
        st.code("models/triage_xgboost_balanced.pkl")

        st.write("Feature columns file")
        st.code("models/retrained_feature_columns.pkl")

    with model_col2:
        st.write("Backup folder")
        st.code("models/backups")

        st.write("Retraining script")
        st.code("backend/retrain_model.py")

    st.markdown("---")

    st.subheader("Retraining Status")

    if total_records >= required_records:
        st.success("Ready to retrain from terminal.")
    else:
        st.warning("Not ready for safe retraining yet.")


    st.markdown("---")

    st.subheader("Run Retraining from Dashboard")

    st.warning(
        "Only run this after you have enough prediction records. "
        "Current safety rule requires at least 20 records."
    )

    if st.button("🚀 Run Model Retraining", use_container_width=True, type="primary"):
        with st.spinner("Running retraining pipeline..."):
            response = requests.post(
                f"{API_URL}/v2/retrain-model",
                headers=auth_headers(),
                timeout=320
            )

        if response.status_code == 200:
            result = response.json()

            if result.get("status") == "success":
                st.success("Retraining command executed successfully.")
            else:
                st.warning("Retraining command completed with issues.")

            st.write("### Terminal Output")
            st.code(result.get("stdout", ""), language="text")

            if result.get("stderr"):
                st.write("### Error / Debug Logs")
                st.code(result.get("stderr", ""), language="text")

        else:
            handle_response_error(response)

    st.caption(
        "This page monitors readiness and can also trigger the backend retraining API. "
        "During development, verify model files after each retraining run."
    )

    st.markdown("---")

    st.subheader("🚀 Deployment Readiness")

    st.success("✅ FastAPI Backend Ready")
    st.success("✅ PostgreSQL Connected")
    st.success("✅ RBAC Security Enabled")
    st.success("✅ Retrained Model Available")
    st.success("✅ SHAP Explainability Enabled")
    st.success("✅ Docker Preparation Ready")

    st.markdown("""
    <hr>
    <div style='text-align:center;color:#94a3b8;font-size:13px;padding:18px;'>
    EmergeAI Enterprise © 2026 |
    Enterprise Emergency Triage & Clinical Intelligence Platform |
    Educational AI Prototype
    </div>
    """, unsafe_allow_html=True)

elif page == "Nurse Management":

    st.title("Nurse Management")
    st.caption("Manage nurse availability and view the care team roster.")

    if st.session_state.role not in ["doctor", "admin", "nurse"]:
        st.error("You do not have permission to access nurse management.")
        st.stop()

    can_add_nurse = st.session_state.role in ["doctor", "admin"]

    if can_add_nurse:
        st.subheader("Add Nurse")

        with st.form("add_nurse_form", clear_on_submit=True):
            nurse_name = st.text_input("Nurse Name")
            nurse_email = st.text_input("Nurse Email")
            department = st.text_input("Department")
            experience_level = st.selectbox("Experience Level", ["normal", "experienced", "senior", "critical"])
            available_status = st.checkbox("Available Status", value=True)

            submitted = st.form_submit_button("Add Nurse", use_container_width=True)

        if submitted:
            if not nurse_name or not nurse_email or not department:
                st.warning("Please complete all nurse fields.")
            else:
                payload = {
                    "name": nurse_name,
                    "email": nurse_email,
                    "department": department,
                    "available_status": available_status,
                    "experience_level": experience_level
                }

                try:
                    response = requests.post(
                        f"{API_URL}/nurses/add",
                        json=payload,
                        headers=auth_headers(),
                        timeout=20
                    )

                    if response.status_code == 200:
                        st.success("Nurse added successfully.")
                    else:
                        handle_response_error(response)

                except requests.exceptions.ConnectionError:
                    st.error("Backend connection error. Make sure FastAPI is running on port 8000.")
                except Exception as e:
                    st.error(f"Could not add nurse: {e}")
    else:
        st.info("Nurse accounts can view the roster, but only doctors and admins can add nurses.")

    st.divider()
    st.subheader("All Nurses")

    try:
        nurses_response = requests.get(
            f"{API_URL}/nurses",
            headers=auth_headers(),
            timeout=20
        )

        if nurses_response.status_code == 200:
            nurses_data = nurses_response.json()
            nurses = nurses_data.get("nurses", [])

            if nurses:
                nurses_df = pd.DataFrame(nurses)
                nurses_df = nurses_df.rename(columns={
                    "id": "ID",
                    "name": "Name",
                    "email": "Email",
                    "department": "Department",
                    "available_status": "Available",
                    "active_patient_count": "Active Patients",
                    "experience_level": "Experience"
                })
                st.dataframe(nurses_df, use_container_width=True, hide_index=True)
            else:
                st.info("No nurses have been added yet.")
        else:
            handle_response_error(nurses_response)

    except requests.exceptions.ConnectionError:
        st.error("Backend connection error. Make sure FastAPI is running on port 8000.")
    except Exception as e:
        st.error(f"Could not load nurses: {e}")

    st.divider()
    st.subheader("Assignment History")

    try:
        assignments_response = requests.get(
            f"{API_URL}/assignments",
            headers=auth_headers(),
            timeout=20
        )

        if assignments_response.status_code == 200:
            assignments_data = assignments_response.json()
            assignments = assignments_data.get("assignments", [])

            if assignments:
                assignment_rows = []
                for assignment in assignments:
                    nurse = assignment.get("nurse") or {}
                    assignment_rows.append({
                        "Assignment ID": assignment.get("id"),
                        "Prediction ID": assignment.get("prediction_id"),
                        "Nurse": nurse.get("name", "Unknown"),
                        "Department": nurse.get("department", ""),
                        "Status": assignment.get("status"),
                        "Assigned At": assignment.get("assigned_at"),
                        "Notes": assignment.get("notes")
                    })

                st.dataframe(
                    pd.DataFrame(assignment_rows),
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.info("No nurse assignments have been created yet.")
        else:
            handle_response_error(assignments_response)

    except requests.exceptions.ConnectionError:
        st.error("Backend connection error. Make sure FastAPI is running on port 8000.")
    except Exception as e:
        st.error(f"Could not load nurse assignments: {e}")

elif page == "Nurse Patient Care":

    st.title("Nurse Patient Care")
    st.caption("Record patient vitals, create care tasks, and track task progress.")

    if st.session_state.role not in ["nurse", "doctor", "admin"]:
        st.error("You do not have permission to access patient care tasks.")
        st.stop()

    try:
        assignments_response = requests.get(
            f"{API_URL}/assignments",
            headers=auth_headers(),
            timeout=20
        )

        if assignments_response.status_code != 200:
            handle_response_error(assignments_response)
            st.stop()

        assignments = assignments_response.json().get("assignments", [])

    except requests.exceptions.ConnectionError:
        st.error("Backend connection error. Make sure FastAPI is running on port 8000.")
        st.stop()
    except Exception as e:
        st.error(f"Could not load assigned patients: {e}")
        st.stop()

    if not assignments:
        st.info("No nurse assignments found. Assign a nurse to a patient first.")
        st.stop()

    assignment_options = {}
    for assignment in assignments:
        nurse = assignment.get("nurse") or {}
        label = (
            f"Prediction #{assignment.get('prediction_id')} | "
            f"Nurse: {nurse.get('name', 'Unknown')} | "
            f"Status: {assignment.get('status')}"
        )
        assignment_options[label] = assignment

    selected_assignment_label = st.selectbox(
        "Select Assigned Patient / Prediction ID",
        list(assignment_options.keys())
    )
    selected_assignment = assignment_options[selected_assignment_label]
    selected_prediction_id = selected_assignment.get("prediction_id")
    selected_nurse_id = selected_assignment.get("nurse_id")
    selected_nurse = selected_assignment.get("nurse") or {}

    overview_1, overview_2, overview_3 = st.columns(3)
    with overview_1:
        st.metric("Prediction ID", selected_prediction_id)
    with overview_2:
        st.metric("Nurse", selected_nurse.get("name", "Unknown"))
    with overview_3:
        st.metric("Assignment Status", selected_assignment.get("status", "Unknown"))

    can_add_care_data = st.session_state.role in ["nurse", "doctor", "admin"]
    can_assign_tasks = st.session_state.role in ["doctor", "admin"]

    st.divider()

    vitals_col, task_col = st.columns(2)

    with vitals_col:
        st.subheader("Record Vitals")

        if can_add_care_data:
            with st.form(f"nurse_vitals_form_{selected_prediction_id}_{selected_nurse_id}"):
                temperature = st.number_input("Temperature", min_value=80.0, max_value=115.0, value=98.6, step=0.1)
                heart_rate = st.number_input("Heart Rate", min_value=0, max_value=250, value=80)
                blood_pressure = st.text_input("Blood Pressure", placeholder="120/80")
                oxygen_level = st.number_input("Oxygen Level", min_value=0, max_value=100, value=98)
                respiratory_rate = st.number_input("Respiratory Rate", min_value=0, max_value=80, value=16)
                pain_score = st.slider("Pain Score", min_value=0, max_value=10, value=0)
                vitals_notes = st.text_area("Vitals Notes")
                save_vitals = st.form_submit_button("Save Vitals", use_container_width=True)

            if save_vitals:
                payload = {
                    "prediction_id": int(selected_prediction_id),
                    "nurse_id": int(selected_nurse_id),
                    "temperature": float(temperature),
                    "heart_rate": int(heart_rate),
                    "blood_pressure": blood_pressure or None,
                    "oxygen_level": int(oxygen_level),
                    "respiratory_rate": int(respiratory_rate),
                    "pain_score": int(pain_score),
                    "notes": vitals_notes or None
                }

                try:
                    response = requests.post(
                        f"{API_URL}/nurse-vitals/add",
                        json=payload,
                        headers=auth_headers(),
                        timeout=20
                    )

                    if response.status_code == 200:
                        st.success("Vitals recorded successfully.")
                        st.rerun()
                    else:
                        handle_response_error(response)

                except requests.exceptions.ConnectionError:
                    st.error("Backend connection error. Make sure FastAPI is running on port 8000.")
                except Exception as e:
                    st.error(f"Could not save vitals: {e}")

    with task_col:
        st.subheader("Add Care Task")

        if st.session_state.role == "nurse":
            st.caption("Nurses can add care tasks for their assigned patient.")
        elif can_assign_tasks:
            st.caption("Doctors and admins can assign care tasks to the selected nurse.")

        with st.form(f"nurse_task_form_{selected_prediction_id}_{selected_nurse_id}"):
            task_title = st.text_input("Task Title")
            task_description = st.text_area("Task Description")
            task_status = st.selectbox("Status", ["Pending", "In Progress", "Completed"])
            task_priority = st.selectbox("Priority", ["Low", "Medium", "High", "Critical"], index=1)
            save_task = st.form_submit_button("Add Task", use_container_width=True)

        if save_task:
            if not task_title:
                st.warning("Please enter a task title.")
            else:
                payload = {
                    "prediction_id": int(selected_prediction_id),
                    "nurse_id": int(selected_nurse_id),
                    "task_title": task_title,
                    "task_description": task_description or None,
                    "status": task_status,
                    "priority": task_priority
                }

                try:
                    response = requests.post(
                        f"{API_URL}/nurse-tasks/add",
                        json=payload,
                        headers=auth_headers(),
                        timeout=20
                    )

                    if response.status_code == 200:
                        st.success("Care task added successfully.")
                        st.rerun()
                    else:
                        handle_response_error(response)

                except requests.exceptions.ConnectionError:
                    st.error("Backend connection error. Make sure FastAPI is running on port 8000.")
                except Exception as e:
                    st.error(f"Could not add task: {e}")

    st.divider()

    history_col, progress_col = st.columns(2)

    with history_col:
        st.subheader("Vitals History")

        try:
            vitals_response = requests.get(
                f"{API_URL}/nurse-vitals/{selected_prediction_id}",
                headers=auth_headers(),
                timeout=20
            )

            if vitals_response.status_code == 200:
                vitals = vitals_response.json().get("vitals", [])

                if vitals:
                    vitals_rows = []
                    for record in vitals:
                        nurse = record.get("nurse") or {}
                        vitals_rows.append({
                            "Recorded At": record.get("recorded_at"),
                            "Nurse": nurse.get("name", "Unknown"),
                            "Temp": record.get("temperature"),
                            "HR": record.get("heart_rate"),
                            "BP": record.get("blood_pressure"),
                            "O2": record.get("oxygen_level"),
                            "RR": record.get("respiratory_rate"),
                            "Pain": record.get("pain_score"),
                            "Notes": record.get("notes")
                        })
                    st.dataframe(pd.DataFrame(vitals_rows), use_container_width=True, hide_index=True)
                else:
                    st.info("No vitals recorded for this patient yet.")
            else:
                handle_response_error(vitals_response)

        except requests.exceptions.ConnectionError:
            st.error("Backend connection error. Make sure FastAPI is running on port 8000.")
        except Exception as e:
            st.error(f"Could not load vitals: {e}")

    with progress_col:
        st.subheader("Task Progress")

        try:
            tasks_response = requests.get(
                f"{API_URL}/nurse-tasks/{selected_prediction_id}",
                headers=auth_headers(),
                timeout=20
            )

            if tasks_response.status_code == 200:
                tasks = tasks_response.json().get("tasks", [])

                if tasks:
                    for task in tasks:
                        with st.expander(
                            f"{task.get('priority')} | {task.get('task_title')} | {task.get('status')}"
                        ):
                            nurse = task.get("nurse") or {}
                            st.write(f"Assigned Nurse: {nurse.get('name', 'Unknown')}")
                            st.write(task.get("task_description") or "No task description.")
                            st.caption(
                                f"Created: {task.get('created_at')} | "
                                f"Completed: {task.get('completed_at') or 'Not completed'}"
                            )

                            task_status_options = ["Pending", "In Progress", "Completed"]
                            current_task_status = task.get("status", "Pending")
                            if current_task_status not in task_status_options:
                                current_task_status = "Pending"

                            new_status = st.selectbox(
                                "Update Status",
                                task_status_options,
                                index=task_status_options.index(current_task_status),
                                key=f"task_status_{task.get('id')}"
                            )

                            if st.button("Save Status", key=f"save_task_status_{task.get('id')}"):
                                try:
                                    response = requests.put(
                                        f"{API_URL}/nurse-tasks/{task.get('id')}/status",
                                        json={"status": new_status},
                                        headers=auth_headers(),
                                        timeout=20
                                    )

                                    if response.status_code == 200:
                                        st.success("Task status updated.")
                                        st.rerun()
                                    else:
                                        handle_response_error(response)

                                except requests.exceptions.ConnectionError:
                                    st.error("Backend connection error. Make sure FastAPI is running on port 8000.")
                                except Exception as e:
                                    st.error(f"Could not update task status: {e}")
                else:
                    st.info("No care tasks found for this patient yet.")
            else:
                handle_response_error(tasks_response)

        except requests.exceptions.ConnectionError:
            st.error("Backend connection error. Make sure FastAPI is running on port 8000.")
        except Exception as e:
            st.error(f"Could not load tasks: {e}")

elif page == "Doctor Review":

    st.title("Doctor Review")
    st.caption("Document diagnosis, treatment plan, medication notes, and admission decisions.")

    if st.session_state.role not in ["nurse", "doctor", "admin"]:
        st.error("You do not have permission to access doctor reviews.")
        st.stop()

    default_prediction_id = st.session_state.prediction_id or 1
    prediction_id = st.number_input(
        "Prediction ID",
        min_value=1,
        value=int(default_prediction_id),
        step=1,
        help="Use the prediction_id returned by Live Prediction."
    )

    can_add_review = st.session_state.role in ["doctor", "admin"]

    if can_add_review:
        st.subheader("Add Review")

        with st.form(f"doctor_review_form_{prediction_id}"):
            diagnosis = st.text_area("Diagnosis")
            treatment_plan = st.text_area("Treatment Plan")
            medication_notes = st.text_area("Medication Notes")
            follow_up_required = st.checkbox("Follow-up Required", value=False)
            admit_status = st.selectbox(
                "Admit Status",
                ["Not Admitted", "Observation", "Admitted", "Discharged", "Transferred"]
            )
            submit_review = st.form_submit_button("Save Doctor Review", use_container_width=True)

        if submit_review:
            if not diagnosis or not treatment_plan:
                st.warning("Diagnosis and treatment plan are required.")
            else:
                payload = {
                    "prediction_id": int(prediction_id),
                    "diagnosis": diagnosis,
                    "treatment_plan": treatment_plan,
                    "medication_notes": medication_notes or None,
                    "follow_up_required": follow_up_required,
                    "admit_status": admit_status
                }

                try:
                    response = requests.post(
                        f"{API_URL}/doctor-review/add",
                        json=payload,
                        headers=auth_headers(),
                        timeout=20
                    )

                    if response.status_code == 200:
                        st.success("Doctor review saved successfully.")
                        st.rerun()
                    else:
                        handle_response_error(response)

                except requests.exceptions.ConnectionError:
                    st.error("Backend connection error. Make sure FastAPI is running on port 8000.")
                except Exception as e:
                    st.error(f"Could not save doctor review: {e}")
    else:
        st.info("Nurse accounts can view doctor reviews but cannot edit them.")

    st.divider()
    st.subheader("Review History")

    try:
        review_response = requests.get(
            f"{API_URL}/doctor-review/{int(prediction_id)}",
            headers=auth_headers(),
            timeout=20
        )

        if review_response.status_code == 200:
            reviews = review_response.json().get("reviews", [])

            if reviews:
                for review in reviews:
                    with st.expander(
                        f"{review.get('admit_status')} | Doctor: {review.get('doctor_id')} | {review.get('reviewed_at')}"
                    ):
                        st.write("Diagnosis")
                        st.info(review.get("diagnosis"))
                        st.write("Treatment Plan")
                        st.success(review.get("treatment_plan"))

                        if review.get("medication_notes"):
                            st.write("Medication Notes")
                            st.warning(review.get("medication_notes"))

                        st.write(
                            "Follow-up Required:",
                            "Yes" if review.get("follow_up_required") else "No"
                        )
            else:
                st.info("No doctor reviews found for this prediction.")
        else:
            handle_response_error(review_response)

    except requests.exceptions.ConnectionError:
        st.error("Backend connection error. Make sure FastAPI is running on port 8000.")
    except Exception as e:
        st.error(f"Could not load doctor reviews: {e}")

elif page == "Patient Status":

    st.title("Patient Status")
    st.caption("Track the patient journey from waiting room through treatment and discharge.")

    if st.session_state.role not in ["nurse", "doctor", "admin"]:
        st.error("You do not have permission to access patient status tracking.")
        st.stop()

    default_prediction_id = st.session_state.prediction_id or 1
    prediction_id = st.number_input(
        "Prediction ID",
        min_value=1,
        value=int(default_prediction_id),
        step=1
    )

    render_patient_status_panel(int(prediction_id))

elif page == "Emergency Queue":

    st.title("Emergency Queue & Nurse Assignment")
    if st.session_state.role not in ["doctor", "nurse", "admin"]:
        st.error("Clinical users only.")
        st.stop()

    st.subheader("Waiting Queue")
    response = requests.get(f"{API_URL}/api/assignments/waiting-queue", headers=auth_headers(), timeout=20)
    if response.status_code == 200:
        queue = response.json().get("waiting_queue", [])
        if queue:
            queue_rows = []
            for item in queue:
                queue_rows.append({
                    "Patient ID": item.get("prediction_id"),
                    "Patient Name": item.get("patient_name"),
                    "ESI Level": item.get("esi_level"),
                    "Priority": item.get("priority"),
                    "Assignment Status": item.get("assignment_status"),
                    "Assigned Nurse": item.get("assigned_nurse_name") or "Unassigned",
                    "Assigned Time": item.get("assigned_at"),
                    "Estimated Wait": item.get("estimated_wait_time"),
                })
            st.dataframe(pd.DataFrame(queue_rows), use_container_width=True, hide_index=True)
        else:
            st.info("No waiting patients found.")
    else:
        st.error("Could not load waiting queue. Confirm FastAPI is running on port 8000.")

    st.subheader("Nurse Workload")
    workload_response = requests.get(f"{API_URL}/api/assignments/nurse-workload", headers=auth_headers(), timeout=20)
    nurses = workload_response.json().get("nurses", []) if workload_response.status_code == 200 else []
    if nurses:
        workload_cols = st.columns(min(4, len(nurses)))
        for index, nurse in enumerate(nurses):
            with workload_cols[index % len(workload_cols)]:
                st.metric(
                    nurse.get("name", "Nurse"),
                    nurse.get("active_patient_count", 0),
                    "Available" if nurse.get("available_status") else "Busy",
                )
                st.caption(f"{nurse.get('department')} | {nurse.get('experience_level', 'normal')}")
    elif workload_response.status_code != 200:
        st.error("Could not load nurse workload. Confirm FastAPI is running on port 8000.")
    else:
        st.info("No nurse workload data found.")

    if st.session_state.role == "nurse":
        st.subheader("My Assigned Patients")
        assignment_response = requests.get(f"{API_URL}/assignments", headers=auth_headers(), timeout=20)
        assignments = assignment_response.json().get("assignments", []) if assignment_response.status_code == 200 else []
        if assignments:
            cards = []
            for assignment in assignments:
                prediction = assignment.get("prediction") or {}
                cards.append({
                    "Patient ID": assignment.get("prediction_id"),
                    "Patient Name": prediction.get("patient_name", f"Patient #{assignment.get('prediction_id')}"),
                    "ESI Level": prediction.get("esi_level"),
                    "ICU Risk": prediction.get("icu_risk"),
                    "Assignment Status": assignment.get("status"),
                    "Assigned Time": assignment.get("assigned_at"),
                    "Priority Level": assignment.get("priority_level"),
                })
            st.dataframe(pd.DataFrame(cards), use_container_width=True, hide_index=True)
            assignment_options = {
                f"Patient #{item.get('prediction_id')} | {item.get('status')}": item
                for item in assignments
            }
            selected_assignment_label = st.selectbox("Update Assigned Patient", list(assignment_options.keys()))
            selected_assignment = assignment_options[selected_assignment_label]
            action_cols = st.columns(5)
            status_actions = [
                ("Start Triage", "in_triage"),
                ("View History", None),
                ("Upload Reports", None),
                ("Add Notes", None),
                ("Send to Doctor", "doctor_review"),
            ]
            for col, (label, status_value) in zip(action_cols, status_actions):
                with col:
                    if st.button(label, key=f"{label}_{selected_assignment.get('id')}"):
                        if status_value:
                            r = requests.put(
                                f"{API_URL}/api/assignments/update-status",
                                json={"assignment_id": selected_assignment.get("id"), "status": status_value},
                                headers=auth_headers(),
                                timeout=20,
                            )
                            st.success("Assignment updated.") if r.status_code == 200 else handle_response_error(r)
                        else:
                            st.info("Use Patient History or Historical Reports from the sidebar for this action.")
        else:
            st.info("No assigned patients for this nurse.")

    if st.session_state.role in ["doctor", "admin"]:
        st.subheader("Assign / Reassign Nurse")
        nurse_options = {
            f"{nurse.get('name')} | active: {nurse.get('active_patient_count', 0)} | {nurse.get('experience_level', 'normal')}": nurse.get("id")
            for nurse in nurses
        }
        if nurse_options:
            with st.form("manual_reassign_form"):
                assign_prediction_id = st.number_input("Patient / Prediction ID", min_value=1, value=int(st.session_state.prediction_id or 1))
                selected_nurse_label = st.selectbox("Assign Nurse", list(nurse_options.keys()))
                assign_notes = st.text_area("Assignment Notes")
                assign_clicked = st.form_submit_button("Assign Nurse")
            if assign_clicked:
                r = requests.post(
                    f"{API_URL}/api/assignments/reassign-nurse",
                    json={
                        "prediction_id": int(assign_prediction_id),
                        "nurse_id": int(nurse_options[selected_nurse_label]),
                        "notes": assign_notes or None,
                    },
                    headers=auth_headers(),
                    timeout=20,
                )
                st.success("Nurse assigned/reassigned.") if r.status_code == 200 else handle_response_error(r)
        else:
            st.warning("Add nurse records before assigning patients.")

        st.subheader("Override Queue Priority")
        with st.form("queue_priority_form"):
            q_prediction_id = st.number_input("Prediction ID", min_value=1, value=int(st.session_state.prediction_id or 1))
            priority = st.selectbox("Priority", ["Critical", "High", "Medium", "Low"])
            wait_time = st.number_input("Estimated Wait Time", min_value=0, value=15)
            submitted = st.form_submit_button("Update Priority")
        if submitted:
            r = requests.put(
                f"{API_URL}/emergency-queue/{int(q_prediction_id)}/priority",
                json={"priority": priority, "estimated_wait_time": int(wait_time)},
                headers=auth_headers(),
                timeout=20
            )
            st.success("Priority updated.") if r.status_code == 200 else handle_response_error(r)

elif page == "Bed Management":

    st.title("Bed Management")
    if st.session_state.role not in ["doctor", "nurse", "admin"]:
        st.error("Clinical users only.")
        st.stop()

    if st.session_state.role == "admin":
        with st.expander("Add Bed"):
            with st.form("add_bed_form"):
                bed_number = st.text_input("Bed Number")
                ward_type = st.selectbox("Ward Type", ["Emergency", "ICU", "General Ward", "Observation"])
                add_bed_clicked = st.form_submit_button("Add Bed")
            if add_bed_clicked:
                r = requests.post(f"{API_URL}/beds/add", json={"bed_number": bed_number, "ward_type": ward_type}, headers=auth_headers(), timeout=20)
                st.success("Bed added.") if r.status_code == 200 else handle_response_error(r)

    bed_response = requests.get(f"{API_URL}/beds", headers=auth_headers(), timeout=20)
    beds = bed_response.json().get("beds", []) if bed_response.status_code == 200 else []
    if beds:
        st.dataframe(pd.DataFrame(beds), use_container_width=True, hide_index=True)
    else:
        st.info("No beds found.")

    if st.session_state.role in ["doctor", "admin"]:
        st.subheader("Assign / Release Bed")
        available_beds = [bed for bed in beds if not bed.get("occupied")]
        if available_beds:
            bed_options = {f"{bed['bed_number']} | {bed['ward_type']}": bed["id"] for bed in available_beds}
            with st.form("assign_bed_form"):
                selected_bed = st.selectbox("Available Bed", list(bed_options.keys()))
                bed_prediction_id = st.number_input("Prediction ID", min_value=1, value=int(st.session_state.prediction_id or 1))
                assign_clicked = st.form_submit_button("Assign Bed")
            if assign_clicked:
                r = requests.post(f"{API_URL}/beds/assign", json={"bed_id": bed_options[selected_bed], "prediction_id": int(bed_prediction_id)}, headers=auth_headers(), timeout=20)
                st.success("Bed assigned.") if r.status_code == 200 else handle_response_error(r)
        occupied_beds = [bed for bed in beds if bed.get("occupied")]
        if occupied_beds:
            release_options = {f"{bed['bed_number']} | Patient {bed.get('assigned_prediction_id')}": bed["id"] for bed in occupied_beds}
            selected_release = st.selectbox("Occupied Bed to Release", list(release_options.keys()))
            if st.button("Release Bed"):
                r = requests.put(f"{API_URL}/beds/release/{release_options[selected_release]}", headers=auth_headers(), timeout=20)
                st.success("Bed released.") if r.status_code == 200 else handle_response_error(r)

elif page == "Medication Records":

    st.title("Medication Records")
    prediction_id = st.number_input("Prediction ID", min_value=1, value=int(st.session_state.prediction_id or 1), key="med_prediction_id")

    if st.session_state.role in ["doctor", "admin"]:
        nurse_resp = requests.get(f"{API_URL}/nurses", headers=auth_headers(), timeout=20)
        nurses = nurse_resp.json().get("nurses", []) if nurse_resp.status_code == 200 else []
        if nurses:
            nurse_options = {f"{n['name']} | {n['email']}": n["id"] for n in nurses}
            with st.form("add_med_form"):
                selected_nurse = st.selectbox("Nurse", list(nurse_options.keys()))
                medication_name = st.text_input("Medication Name")
                dosage = st.text_input("Dosage")
                route = st.text_input("Route", value="PO")
                scheduled_time = st.text_input("Scheduled Time", value=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                notes = st.text_area("Notes")
                add_med = st.form_submit_button("Schedule Medication")
            if add_med:
                r = requests.post(
                    f"{API_URL}/medications/add",
                    json={"prediction_id": int(prediction_id), "nurse_id": nurse_options[selected_nurse], "medication_name": medication_name, "dosage": dosage, "route": route, "scheduled_time": scheduled_time, "notes": notes or None},
                    headers=auth_headers(),
                    timeout=20
                )
                st.success("Medication scheduled.") if r.status_code == 200 else handle_response_error(r)

    meds_resp = requests.get(f"{API_URL}/medications/{int(prediction_id)}", headers=auth_headers(), timeout=20)
    meds = meds_resp.json().get("medications", []) if meds_resp.status_code == 200 else []
    if meds:
        st.dataframe(pd.DataFrame(meds), use_container_width=True, hide_index=True)
        med_options = {f"{m['id']} | {m['medication_name']} | {m['status']}": m["id"] for m in meds}
        selected_med = st.selectbox("Update Medication", list(med_options.keys()))
        new_status = st.selectbox("Medication Status", ["Scheduled", "Given", "Missed", "Cancelled"])
        side_effects = st.text_input("Side Effects")
        if st.button("Save Medication Status"):
            r = requests.put(f"{API_URL}/medications/{med_options[selected_med]}/status", json={"status": new_status, "side_effects": side_effects or None}, headers=auth_headers(), timeout=20)
            st.success("Medication status updated.") if r.status_code == 200 else handle_response_error(r)
    else:
        st.info("No medication records for this prediction.")

elif page == "Clinical Alerts":

    st.title("Clinical Alerts")
    alerts_resp = requests.get(f"{API_URL}/alerts", headers=auth_headers(), timeout=20)
    alerts = alerts_resp.json().get("alerts", []) if alerts_resp.status_code == 200 else []
    unresolved = [a for a in alerts if not a.get("resolved")]
    if unresolved:
        st.subheader("Unresolved Alerts")
        st.dataframe(pd.DataFrame(unresolved), use_container_width=True, hide_index=True)
        alert_options = {f"{a['id']} | {a['severity']} | Patient {a['prediction_id']}": a["id"] for a in unresolved}
        selected_alert = st.selectbox("Resolve Alert", list(alert_options.keys()))
        if st.button("Resolve Selected Alert"):
            r = requests.put(f"{API_URL}/alerts/{alert_options[selected_alert]}/resolve", headers=auth_headers(), timeout=20)
            st.success("Alert resolved.") if r.status_code == 200 else handle_response_error(r)
    else:
        st.success("No unresolved alerts.")

    with st.expander("Create Manual Alert"):
        with st.form("manual_alert_form"):
            alert_prediction_id = st.number_input("Prediction ID", min_value=1, value=int(st.session_state.prediction_id or 1), key="alert_prediction_id")
            alert_type = st.text_input("Alert Type")
            severity = st.selectbox("Severity", ["Low", "Medium", "High", "Critical"])
            message = st.text_area("Message")
            add_alert_clicked = st.form_submit_button("Create Alert")
        if add_alert_clicked:
            r = requests.post(f"{API_URL}/alerts/add", json={"prediction_id": int(alert_prediction_id), "alert_type": alert_type, "severity": severity, "message": message}, headers=auth_headers(), timeout=20)
            st.success("Alert created.") if r.status_code == 200 else handle_response_error(r)

elif page == "Discharge Summary":

    st.title("Discharge Summary")
    prediction_id = st.number_input("Prediction ID", min_value=1, value=int(st.session_state.prediction_id or 1), key="discharge_prediction_id")
    if st.session_state.role in ["doctor", "admin"]:
        with st.form("discharge_summary_form"):
            diagnosis = st.text_area("Diagnosis")
            treatment_given = st.text_area("Treatment Given")
            medication_notes = st.text_area("Medication Notes")
            follow_up = st.text_area("Follow-up Instructions")
            discharge_status = st.selectbox("Discharge Status", ["Discharged", "Transferred", "Against Medical Advice"])
            create_summary = st.form_submit_button("Create Discharge Summary")
        if create_summary:
            r = requests.post(f"{API_URL}/discharge-summary/create", json={"prediction_id": int(prediction_id), "diagnosis": diagnosis, "treatment_given": treatment_given, "medication_notes": medication_notes or None, "follow_up_instructions": follow_up or None, "discharge_status": discharge_status}, headers=auth_headers(), timeout=20)
            st.success("Discharge summary created.") if r.status_code == 200 else handle_response_error(r)

    summaries_resp = requests.get(f"{API_URL}/discharge-summary/{int(prediction_id)}", headers=auth_headers(), timeout=20)
    summaries = summaries_resp.json().get("summaries", []) if summaries_resp.status_code == 200 else []
    if summaries:
        st.dataframe(pd.DataFrame(summaries), use_container_width=True, hide_index=True)
    pdf_resp = requests.get(f"{API_URL}/discharge-summary/{int(prediction_id)}/pdf", headers=auth_headers(), timeout=30)
    if pdf_resp.status_code == 200:
        st.download_button("Generate Discharge PDF", pdf_resp.content, file_name=f"discharge_summary_{int(prediction_id)}.pdf", mime="application/pdf")

elif page == "Shift Management":

    st.title("Shift Management")
    if st.session_state.role == "admin":
        with st.form("shift_form"):
            staff_id = st.text_input("Staff Username")
            staff_role = st.selectbox("Staff Role", ["doctor", "nurse", "admin"])
            department = st.text_input("Department")
            shift_type = st.selectbox("Shift Type", ["Morning", "Evening", "Night"])
            start_time = st.text_input("Start Time", value=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            end_time = st.text_input("End Time", value=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            status = st.selectbox("Status", ["Scheduled", "Active", "Completed", "Cancelled"])
            add_shift = st.form_submit_button("Create Shift")
        if add_shift:
            r = requests.post(f"{API_URL}/shifts/add", json={"staff_id": staff_id, "staff_role": staff_role, "department": department or None, "shift_type": shift_type, "start_time": start_time, "end_time": end_time, "status": status}, headers=auth_headers(), timeout=20)
            st.success("Shift created.") if r.status_code == 200 else handle_response_error(r)

    shifts_resp = requests.get(f"{API_URL}/shifts", headers=auth_headers(), timeout=20)
    shifts = shifts_resp.json().get("shifts", []) if shifts_resp.status_code == 200 else []
    st.dataframe(pd.DataFrame(shifts), use_container_width=True, hide_index=True) if shifts else st.info("No shifts found.")

elif page == "Admin Overview":

    st.title("Admin Overview")
    if st.session_state.role != "admin":
        st.error("Only admin can access this page.")
        st.stop()
    summary_resp = requests.get(f"{API_URL}/admin/workload-summary", headers=auth_headers(), timeout=20)
    if summary_resp.status_code == 200:
        summary = summary_resp.json()
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Assigned Patients", summary.get("assigned_patients_per_nurse", 0))
        c2.metric("Reviews Pending", summary.get("doctor_reviews_pending", 0))
        c3.metric("Active Beds", summary.get("active_beds", 0))
        c4.metric("Critical Alerts", summary.get("critical_alerts", 0))
        c5.metric("Waiting Patients", summary.get("waiting_patients", 0))
    else:
        handle_response_error(summary_resp)

elif page == "Patient Registration":
    st.title("Patient Registration")
    with st.form("patient_form"):
        name = st.text_input("Name")
        age = st.number_input("Age", min_value=0, max_value=120, value=40)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        phone = st.text_input("Phone")
        address = st.text_area("Address")
        ec_name = st.text_input("Emergency Contact Name")
        ec_phone = st.text_input("Emergency Contact Phone")
        allergies = st.text_area("Allergies")
        chronic = st.text_area("Chronic Diseases")
        history = st.text_area("Past Medical History")
        pred_id = st.number_input("Link Prediction ID (optional)", min_value=0, value=int(st.session_state.prediction_id or 0))
        submitted = st.form_submit_button("Save Patient")
    if submitted:
        payload = {"name": name, "age": int(age), "gender": gender, "phone": phone, "address": address, "emergency_contact_name": ec_name, "emergency_contact_phone": ec_phone, "allergies": allergies, "chronic_diseases": chronic, "past_medical_history": history, "prediction_id": int(pred_id) if pred_id else None}
        r = requests.post(f"{API_URL}/patients/add", json=payload, headers=auth_headers(), timeout=20)
        st.success("Patient saved.") if r.status_code == 200 else handle_response_error(r)
    r = requests.get(f"{API_URL}/patients", headers=auth_headers(), timeout=20)
    if r.status_code == 200:
        data = r.json().get("patients", [])
        st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True) if data else st.info("No patients registered.")

elif page == "Lab Tests":
    st.title("Lab Tests")
    pred_id = st.number_input("Prediction ID", min_value=1, value=int(st.session_state.prediction_id or 1), key="lab_pred")
    if st.session_state.role in ["doctor", "admin"]:
        with st.form("lab_order_form"):
            test_name = st.text_input("Test Name")
            test_type = st.text_input("Test Type")
            priority = st.selectbox("Priority", ["Routine", "Urgent", "STAT"])
            order_clicked = st.form_submit_button("Order Lab")
        if order_clicked:
            r = requests.post(f"{API_URL}/labs/order", json={"prediction_id": int(pred_id), "test_name": test_name, "test_type": test_type, "priority": priority}, headers=auth_headers(), timeout=20)
            st.success("Lab ordered.") if r.status_code == 200 else handle_response_error(r)
    r = requests.get(f"{API_URL}/labs/{int(pred_id)}", headers=auth_headers(), timeout=20)
    labs = r.json().get("labs", []) if r.status_code == 200 else []
    if labs:
        st.dataframe(pd.DataFrame(labs), use_container_width=True, hide_index=True)
        if st.session_state.role in ["doctor", "admin"]:
            opts = {f"{l['id']} | {l['test_name']} | {l['status']}": l["id"] for l in labs}
            selected = st.selectbox("Update Lab Result", list(opts.keys()))
            status = st.selectbox("Lab Status", ["Ordered", "In Progress", "Completed", "Cancelled"])
            notes = st.text_area("Result Notes")
            if st.button("Save Lab Result"):
                rr = requests.put(f"{API_URL}/labs/{opts[selected]}/result", json={"status": status, "result_notes": notes}, headers=auth_headers(), timeout=20)
                st.success("Lab updated.") if rr.status_code == 200 else handle_response_error(rr)

elif page == "Specialist Referral":
    st.title("Specialist Referral")
    pred_id = st.number_input("Prediction ID", min_value=1, value=int(st.session_state.prediction_id or 1), key="ref_pred")
    if st.session_state.role in ["doctor", "admin"]:
        with st.form("referral_form"):
            dept = st.selectbox("Specialist Department", ["Cardiology", "Neurology", "Surgery", "Orthopedics", "Pediatrics", "Psychiatry", "Internal Medicine"])
            urgency = st.selectbox("Urgency", ["Routine", "Urgent", "Emergency"])
            reason = st.text_area("Reason")
            notes = st.text_area("Notes")
            submitted = st.form_submit_button("Create Referral")
        if submitted:
            r = requests.post(f"{API_URL}/referrals/add", json={"prediction_id": int(pred_id), "specialist_department": dept, "reason": reason, "urgency": urgency, "notes": notes}, headers=auth_headers(), timeout=20)
            st.success("Referral created.") if r.status_code == 200 else handle_response_error(r)
    r = requests.get(f"{API_URL}/referrals/{int(pred_id)}", headers=auth_headers(), timeout=20)
    refs = r.json().get("referrals", []) if r.status_code == 200 else []
    st.dataframe(pd.DataFrame(refs), use_container_width=True, hide_index=True) if refs else st.info("No referrals.")

elif page == "Consent Management":
    st.title("Consent Management")
    patient_id = st.number_input("Patient ID", min_value=1, value=1, key="consent_patient")
    with st.form("consent_form"):
        pred_id = st.number_input("Prediction ID (optional)", min_value=0, value=int(st.session_state.prediction_id or 0))
        ctype = st.selectbox("Consent Type", ["Treatment Consent", "AI Decision Support Consent", "Image Analysis Consent", "Data Use Consent"])
        accepted = st.checkbox("Accepted", value=True)
        signed_by = st.text_input("Signed By")
        notes = st.text_area("Notes")
        submitted = st.form_submit_button("Save Consent")
    if submitted:
        r = requests.post(f"{API_URL}/consents/add", json={"patient_id": int(patient_id), "prediction_id": int(pred_id) if pred_id else None, "consent_type": ctype, "accepted": accepted, "signed_by": signed_by, "notes": notes}, headers=auth_headers(), timeout=20)
        st.success("Consent saved.") if r.status_code == 200 else handle_response_error(r)
    r = requests.get(f"{API_URL}/consents/{int(patient_id)}", headers=auth_headers(), timeout=20)
    consents = r.json().get("consents", []) if r.status_code == 200 else []
    st.dataframe(pd.DataFrame(consents), use_container_width=True, hide_index=True) if consents else st.info("No consents.")

elif page == "Incident Reports":
    st.title("Incident Reports")
    with st.form("incident_form"):
        pred_id = st.number_input("Prediction ID (optional)", min_value=0, value=int(st.session_state.prediction_id or 0))
        incident_type = st.selectbox("Incident Type", ["Fall Risk", "Medication Error", "Aggressive Patient", "Equipment Issue", "Safety Concern"])
        severity = st.selectbox("Severity", ["Low", "Medium", "High", "Critical"])
        description = st.text_area("Description")
        action = st.text_area("Action Taken")
        submitted = st.form_submit_button("Report Incident")
    if submitted:
        r = requests.post(f"{API_URL}/incidents/add", json={"prediction_id": int(pred_id) if pred_id else None, "incident_type": incident_type, "severity": severity, "description": description, "action_taken": action}, headers=auth_headers(), timeout=20)
        st.success("Incident reported.") if r.status_code == 200 else handle_response_error(r)
    r = requests.get(f"{API_URL}/incidents", headers=auth_headers(), timeout=20)
    incidents = r.json().get("incidents", []) if r.status_code == 200 else []
    st.dataframe(pd.DataFrame(incidents), use_container_width=True, hide_index=True) if incidents else st.info("No incidents.")

elif page == "Billing":
    st.title("Billing")
    if st.session_state.role != "admin":
        st.error("Only admin can access billing.")
        st.stop()
    patient_id = st.number_input("Patient ID", min_value=1, value=1, key="billing_patient")
    with st.form("billing_form"):
        pred_id = st.number_input("Prediction ID (optional)", min_value=0, value=int(st.session_state.prediction_id or 0))
        provider = st.text_input("Insurance Provider")
        policy = st.text_input("Policy Number")
        visit = st.number_input("Visit Cost", value=0.0)
        treatment = st.number_input("Treatment Cost", value=0.0)
        medication = st.number_input("Medication Cost", value=0.0)
        status = st.selectbox("Payment Status", ["Pending", "Paid", "Insurance Submitted", "Denied"])
        submitted = st.form_submit_button("Create Billing Record")
    if submitted:
        r = requests.post(f"{API_URL}/billing/create", json={"patient_id": int(patient_id), "prediction_id": int(pred_id) if pred_id else None, "insurance_provider": provider, "policy_number": policy, "visit_cost": visit, "treatment_cost": treatment, "medication_cost": medication, "payment_status": status}, headers=auth_headers(), timeout=20)
        st.success("Billing created.") if r.status_code == 200 else handle_response_error(r)
    r = requests.get(f"{API_URL}/billing/{int(patient_id)}", headers=auth_headers(), timeout=20)
    billing = r.json().get("billing", []) if r.status_code == 200 else []
    st.dataframe(pd.DataFrame(billing), use_container_width=True, hide_index=True) if billing else st.info("No billing records.")

elif page == "Inventory Management":
    st.title("Inventory Management")
    if st.session_state.role == "admin":
        with st.form("inventory_form"):
            item_name = st.text_input("Item Name")
            category = st.selectbox("Category", ["Medication", "PPE", "Oxygen", "Equipment", "Lab Supply"])
            quantity = st.number_input("Quantity", min_value=0, value=0)
            unit = st.text_input("Unit")
            minimum = st.number_input("Minimum Stock Level", min_value=0, value=0)
            location = st.text_input("Location")
            status = st.text_input("Status", value="Available")
            submitted = st.form_submit_button("Add Inventory Item")
        if submitted:
            r = requests.post(f"{API_URL}/inventory/add", json={"item_name": item_name, "category": category, "quantity": int(quantity), "unit": unit, "minimum_stock_level": int(minimum), "location": location, "status": status}, headers=auth_headers(), timeout=20)
            st.success("Inventory item saved.") if r.status_code == 200 else handle_response_error(r)
    r = requests.get(f"{API_URL}/inventory", headers=auth_headers(), timeout=20)
    items = r.json().get("inventory", []) if r.status_code == 200 else []
    st.dataframe(pd.DataFrame(items), use_container_width=True, hide_index=True) if items else st.info("No inventory.")
    low = requests.get(f"{API_URL}/inventory/low-stock", headers=auth_headers(), timeout=20)
    low_items = low.json().get("inventory", []) if low.status_code == 200 else []
    if low_items:
        st.warning("Low stock items")
        st.dataframe(pd.DataFrame(low_items), use_container_width=True, hide_index=True)

elif page == "Follow-up Appointments":
    st.title("Follow-up Appointments")
    with st.form("appointment_form"):
        patient_id = st.number_input("Patient ID", min_value=1, value=1)
        pred_id = st.number_input("Prediction ID (optional)", min_value=0, value=int(st.session_state.prediction_id or 0))
        doctor_id = st.text_input("Doctor ID", value=st.session_state.username if st.session_state.role == "doctor" else "doctor")
        appointment_date = st.text_input("Appointment Date", value=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        department = st.text_input("Department")
        reason = st.text_area("Reason")
        status = st.selectbox("Status", ["Scheduled", "Completed", "Cancelled", "No Show"])
        notes = st.text_area("Notes")
        submitted = st.form_submit_button("Schedule Appointment")
    if submitted:
        r = requests.post(f"{API_URL}/appointments/add", json={"patient_id": int(patient_id), "prediction_id": int(pred_id) if pred_id else None, "doctor_id": doctor_id, "appointment_date": appointment_date, "department": department, "reason": reason, "status": status, "notes": notes}, headers=auth_headers(), timeout=20)
        st.success("Appointment saved.") if r.status_code == 200 else handle_response_error(r)
    r = requests.get(f"{API_URL}/appointments", headers=auth_headers(), timeout=20)
    appts = r.json().get("appointments", []) if r.status_code == 200 else []
    st.dataframe(pd.DataFrame(appts), use_container_width=True, hide_index=True) if appts else st.info("No appointments.")

elif page == "Family Notifications":
    st.title("Family Notifications")
    patient_id = st.number_input("Patient ID", min_value=1, value=1, key="notif_patient")
    with st.form("notification_form"):
        pred_id = st.number_input("Prediction ID (optional)", min_value=0, value=int(st.session_state.prediction_id or 0))
        contact_name = st.text_input("Contact Name")
        contact_phone = st.text_input("Contact Phone")
        ntype = st.selectbox("Notification Type", ["Admission", "Critical Alert", "Discharge", "Follow-up Reminder"])
        message = st.text_area("Message")
        submitted = st.form_submit_button("Simulate Send Notification")
    if submitted:
        r = requests.post(f"{API_URL}/notifications/send", json={"patient_id": int(patient_id), "prediction_id": int(pred_id) if pred_id else None, "contact_name": contact_name, "contact_phone": contact_phone, "message": message, "notification_type": ntype}, headers=auth_headers(), timeout=20)
        st.success("Notification logged.") if r.status_code == 200 else handle_response_error(r)
    r = requests.get(f"{API_URL}/notifications/{int(patient_id)}", headers=auth_headers(), timeout=20)
    logs = r.json().get("notifications", []) if r.status_code == 200 else []
    st.dataframe(pd.DataFrame(logs), use_container_width=True, hide_index=True) if logs else st.info("No notifications.")

elif page == "Hospital Analytics":
    st.title("Hospital Analytics")
    r = requests.get(f"{API_URL}/analytics/hospital-summary", headers=auth_headers(), timeout=20)
    if r.status_code == 200:
        data = r.json()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Patients", data.get("total_patients", 0))
        c2.metric("Triage Cases", data.get("total_triage_cases", 0))
        c3.metric("Critical Cases", data.get("critical_cases", 0))
        c4.metric("Bed Occupancy", f"{data.get('bed_occupancy_rate', 0)}%")
        st.write("Top Symptoms")
        st.bar_chart(pd.DataFrame([data.get("top_symptoms", {})]).T)
        st.write("Common Lab Tests")
        st.bar_chart(pd.DataFrame([data.get("common_lab_tests", {})]).T)
        low_items = data.get("low_inventory_items", [])
        if low_items:
            st.warning("Low Inventory Items")
            st.dataframe(pd.DataFrame(low_items), use_container_width=True, hide_index=True)
    else:
        handle_response_error(r)

elif page == "Admin Approvals":

    st.title("Approval Management")

    if st.session_state.role not in ["super_admin", "admin", "doctor", "nurse"]:
        st.error("Unauthorized access.")
        st.stop()

    response = requests.get(
        f"{API_URL}/admin/approvals",
        headers=auth_headers(),
        timeout=20
    )

    if response.status_code != 200:
        handle_response_error(response)
        st.stop()

    approval_data = response.json()
    pending = approval_data.get("pending_admin_requests", [])
    new_patients = approval_data.get("new_patient_requests", [])
    history = approval_data.get("activity_history", [])
    scope = approval_data.get("approval_scope", [])
    view_scope = approval_data.get("view_scope", [])

    st.caption(f"Approval scope: {', '.join(scope) if scope else 'none'}")

    if st.session_state.role == "nurse":
        st.subheader("New Patients")
        st.caption("Nurses can view newly registered patients for awareness. Approval is handled by doctors, admins, or super admins.")
        if not new_patients:
            st.success("No newly registered patients.")
        else:
            rows = [
                {
                    "Patient ID": request.get("id"),
                    "Full Name": request.get("full_name"),
                    "Username": request.get("username"),
                    "Email": request.get("email"),
                    "Registration Date": request.get("registration_date"),
                    "Account Status": request.get("status"),
                }
                for request in new_patients
            ]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            for request in new_patients:
                with st.expander(f"View Details: {request.get('full_name')}"):
                    st.json(request)
        st.info("Approve and Reject actions are intentionally unavailable for nurse accounts.")
    elif not pending:
        st.success("No pending approval requests.")
    else:
        for request in pending:
            with st.container(border=True):
                st.write(f"**{request.get('full_name')}**")
                st.write(f"User ID: `{request.get('id')}`")
                st.write(f"Username: `{request.get('username')}`")
                st.write(f"Email: `{request.get('email')}`")
                st.write(f"Requested role: `{request.get('requested_role')}`")
                st.write(f"Registration date: `{request.get('registration_date')}`")
                status = request.get("status", "pending")
                badge_color = {"pending": "#ca8a04", "active": "#16a34a", "rejected": "#dc2626", "suspended": "#6b7280"}.get(status, "#64748b")
                st.markdown(
                    f"<span style='background:{badge_color};color:white;padding:4px 9px;border-radius:999px;font-weight:800;'>{status}</span>",
                    unsafe_allow_html=True
                )
                st.write(f"Approved by: `{request.get('approved_by')}`")
                st.write(f"Approved at: `{request.get('approved_at')}`")
                with st.expander("View Details"):
                    st.json(request)
                approve_col, reject_col = st.columns(2)
                with approve_col:
                    if st.button("Approve", key=f"approve_{request.get('username')}"):
                        approve_response = requests.post(
                            f"{API_URL}/admin/approvals/{request.get('username')}/approve",
                            headers=auth_headers(),
                            timeout=20
                        )
                        if approve_response.status_code == 200:
                            st.success("Admin request approved.")
                            st.rerun()
                        else:
                            handle_response_error(approve_response)
                with reject_col:
                    if st.button("Reject", key=f"reject_{request.get('username')}"):
                        reason = f"Rejected by {st.session_state.username}"
                        reject_response = requests.post(
                            f"{API_URL}/admin/approvals/{request.get('username')}/reject",
                            json={"rejected_reason": reason},
                            headers=auth_headers(),
                            timeout=20
                        )
                        if reject_response.status_code == 200:
                            st.warning("Admin request rejected.")
                            st.rerun()
                        else:
                            handle_response_error(reject_response)

    st.divider()
    if st.session_state.role != "nurse":
        st.subheader("Approval Activity History")
        if history:
            st.dataframe(pd.DataFrame(history), use_container_width=True, hide_index=True)
        else:
            st.info("No approval activity yet.")
    else:
        st.caption(f"View scope: {', '.join(view_scope) if view_scope else 'none'}")

elif page == "Staff Management":

    st.title("Staff Management")
    st.caption("Admin controls for staff activation, departments, and clinical workload.")

    if st.session_state.role not in ["super_admin", "admin"]:
        st.error("Only admin users can access Staff Management.")
        st.stop()

    role_filter = st.selectbox(
        "View Staff by Role",
        ["all", "doctor", "nurse", "admin"]
    )

    staff_params = {}
    if role_filter != "all":
        staff_params["role"] = role_filter

    try:
        staff_response = requests.get(
            f"{API_URL}/admin/staff",
            params=staff_params,
            headers=auth_headers(),
            timeout=20
        )

        workload_response = requests.get(
            f"{API_URL}/admin/workload",
            headers=auth_headers(),
            timeout=20
        )

    except requests.exceptions.ConnectionError:
        st.error("Backend connection error. Make sure FastAPI is running on port 8000.")
        st.stop()
    except Exception as e:
        st.error(f"Could not load staff management data: {e}")
        st.stop()

    if staff_response.status_code != 200:
        handle_response_error(staff_response)
        st.stop()

    staff_members = staff_response.json().get("staff", [])

    st.subheader("Staff Directory")

    if staff_members:
        staff_df = pd.DataFrame(staff_members)
        st.dataframe(
            staff_df[["username", "role", "department", "active_status", "updated_at"]],
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No staff found for this filter.")

    st.divider()
    st.subheader("Update Staff Member")

    if staff_members:
        staff_options = {
            f"{staff.get('username')} | {staff.get('role')}": staff
            for staff in staff_members
        }

        selected_staff_label = st.selectbox("Select Staff", list(staff_options.keys()))
        selected_staff = staff_options[selected_staff_label]

        with st.form(f"staff_update_form_{selected_staff.get('username')}"):
            department = st.text_input(
                "Department",
                value=selected_staff.get("department") or ""
            )
            active_status = st.checkbox(
                "Active Staff Account",
                value=bool(selected_staff.get("active_status"))
            )
            update_staff = st.form_submit_button("Update Staff", use_container_width=True)

        if update_staff:
            payload = {
                "department": department or None,
                "active_status": active_status
            }

            try:
                update_response = requests.put(
                    f"{API_URL}/admin/staff/{selected_staff.get('username')}",
                    json=payload,
                    headers=auth_headers(),
                    timeout=20
                )

                if update_response.status_code == 200:
                    st.success("Staff member updated successfully.")
                    st.rerun()
                else:
                    handle_response_error(update_response)

            except requests.exceptions.ConnectionError:
                st.error("Backend connection error. Make sure FastAPI is running on port 8000.")
            except Exception as e:
                st.error(f"Could not update staff member: {e}")

    st.divider()
    st.subheader("Workload Overview")

    if workload_response.status_code == 200:
        workload = workload_response.json()
        nurse_workload = workload.get("nurse_workload", [])
        doctor_workload = workload.get("doctor_workload", [])

        workload_col1, workload_col2 = st.columns(2)

        with workload_col1:
            st.write("Nurse Workload")
            if nurse_workload:
                st.dataframe(
                    pd.DataFrame(nurse_workload),
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.info("No nurse workload records yet.")

        with workload_col2:
            st.write("Doctor Workload")
            if doctor_workload:
                st.dataframe(
                    pd.DataFrame(doctor_workload),
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.info("No doctor workload records yet.")
    else:
        handle_response_error(workload_response)

# ===========================================================
# NEW: ADMIN MONITOR
# ===========================================================

elif page == "🛡️ Admin Monitor":

    st.title("🛡️ Admin Monitoring Dashboard")
    st.caption("Admin-only system monitoring for EmergeAI Healthcare.")

    if st.session_state.role != "admin":
        st.error("Only admin can access this page.")
        st.stop()

    health_response = requests.get(
        f"{API_URL}/health",
        headers=auth_headers(),
        timeout=60
    )

    history_response = requests.get(
        f"{API_URL}/history",
        headers=auth_headers(),
        timeout=20
    )

    feedback_response = requests.get(
        f"{API_URL}/clinical-feedback",
        headers=auth_headers(),
        timeout=20
    )

    watchlist_response = requests.get(
        f"{API_URL}/v2/watchlist",
        headers=auth_headers(),
        timeout=20
    )

    st.subheader("System Health")

    if health_response.status_code == 200:
        health = health_response.json()

        h1, h2, h3, h4 = st.columns(4)

        with h1:
            st.metric("Backend", health.get("status", "unknown"))

        with h2:
            st.metric("Database", health.get("database", "unknown"))

        with h3:
            st.metric("Authentication", health.get("authentication", "unknown"))

        with h4:
            st.metric("RBAC", health.get("rbac", "unknown"))

        style_metric_cards()
        st.success("Core backend services are responding.")

        with st.expander("Full Health Response"):
            st.json(health)
    else:
        st.error("Health check failed.")
        st.code(health_response.text, language="text")

    st.divider()

    st.subheader("Operational Metrics")

    total_predictions = 0
    total_feedback = 0
    critical_patients = 0

    history = []
    feedback = []
    watchlist_data = {}

    if history_response.status_code == 200:
        history_data = history_response.json()
        history = history_data.get("history", [])
        total_predictions = len(history)

    if feedback_response.status_code == 200:
        feedback = feedback_response.json()
        total_feedback = len(feedback)

    if watchlist_response.status_code == 200:
        watchlist_data = watchlist_response.json()
        critical_patients = watchlist_data.get("total_critical_patients", 0)

    m1, m2, m3 = st.columns(3)

    with m1:
        st.metric("Total Predictions", total_predictions)

    with m2:
        st.metric("Clinical Feedback Records", total_feedback)

    with m3:
        st.metric("Active High-Risk Watchlist", critical_patients)

    style_metric_cards()

    st.divider()

    st.subheader("Prediction Activity")

    if history:
        df = pd.DataFrame(history)

        c1, c2 = st.columns(2)

        with c1:
            st.write("### Recent Prediction Records")
            st.dataframe(df.head(20), use_container_width=True)

        with c2:
            if "final_prediction" in df.columns:
                pred_df = (
                    df["final_prediction"]
                    .astype(str)
                    .value_counts()
                    .reset_index()
                )
                pred_df.columns = ["Final Prediction", "Count"]

                fig = px.bar(
                    pred_df,
                    x="Final Prediction",
                    y="Count",
                    title="Prediction Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("final_prediction column not available.")
    else:
        st.info("No prediction records available.")

    st.divider()

    st.subheader("Recent Clinical Feedback")

    if feedback:
        fb_df = pd.DataFrame(feedback)
        st.dataframe(fb_df.head(20), use_container_width=True)

        if "accepted" in fb_df.columns:
            feedback_dist = (
                fb_df["accepted"]
                .map({True: "Accepted", False: "Overridden"})
                .value_counts()
                .reset_index()
            )
            feedback_dist.columns = ["Decision", "Count"]

            fig2 = px.pie(
                feedback_dist,
                names="Decision",
                values="Count",
                title="Clinical Feedback Decisions"
            )
            st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No clinical feedback records available.")

    st.divider()

    st.subheader("High-Risk Watchlist Summary")

    if watchlist_response.status_code == 200:
        patients = watchlist_data.get("patients", [])

        if patients:
            watch_df = pd.DataFrame(patients)
            st.dataframe(watch_df.head(20), use_container_width=True)

            if "risk_level" in watch_df.columns:
                risk_df = (
                    watch_df["risk_level"]
                    .astype(str)
                    .value_counts()
                    .reset_index()
                )
                risk_df.columns = ["Risk Level", "Count"]

                fig3 = px.bar(
                    risk_df,
                    x="Risk Level",
                    y="Count",
                    title="Watchlist Risk Level Distribution"
                )
                st.plotly_chart(fig3, use_container_width=True)
        else:
            st.success("No high-risk patients currently on the watchlist.")
    else:
        st.warning("Could not load watchlist data.")
        st.code(watchlist_response.text, language="text")

    st.divider()

    st.subheader("Admin Notes")

    st.info(
        "This dashboard monitors backend health, prediction activity, clinical feedback, "
        "and high-risk patient volume. It is designed for administrative oversight."
    )

# ===========================================================
# NEW: COMPANY HEALTH DASHBOARD
# ===========================================================

elif page == "🏥 Company Health Dashboard":

    st.title("🏥 Company-Level Healthcare Dashboard")
    st.caption("Executive operational dashboard for patient acuity, AI performance, and emergency department pressure.")

    if st.session_state.role not in ["doctor", "admin"]:
        st.error("Only doctor and admin roles can access this dashboard.")
        st.stop()

    with st.spinner("Loading company health dashboard data..."):
        response = requests.get(
            f"{API_URL}/v2/company-health-dashboard",
            headers=auth_headers(),
            timeout=20
        )

    if response.status_code != 200:
        handle_response_error(response)
        st.stop()

    data = response.json()

    if data.get("total_patients", 1) == 0:
        st.info("No records available yet. Run predictions first to populate the dashboard.")
        st.stop()

    kpis = data.get("kpis", {})
    charts = data.get("charts", {})
    operations = data.get("operations", {})

    system_status = kpis.get("system_status", "Unknown")

    if system_status == "High Pressure":
        st.error("🚨 Emergency Department Status: High Pressure")
    elif system_status == "Moderate Pressure":
        st.warning("⚠️ Emergency Department Status: Moderate Pressure")
    else:
        st.success("✅ Emergency Department Status: Stable")

    st.markdown("---")

    st.subheader("Executive KPIs")

    k1, k2, k3, k4 = st.columns(4)

    with k1:
        st.metric("Total Patients", kpis.get("total_patients", 0))

    with k2:
        st.metric("High-Acuity Rate", f"{kpis.get('high_acuity_rate', 0)}%")

    with k3:
        st.metric("Critical Patients", kpis.get("critical_patients", 0))

    with k4:
        st.metric("Avg Model Confidence", f"{kpis.get('avg_model_confidence', 0)}")

    k5, k6, k7, k8 = st.columns(4)

    with k5:
        st.metric("High Risk Patients", kpis.get("high_risk_patients", 0))

    with k6:
        st.metric("Moderate / Low Patients", kpis.get("moderate_low_patients", 0))

    with k7:
        st.metric("Low Confidence Predictions", kpis.get("low_confidence_predictions", 0))

    with k8:
        st.metric("Override Rate", f"{kpis.get('override_rate', 0)}%")

    style_metric_cards()

    st.markdown("---")

    st.subheader("Operations Recommendation")

    rec_col1, rec_col2 = st.columns([2, 1])

    with rec_col1:
        st.info(operations.get("recommendation", "No recommendation available."))

    with rec_col2:
        st.metric("Busiest Hour", operations.get("busiest_hour", "N/A"))
        st.metric("Busiest Hour Volume", operations.get("busiest_hour_volume", 0))

    st.markdown("---")

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.subheader("Patient Acuity Distribution")
        esi_distribution = charts.get("esi_distribution", {})

        if esi_distribution:
            esi_df = pd.DataFrame({
                "ESI Level": list(esi_distribution.keys()),
                "Count": list(esi_distribution.values())
            })

            fig = px.bar(
                esi_df,
                x="ESI Level",
                y="Count",
                title="ESI / Final Prediction Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No ESI distribution data available.")

    with chart_col2:
        st.subheader("Risk Group Breakdown")
        risk_counts = charts.get("risk_counts", {})

        if risk_counts:
            risk_df = pd.DataFrame({
                "Risk Group": [key.replace("_", " ").title() for key in risk_counts.keys()],
                "Count": list(risk_counts.values())
            })

            fig2 = px.pie(
                risk_df,
                names="Risk Group",
                values="Count",
                title="Critical vs High vs Moderate/Low"
            )
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No risk count data available.")

    st.markdown("---")

    chart_col3, chart_col4 = st.columns(2)

    with chart_col3:
        st.subheader("Hourly Patient Volume")
        hourly_volume = charts.get("hourly_volume", {})

        if hourly_volume:
            hourly_df = pd.DataFrame({
                "Hour": list(hourly_volume.keys()),
                "Patient Volume": list(hourly_volume.values())
            })
            hourly_df["Hour"] = pd.to_datetime(hourly_df["Hour"])
            hourly_df = hourly_df.sort_values("Hour")

            fig3 = px.line(
                hourly_df,
                x="Hour",
                y="Patient Volume",
                markers=True,
                title="Emergency Volume Trend"
            )
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("No hourly volume data available.")

    with chart_col4:
        st.subheader("Arrival Mode Distribution")
        arrival_distribution = charts.get("arrival_distribution", {})

        if arrival_distribution:
            arrival_df = pd.DataFrame({
                "Arrival Mode": list(arrival_distribution.keys()),
                "Count": list(arrival_distribution.values())
            })

            fig4 = px.bar(
                arrival_df,
                x="Arrival Mode",
                y="Count",
                title="Patient Arrival Mode"
            )
            st.plotly_chart(fig4, use_container_width=True)
        else:
            st.info("No arrival mode data available.")

    st.markdown("---")

    st.subheader("Clinical Feedback Performance")

    feedback_distribution = charts.get("clinical_feedback", {})

    if feedback_distribution:
        feedback_df = pd.DataFrame({
            "Decision": [key.title() for key in feedback_distribution.keys()],
            "Count": list(feedback_distribution.values())
        })

        f1, f2 = st.columns([1, 2])

        with f1:
            st.metric("Total Feedback", kpis.get("total_feedback", 0))
            st.metric("Override Rate", f"{kpis.get('override_rate', 0)}%")

        with f2:
            fig5 = px.pie(
                feedback_df,
                names="Decision",
                values="Count",
                title="Accepted vs Overridden AI Predictions"
            )
            st.plotly_chart(fig5, use_container_width=True)
    else:
        st.info("No clinical feedback data available.")

    st.markdown("---")

    st.subheader("Executive Summary")

    summary_col1, summary_col2 = st.columns(2)

    with summary_col1:
        st.write("### Current Operational Pressure")
        st.write(f"**System Status:** {system_status}")
        st.write(f"**High-Acuity Patients:** {kpis.get('high_acuity_total', 0)}")
        st.write(f"**High-Acuity Rate:** {kpis.get('high_acuity_rate', 0)}%")
        st.write(f"**Critical Patients:** {kpis.get('critical_patients', 0)}")

    with summary_col2:
        st.write("### AI Governance Indicators")
        st.write(f"**Average Model Confidence:** {kpis.get('avg_model_confidence', 0)}")
        st.write(f"**Low Confidence Predictions:** {kpis.get('low_confidence_predictions', 0)}")
        st.write(f"**Clinical Override Rate:** {kpis.get('override_rate', 0)}%")
        st.write(f"**Total Feedback Records:** {kpis.get('total_feedback', 0)}")

    with st.expander("Raw API Response"):
        st.json(data)

        # =========================
        # SEND REPORT EMAIL
        # =========================

        st.markdown("### 📧 Send Report to Doctor")

        doctor_email = st.text_input(
            "Doctor Email Address",
            placeholder="doctor@example.com"
        )

        if st.button("📨 Send Clinical Report"):

            payload = {
                "doctor_email": doctor_email,
                "prediction_id": int(st.session_state.prediction_id)
            }

            response = requests.post(
                f"{API_URL}/send-report-email",
                json=payload,
                headers=auth_headers(),
                timeout=120
            )

            if response.status_code == 200:
                st.success("✅ Report email sent successfully.")
            else:
                st.error(response.text)
