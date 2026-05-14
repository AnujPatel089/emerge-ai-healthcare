"""
EmergAI Healthcare Streamlit dashboard.

This app can run independently from FastAPI. If the trained emergency model is
available in models/, it uses ML prediction; otherwise it falls back to the
clinical rule engine so the interface still works.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

from src.feature_engineering import add_medical_features, model_feature_columns
from src.auth_utils import (
    authenticate_local_user,
    can_approve_role,
    pending_local_approvals,
    register_local_user,
    set_local_account_status,
    visible_local_registrations,
)
from src.triage_rules import (
    ESI_COLORS,
    ESI_LABELS,
    estimate_icu_risk,
    estimate_readmission_risk,
    recommended_wait_time_minutes,
    rule_based_esi,
)
from src.report_text_extractor import extract_text_from_file
from src.image_assessment import IMAGE_EXTENSIONS, assess_medical_image
from src.medical_history_analyzer import analyze_medical_history
from src.risk_flag_engine import generate_image_risk_flags, generate_risk_flags
from src.clinical_summary_generator import DISCLAIMER as HISTORICAL_REPORT_DISCLAIMER, generate_clinical_summary


DATA_PATH = Path("data/emergency_synthetic_data.csv")
MODEL_PATH = Path("models/emergency_triage_model.pkl")
FEATURE_IMPORTANCE_PATH = Path("reports/emergency_feature_importance.png")
LOCAL_LOG_PATH = Path("logs/emergency_streamlit_predictions.csv")
HISTORICAL_REPORT_DIR = Path("reports/historical_reports")

DEMO_USERS = {
    "admin": {"password": "admin123", "role": "admin", "status": "active", "full_name": "Demo Admin"},
    "superadmin": {"password": "superadmin123", "role": "super_admin", "status": "active", "full_name": "Demo Super Admin"},
    "doctor": {"password": "doctor123", "role": "doctor", "status": "active", "full_name": "Demo Doctor"},
    "nurse": {"password": "nurse123", "role": "nurse", "status": "active", "full_name": "Demo Nurse"},
    "anuj": {"password": "anuj123", "role": "admin", "status": "active", "full_name": "Anuj Demo"},
    "chintan": {"password": "chintan123", "role": "admin", "status": "active", "full_name": "Chintan Demo"},
}

ROLE_TABS = {
    "super_admin": ["Analytics Dashboard", "Dataset Preview", "Reports", "Approvals", "Settings"],
    "admin": ["Patient Triage", "Historical Reports", "Assignment Queue", "Analytics Dashboard", "Dataset Preview", "Reports", "Approvals", "Settings"],
    "doctor": ["Patient Triage", "Historical Reports", "Assignment Queue", "Reports", "Approvals"],
    "nurse": ["Patient Triage", "Historical Reports", "Assignment Queue", "Reports", "Approvals"],
    "patient": ["Historical Reports", "Reports"],
}

ROLE_DISPLAY_NAMES = {
    "super_admin": "Super Admin",
    "admin": "Hospital Admin",
    "doctor": "Emergency Doctor",
    "nurse": "Triage Nurse",
    "patient": "Patient",
}


st.set_page_config(
    page_title="EmergAI Healthcare",
    page_icon="hospital",
    layout="wide",
)


for key, default in {
    "is_authenticated": False,
    "username": None,
    "role": None,
    "account_status": None,
    "auth_view": "login",
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


def logout() -> None:
    for key in ["is_authenticated", "username", "role", "account_status"]:
        st.session_state[key] = None if key != "is_authenticated" else False
    st.session_state.auth_view = "login"
    st.rerun()


def authenticate_demo_or_local(username: str, password: str, role: str) -> tuple[bool, str, dict | None]:
    user = DEMO_USERS.get(username.strip().lower())
    if user:
        if user["password"] != password:
            return False, "Invalid username or password.", None
        if user["role"] != role:
            return False, f"This demo account logs in as {user['role']}, not {role}.", None
        return True, "Login successful.", {
            "username": username.strip().lower(),
            "role": user["role"],
            "status": user["status"],
            "full_name": user["full_name"],
        }
    return authenticate_local_user(username, password, role)


def render_auth_page() -> None:
    st.title("EmergAI Healthcare")
    st.caption("Emergency triage AI, synthetic healthcare analytics, and role-based demo access.")
    st.warning("Educational project only. Not for real diagnosis, treatment, or emergency decisions.")

    if st.session_state.auth_view == "register":
        render_registration_page()
        return

    c1, c2 = st.columns([1.1, 0.9])
    with c1:
        st.subheader("Clinical AI Workspace")
        st.write(
            "Sign in as an approved user, register a new role-based account, "
            "or explore synthetic patient workflows in Patient Demo Mode."
        )
        st.info(
            "Patient Demo Mode is for educational and demonstration purposes only. "
            "Data shown is synthetic and not real patient information."
        )

    with c2:
        st.subheader("Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        role = st.selectbox("Role", ["super_admin", "admin", "doctor", "nurse", "patient"], format_func=lambda value: value.replace("_", " ").title())

        login_col, register_col = st.columns(2)
        with login_col:
            if st.button("Login", use_container_width=True):
                ok, message, user = authenticate_demo_or_local(username, password, role)
                if ok and user:
                    st.session_state.is_authenticated = True
                    st.session_state.username = user["full_name"] if user.get("full_name") else user["username"]
                    st.session_state.role = user["role"]
                    st.session_state.account_status = user["status"]
                    st.success(message)
                    st.rerun()
                else:
                    st.error(message)
        with register_col:
            if st.button("Register", use_container_width=True):
                st.session_state.auth_view = "register"
                st.rerun()

        if st.button("Continue as Patient", use_container_width=True):
            st.session_state.username = "Patient Demo User"
            st.session_state.role = "patient"
            st.session_state.account_status = "active"
            st.session_state.is_authenticated = True
            st.rerun()

        st.caption("Demo logins: admin/admin123, doctor/doctor123, nurse/nurse123.")


def render_registration_page() -> None:
    st.subheader("Create Account")
    with st.form("registration_form"):
        full_name = st.text_input("Full Name")
        username = st.text_input("Username")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")
        requested_role = st.selectbox("Requested Role", ["patient", "doctor", "nurse", "admin"])
        submitted = st.form_submit_button("Submit Registration", use_container_width=True)

    if submitted:
        if not all([full_name.strip(), username.strip(), email.strip(), password, confirm_password]):
            st.error("Required fields cannot be empty.")
        elif password != confirm_password:
            st.error("Password and confirm password must match.")
        else:
            ok, message = register_local_user(full_name, username, email, password, requested_role)
            if ok:
                st.warning(message)
                st.session_state.auth_view = "login"
            else:
                st.error(message)

    if st.button("Back to Login"):
        st.session_state.auth_view = "login"
        st.rerun()


def render_sidebar() -> None:
    st.sidebar.markdown("### Session")
    st.sidebar.write(f"Logged in as: **{st.session_state.username}**")
    st.sidebar.write(f"Role: **{ROLE_DISPLAY_NAMES.get(st.session_state.role, st.session_state.role)}**")
    st.sidebar.write(f"Account status: **{st.session_state.account_status}**")
    if st.session_state.role == "nurse":
        pending_count = len(visible_local_registrations("nurse"))
    else:
        pending_count = len(pending_local_approvals(str(st.session_state.role)))
    if pending_count:
        label = "New patients" if st.session_state.role == "nurse" else "Pending approvals"
        st.sidebar.warning(f"{label}: {pending_count}")
    if st.session_state.role == "patient":
        st.sidebar.info(
            "Patient Demo Mode is for educational and demonstration purposes only. "
            "Data shown is synthetic and not real patient information."
        )
    if st.sidebar.button("Logout", use_container_width=True):
        logout()


def status_badge(status: str) -> str:
    colors = {
        "pending": "#ca8a04",
        "active": "#16a34a",
        "rejected": "#dc2626",
        "suspended": "#6b7280",
    }
    color = colors.get(status, "#64748b")
    return f"<span style='background:{color};color:white;padding:4px 9px;border-radius:999px;font-weight:800;'>{status}</span>"


def load_model():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    return None


def load_dataset() -> pd.DataFrame:
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH)
    return pd.DataFrame()


def esi_badge(level: int) -> str:
    color = ESI_COLORS.get(level, "#64748b")
    label = ESI_LABELS.get(level, "Unknown")
    return (
        f"<div style='background:{color};color:white;padding:16px 18px;"
        f"border-radius:8px;font-size:24px;font-weight:800;text-align:center;'>"
        f"ESI {level}: {label}</div>"
    )


def risk_label(value: float) -> str:
    if value >= 0.5:
        return "High"
    if value >= 0.25:
        return "Moderate"
    return "Low"


def build_patient_frame(patient: dict) -> pd.DataFrame:
    row = {
        "Patient_ID": "FORM-PATIENT",
        "Age": patient["Age"],
        "Gender": patient["Gender"],
        "Systolic_BP": patient["Systolic_BP"],
        "Diastolic_BP": patient["Diastolic_BP"],
        "Heart_Rate": patient["Heart_Rate"],
        "Respiratory_Rate": patient["Respiratory_Rate"],
        "Oxygen_Saturation": patient["Oxygen_Saturation"],
        "Temperature": patient["Temperature"],
        "Diabetes": patient["Diabetes"],
        "Hypertension": patient["Hypertension"],
        "Smoking": patient["Smoking"],
        "Chest_Pain": patient["Chest_Pain"],
        "Shortness_of_Breath": patient["Shortness_of_Breath"],
        "Fever": patient["Fever"],
        "Symptom_Description": patient["Symptom_Description"],
        "Triage_Level": 3,
        "ICU_Required": 0,
        "Wait_Time_Minutes": 45,
        "Hospital_Stay_Days": 1.0,
        "Readmission_Risk": 0.1,
    }
    return add_medical_features(pd.DataFrame([row]))


def predict_patient(patient: dict) -> dict:
    model = load_model()
    rule_esi, reasons = rule_based_esi(patient)
    input_df = build_patient_frame(patient)

    if model is not None:
        model_features = input_df[model_feature_columns()]
        raw_prediction = int(model.predict(model_features)[0])
        classes_seen = list(getattr(model.named_steps["classifier"], "classes_", []))
        uses_zero_based_classes = bool(classes_seen) and min(int(cls) for cls in classes_seen) == 0
        model_esi = raw_prediction + 1 if uses_zero_based_classes else raw_prediction
        probabilities = {}
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(model_features)[0]
            classes = classes_seen or list(range(len(probs)))
            for cls, prob in zip(classes, probs):
                cls_int = int(cls) + 1 if uses_zero_based_classes else int(cls)
                probabilities[f"ESI {cls_int}"] = float(prob)
        prediction_source = "ML model with clinical safety rules"
    else:
        model_esi = rule_esi
        probabilities = {}
        prediction_source = "Clinical rule fallback"

    final_esi = min(int(model_esi), int(rule_esi))
    icu_risk = estimate_icu_risk(patient, final_esi)
    readmission_risk = estimate_readmission_risk(patient, final_esi, icu_risk)

    return {
        "model_esi": int(model_esi),
        "rule_esi": int(rule_esi),
        "final_esi": int(final_esi),
        "icu_risk": icu_risk,
        "icu_required": int(icu_risk >= 0.35),
        "readmission_risk": readmission_risk,
        "wait_time": recommended_wait_time_minutes(final_esi),
        "reasons": reasons,
        "probabilities": probabilities,
        "source": prediction_source,
    }


def save_local_prediction(patient: dict, result: dict) -> None:
    LOCAL_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        **patient,
        "model_esi": result["model_esi"],
        "rule_esi": result["rule_esi"],
        "final_esi": result["final_esi"],
        "icu_risk": result["icu_risk"],
        "readmission_risk": result["readmission_risk"],
        "wait_time": result["wait_time"],
        "source": result["source"],
    }
    log_df = pd.DataFrame([row])
    if LOCAL_LOG_PATH.exists():
        log_df.to_csv(LOCAL_LOG_PATH, mode="a", header=False, index=False)
    else:
        log_df.to_csv(LOCAL_LOG_PATH, index=False)


def render_dataset_charts(df: pd.DataFrame) -> None:
    if df.empty:
        st.info("Generate the synthetic dataset first: `python -m src.generate_emergency_data`")
        return

    st.subheader("Emergency Department Analytics")
    c1, c2 = st.columns(2)

    color_map = {str(level): color for level, color in ESI_COLORS.items()}
    df_chart = df.copy()
    df_chart["ESI"] = df_chart["Triage_Level"].astype(str)

    with c1:
        esi_counts = df_chart["ESI"].value_counts().sort_index().reset_index()
        esi_counts.columns = ["ESI", "Patients"]
        fig = px.bar(
            esi_counts,
            x="ESI",
            y="Patients",
            color="ESI",
            color_discrete_map=color_map,
            title="ESI Distribution",
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        icu_counts = df["ICU_Required"].map({0: "No ICU", 1: "ICU Required"}).value_counts().reset_index()
        icu_counts.columns = ["Status", "Patients"]
        fig = px.pie(icu_counts, names="Status", values="Patients", title="ICU Required Distribution")
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        wait_df = df.groupby("Triage_Level", as_index=False)["Wait_Time_Minutes"].mean()
        wait_df["Triage_Level"] = wait_df["Triage_Level"].astype(str)
        fig = px.bar(
            wait_df,
            x="Triage_Level",
            y="Wait_Time_Minutes",
            color="Triage_Level",
            color_discrete_map=color_map,
            title="Average Wait Time by Triage Level",
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with c4:
        age_bins = pd.cut(
            df["Age"],
            bins=[0, 18, 35, 50, 65, 80, 100],
            labels=["0-18", "19-35", "36-50", "51-65", "66-80", "81+"],
        )
        risk_df = df.assign(Age_Group=age_bins).groupby("Age_Group", observed=True)["Readmission_Risk"].mean().reset_index()
        fig = px.line(
            risk_df,
            x="Age_Group",
            y="Readmission_Risk",
            markers=True,
            title="Readmission Risk by Age Group",
        )
        st.plotly_chart(fig, use_container_width=True)

    if FEATURE_IMPORTANCE_PATH.exists():
        st.subheader("Model Feature Importance")
        st.image(str(FEATURE_IMPORTANCE_PATH), use_container_width=True)
    else:
        st.info("Train the model to create feature importance: `python -m src.train_emergency_model`")


def render_local_historical_reports() -> None:
    st.subheader("Historical Reports")
    st.warning(HISTORICAL_REPORT_DISCLAIMER)
    HISTORICAL_REPORT_DIR.mkdir(parents=True, exist_ok=True)

    report_types = [
        "Blood Test", "X-Ray", "MRI", "CT Scan", "Prescription", "Discharge Summary",
        "Previous Diagnosis", "Emergency Visit", "Surgery Report", "Allergy Report", "Medical Image", "Other"
    ]
    with st.form("local_historical_upload"):
        c1, c2 = st.columns(2)
        with c1:
            patient_id = st.text_input("Patient ID")
            patient_name = st.text_input("Patient Name")
            report_type = st.selectbox("Report Type", report_types)
        with c2:
            st.date_input("Upload Date")
            notes = st.text_area("Notes")
        uploaded_file = st.file_uploader("Upload PDF, TXT, CSV, DOCX, JPG, JPEG, or PNG", type=["pdf", "txt", "csv", "docx", "jpg", "jpeg", "png"])
        analyze = st.form_submit_button("Analyze Report/Image", use_container_width=True)

    if analyze:
        if uploaded_file is None or not patient_id.strip() or not patient_name.strip():
            st.error("Patient ID, patient name, and file are required.")
            return
        safe_name = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in uploaded_file.name)
        output_path = HISTORICAL_REPORT_DIR / f"{patient_id}_{safe_name}"
        output_path.write_bytes(uploaded_file.getvalue())
        image_assessment = {}
        if output_path.suffix.lower() in IMAGE_EXTENSIONS:
            st.subheader("Image Preview")
            st.image(uploaded_file.getvalue(), caption=uploaded_file.name, use_container_width=True)
            image_assessment = assess_medical_image(output_path)
            text = image_assessment.get("ocr_text") or image_assessment.get("ocr_message", "")
        else:
            ok, text = extract_text_from_file(output_path)
            if not ok:
                st.error(text)
                return
        analysis = analyze_medical_history("\n".join([text, notes]))
        flags = generate_risk_flags(analysis)
        if image_assessment:
            flags.extend(generate_image_risk_flags(report_type, notes, image_assessment.get("ocr_text", ""), image_assessment.get("image_quality_notes", [])))
        summary = generate_clinical_summary(patient_name, report_type, analysis, flags, notes, image_assessment)

        st.success("Report analyzed.")
        st.subheader("Risk Flags")
        color_map = {"High Risk": "#dc2626", "Medium Risk": "#d97706", "Low Risk": "#16a34a", "Allergy Alert": "#2563eb", "Image Quality Warning": "#ca8a04", "Clinician Review Required": "#2563eb"}
        for flag in flags:
            color = color_map.get(flag["level"], "#64748b")
            st.markdown(
                f"<div style='border-left:6px solid {color};border:1px solid #e5e7eb;padding:12px;border-radius:8px;margin-bottom:8px;'>"
                f"<b style='color:{color};'>{flag['label']}</b><br>{flag['level']} - {flag['reason']}</div>",
                unsafe_allow_html=True,
            )

        st.subheader("Doctor/Nurse Summary")
        st.write(summary["summary_text"])
        if image_assessment:
            st.subheader("Image Metadata and Quality Notes")
            st.json(image_assessment.get("image_metadata", {}))
            notes_list = image_assessment.get("image_quality_notes", [])
            if notes_list:
                st.warning(" | ".join(notes_list))
            else:
                st.success("No basic image quality warning detected.")
            if image_assessment.get("ocr_text"):
                st.text_area("OCR Text", image_assessment["ocr_text"], height=140)
            elif image_assessment.get("ocr_message"):
                st.info(image_assessment["ocr_message"])
        with st.expander("Detected Medical Conditions"):
            st.json(analysis["detected_conditions"])
        with st.expander("Recommended Questions"):
            for question in summary["recommended_questions"]:
                st.write(f"- {question}")
        with st.expander("Extracted Text Preview"):
            st.text(text[:3000])


if not st.session_state.is_authenticated:
    render_auth_page()
    st.stop()

render_sidebar()

st.title("EmergAI Healthcare Emergency Triage")
st.caption("Synthetic data, ESI classification, ICU risk, readmission risk, and emergency department analytics.")
st.warning("Educational project only. Not for real diagnosis, treatment, or emergency decision-making.")

dataset = load_dataset()

allowed_tabs = ROLE_TABS.get(st.session_state.role, ["Patient Triage"])
tabs = st.tabs(allowed_tabs)
tab_lookup = dict(zip(allowed_tabs, tabs))

if "Patient Triage" in tab_lookup:
    tab_predict = tab_lookup["Patient Triage"]
else:
    tab_predict = None

if tab_predict:
 with tab_predict:
    st.subheader("Patient Input")

    with st.form("emergency_patient_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.number_input("Age", 1, 100, 55)
            gender = st.selectbox("Gender", ["Female", "Male", "Nonbinary"])
            diabetes = st.checkbox("Diabetes")
            hypertension = st.checkbox("Hypertension")
            smoking = st.checkbox("Smoking")
        with col2:
            sbp = st.number_input("Systolic BP", 65, 230, 118)
            dbp = st.number_input("Diastolic BP", 35, 135, 76)
            heart_rate = st.number_input("Heart Rate", 35, 190, 96)
            respiratory_rate = st.number_input("Respiratory Rate", 6, 50, 20)
            oxygen = st.number_input("Oxygen Saturation", 70, 100, 96)
            temperature = st.number_input("Temperature (F)", 95.0, 106.0, 98.6, step=0.1)
        with col3:
            chest_pain = st.checkbox("Chest Pain")
            shortness_of_breath = st.checkbox("Shortness of Breath")
            fever = st.checkbox("Fever")
            symptom_description = st.text_area(
                "Symptom Description",
                value="Patient reports worsening symptoms with limited relief at home.",
                height=150,
            )

        submitted = st.form_submit_button("Predict ESI and Risk", use_container_width=True)

    if submitted:
        patient = {
            "Age": int(age),
            "Gender": gender,
            "Systolic_BP": int(sbp),
            "Diastolic_BP": int(dbp),
            "Heart_Rate": int(heart_rate),
            "Respiratory_Rate": int(respiratory_rate),
            "Oxygen_Saturation": int(oxygen),
            "Temperature": float(temperature),
            "Diabetes": int(diabetes),
            "Hypertension": int(hypertension),
            "Smoking": int(smoking),
            "Chest_Pain": int(chest_pain),
            "Shortness_of_Breath": int(shortness_of_breath),
            "Fever": int(fever),
            "Symptom_Description": symptom_description,
        }
        result = predict_patient(patient)
        save_local_prediction(patient, result)

        st.markdown(esi_badge(result["final_esi"]), unsafe_allow_html=True)
        st.caption(result["source"])

        if result["final_esi"] == 1 or result["icu_risk"] >= 0.5:
            st.error("Emergency alert: critical physiology or high ICU risk detected. Escalate immediately in a real clinical setting.")
        elif result["final_esi"] == 2:
            st.warning("High-priority emergency case. Recommended rapid clinical review.")

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Model ESI", result["model_esi"])
        k2.metric("ICU Risk", f"{result['icu_risk']:.1%}", risk_label(result["icu_risk"]))
        k3.metric("Readmission Risk", f"{result['readmission_risk']:.1%}", risk_label(result["readmission_risk"]))
        k4.metric("Target Wait", f"{result['wait_time']} min")

        st.subheader("Clinical Rule Reasons")
        for reason in result["reasons"]:
            st.write(f"- {reason}")

        if result["probabilities"]:
            prob_df = pd.DataFrame(
                {"ESI Level": list(result["probabilities"].keys()), "Probability": list(result["probabilities"].values())}
            )
            fig = px.bar(prob_df, x="ESI Level", y="Probability", title="Model Probability by ESI")
            st.plotly_chart(fig, use_container_width=True)

if "Analytics Dashboard" in tab_lookup:
    with tab_lookup["Analytics Dashboard"]:
        render_dataset_charts(dataset)

if "Historical Reports" in tab_lookup:
    with tab_lookup["Historical Reports"]:
        render_local_historical_reports()

if "Dataset Preview" in tab_lookup:
    with tab_lookup["Dataset Preview"]:
        if dataset.empty:
            st.info("No synthetic emergency dataset found yet.")
        else:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Synthetic Patients", f"{len(dataset):,}")
            c2.metric("ESI 1-2", f"{(dataset['Triage_Level'] <= 2).mean():.1%}")
            c3.metric("ICU Required", f"{dataset['ICU_Required'].mean():.1%}")
            c4.metric("Avg Wait", f"{dataset['Wait_Time_Minutes'].mean():.0f} min")
            st.dataframe(dataset.head(100), use_container_width=True, hide_index=True)

if "Reports" in tab_lookup:
    with tab_lookup["Reports"]:
        st.subheader("Demo Report Generation")
        st.write("Generate a synthetic summary report from the current emergency dataset.")
        if dataset.empty:
            st.info("Generate synthetic data first.")
        else:
            report_text = (
                "EmergAI Demo Report\n"
                "===================\n"
                f"Generated for: {st.session_state.username}\n"
                f"Role: {st.session_state.role}\n"
                f"Synthetic patients: {len(dataset):,}\n"
                f"High acuity rate: {(dataset['Triage_Level'] <= 2).mean():.1%}\n"
                f"ICU required rate: {dataset['ICU_Required'].mean():.1%}\n"
                f"Average wait time: {dataset['Wait_Time_Minutes'].mean():.1f} minutes\n\n"
                "Disclaimer: synthetic educational data only, not real patient information."
            )
            st.code(report_text, language="text")
            st.download_button("Download Demo Report", report_text, file_name="emergeai_demo_report.txt")

if "Assignment Queue" in tab_lookup:
    with tab_lookup["Assignment Queue"]:
        st.subheader("Patient-Nurse Assignment Queue")
        st.caption("Standalone demo view. Full auto-assignment and reassignment APIs run in `frontend/app.py` with FastAPI.")
        if LOCAL_LOG_PATH.exists():
            queue_df = pd.read_csv(LOCAL_LOG_PATH).tail(50).copy()
            queue_df["ESI Level"] = queue_df["final_esi"]
            queue_df["Priority Level"] = queue_df["ESI Level"].map({1: "Critical", 2: "High", 3: "Medium", 4: "Low", 5: "Low"})
            queue_df["Assignment Status"] = queue_df["Priority Level"].map(lambda value: "critical" if value == "Critical" else "waiting")
            queue_df["Assigned Nurse"] = "Use FastAPI assignment workflow"
            st.dataframe(
                queue_df[["created_at", "Age", "Gender", "ESI Level", "icu_risk", "Priority Level", "Assignment Status", "Assigned Nurse"]],
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("Run a triage prediction first to populate the local queue.")

if "Approvals" in tab_lookup:
    with tab_lookup["Approvals"]:
        if st.session_state.role not in ["super_admin", "admin", "doctor", "nurse"]:
            st.error("Unauthorized access.")
        else:
            st.subheader("Approval Management")
            pending = pending_local_approvals(str(st.session_state.role))
            if st.session_state.role == "nurse":
                new_patients = [
                    user for user in visible_local_registrations("nurse")
                    if (user.get("requested_role") or user.get("role")) == "patient"
                ]
                st.subheader("New Patients")
                if not new_patients:
                    st.success("No newly registered patients.")
                else:
                    rows = [
                        {
                            "Patient ID": user.get("id"),
                            "Full Name": user.get("full_name"),
                            "Username": user.get("username"),
                            "Email": user.get("email"),
                            "Registration Date": user.get("created_at"),
                            "Account Status": user.get("account_status") or user.get("status"),
                        }
                        for user in new_patients
                    ]
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                    for user in new_patients:
                        with st.expander(f"View Details: {user.get('full_name')}"):
                            st.json({k: v for k, v in user.items() if k != "password_hash"})
                st.info("Nurses can view new patient registrations but cannot approve or reject accounts.")
            elif not pending:
                st.success("No pending approval requests.")
            else:
                for user in pending:
                    with st.container(border=True):
                        st.write(f"**{user['full_name']}**")
                        st.write(f"User ID: `{user.get('id')}`")
                        st.write(f"Username: `{user['username']}`")
                        st.write(f"Email: `{user['email']}`")
                        st.write(f"Requested role: `{user.get('requested_role') or user['role']}`")
                        st.write(f"Registration date: `{user['created_at']}`")
                        st.markdown(f"Status: {status_badge(user.get('account_status') or user.get('status'))}", unsafe_allow_html=True)
                        st.write(f"Approved by: `{user.get('approved_by')}`")
                        st.write(f"Approved at: `{user.get('approved_at')}`")
                        with st.expander("View Details"):
                            st.json({k: v for k, v in user.items() if k != "password_hash"})
                        approve_col, reject_col = st.columns(2)
                        with approve_col:
                            if st.button("Approve", key=f"approve_{user['username']}"):
                                set_local_account_status(user["username"], "active", str(st.session_state.username), approver_role=str(st.session_state.role))
                                st.success(f"Approved {user['username']}.")
                                st.rerun()
                        with reject_col:
                            if st.button("Reject", key=f"reject_{user['username']}"):
                                set_local_account_status(user["username"], "rejected", str(st.session_state.username), "Rejected in local approval panel", str(st.session_state.role))
                                st.warning(f"Rejected {user['username']}.")
                                st.rerun()

if "Settings" in tab_lookup:
    with tab_lookup["Settings"]:
        if st.session_state.role != "admin":
            st.error("Unauthorized access.")
        else:
            st.subheader("Settings")
            st.info("Admin-only settings placeholder for demo configuration and governance notes.")
