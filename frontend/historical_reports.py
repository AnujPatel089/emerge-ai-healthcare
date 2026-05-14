"""
Historical Reports Streamlit page.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd
import requests
import streamlit as st
from PIL import Image

from src.image_assessment import IMAGE_EXTENSIONS, assess_medical_image
from src.report_text_extractor import extract_text_from_file
from src.medical_history_analyzer import analyze_medical_history
from src.risk_flag_engine import generate_image_risk_flags, generate_risk_flags
from src.clinical_summary_generator import generate_clinical_summary


DISCLAIMER = (
    "This tool analyzes uploaded historical medical reports and medical images for "
    "educational and decision-support purposes only. It does not provide a final "
    "diagnosis. A licensed healthcare professional must verify all findings."
)

REPORT_TYPES = [
    "Blood Test",
    "X-Ray",
    "MRI",
    "CT Scan",
    "Prescription",
    "Discharge Summary",
    "Previous Diagnosis",
    "Emergency Visit",
    "Surgery Report",
    "Allergy Report",
    "Medical Image",
    "Other",
]

RISK_COLORS = {
    "High Risk": "#dc2626",
    "Medium Risk": "#d97706",
    "Low Risk": "#16a34a",
    "Allergy Alert": "#2563eb",
    "Image Quality Warning": "#ca8a04",
    "Clinician Review Required": "#2563eb",
}


def _headers(token: str | None) -> dict:
    return {"Authorization": f"Bearer {token}"} if token else {}


def _risk_card(flag: dict) -> None:
    color = RISK_COLORS.get(flag.get("level"), "#64748b")
    st.markdown(
        f"""
        <div style="border-left:6px solid {color};background:#ffffff;border:1px solid #e5e7eb;
                    padding:12px 14px;border-radius:8px;margin-bottom:8px;">
            <div style="font-weight:900;color:{color};">{flag.get('label')}</div>
            <div style="font-size:13px;color:#475569;">{flag.get('level')} - {flag.get('reason')}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _summary_text(summary: dict) -> dict:
    if "clinical_summary" in summary:
        return summary.get("clinical_summary", {})
    return summary


def render_historical_reports_page(api_url: str, token: str | None, role: str, username: str) -> None:
    st.title("Historical Reports & Image Assessment")
    st.warning(DISCLAIMER)

    if role not in ["patient", "nurse", "doctor", "admin", "super_admin"]:
        st.error("Unauthorized access.")
        return

    if not token:
        render_local_patient_demo_reports(role, username)
        return

    upload_tab, history_tab = st.tabs(["Upload and Analyze", "Report History"])

    with upload_tab:
        st.subheader("Upload Historical Medical Report or Medical Image")
        with st.form("historical_report_upload"):
            c1, c2, c3 = st.columns(3)
            with c1:
                patient_id = st.text_input("Patient ID")
                patient_name = st.text_input("Patient Name")
            with c2:
                report_type = st.selectbox("Report Type", REPORT_TYPES)
                upload_date = st.date_input("Upload Date", value=date.today())
            with c3:
                notes = st.text_area("Notes")
            uploaded_file = st.file_uploader("Report or Image File", type=["pdf", "txt", "csv", "docx", "jpg", "jpeg", "png"])
            analyze = st.form_submit_button("Analyze Report/Image", use_container_width=True)

        if analyze:
            if not patient_id.strip() or not patient_name.strip() or uploaded_file is None:
                st.error("Patient ID, patient name, and report/image file are required.")
            else:
                suffix = Path(uploaded_file.name).suffix.lower()
                if suffix in IMAGE_EXTENSIONS:
                    st.subheader("Image Preview")
                    st.image(uploaded_file.getvalue(), caption=uploaded_file.name, use_container_width=True)
                files = {
                    "file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type or "application/octet-stream")
                }
                data = {
                    "patient_id": patient_id.strip(),
                    "patient_name": patient_name.strip(),
                    "report_type": report_type,
                    "notes": f"Upload date: {upload_date}; {notes or ''}",
                }
                try:
                    response = requests.post(
                        f"{api_url}/historical-reports/upload",
                        data=data,
                        files=files,
                        headers=_headers(token),
                        timeout=60,
                    )
                    if response.status_code != 200:
                        st.error(response.json().get("detail", response.text))
                    else:
                        report = response.json()["report"]
                        st.success("Report/image uploaded and analyzed.")
                        show_report_results(report)
                except Exception as exc:
                    st.error(f"Could not upload or analyze report/image: {exc}")

    with history_tab:
        st.subheader("Report History")
        f1, f2, f3 = st.columns(3)
        with f1:
            patient_filter = st.text_input("Filter by Patient ID")
        with f2:
            type_filter = st.selectbox("Filter by Report Type", ["All"] + REPORT_TYPES)
        with f3:
            risk_filter = st.selectbox("Filter by Risk Level", ["All", "High Risk", "Medium Risk", "Low Risk", "Allergy Alert", "Image Quality Warning", "Clinician Review Required"])

        params = {
            "patient_id": patient_filter or None,
            "report_type": type_filter,
            "risk_level": risk_filter,
        }
        try:
            response = requests.get(
                f"{api_url}/historical-reports",
                params={key: value for key, value in params.items() if value},
                headers=_headers(token),
                timeout=30,
            )
            if response.status_code != 200:
                st.error(response.json().get("detail", response.text))
                return
            reports = response.json().get("reports", [])
        except Exception as exc:
            st.error(f"Could not load report history: {exc}")
            return

        if not reports:
            st.info("No historical reports found.")
            return

        table_rows = []
        for report in reports:
            flags = report.get("risk_flags", [])
            table_rows.append({
                "ID": report.get("id"),
                "Patient ID": report.get("patient_id"),
                "Patient Name": report.get("patient_name"),
                "Report Type": report.get("report_type"),
                "File Type": report.get("file_type"),
                "Uploaded By": report.get("uploaded_by"),
                "Upload Date": report.get("upload_date"),
                "Risk Flags": ", ".join(flag.get("label", "") for flag in flags),
            })
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

        selected_id = st.number_input("View Details for Report ID", min_value=1, value=int(table_rows[0]["ID"]))
        if st.button("View Details", use_container_width=True):
            detail_response = requests.get(
                f"{api_url}/historical-reports/{int(selected_id)}",
                headers=_headers(token),
                timeout=30,
            )
            if detail_response.status_code == 200:
                show_report_results(detail_response.json()["report"], allow_notes=True, api_url=api_url, token=token, role=role)
            else:
                st.error(detail_response.json().get("detail", detail_response.text))

        if role in ["admin", "super_admin"]:
            delete_id = st.number_input("Delete Report ID", min_value=1, value=int(table_rows[0]["ID"]), key="delete_historical_report")
            if st.button("Delete Report", type="secondary"):
                delete_response = requests.delete(
                    f"{api_url}/historical-reports/{int(delete_id)}",
                    headers=_headers(token),
                    timeout=30,
                )
                if delete_response.status_code == 200:
                    st.success("Historical report deleted.")
                    st.rerun()
                else:
                    st.error(delete_response.json().get("detail", delete_response.text))


def show_report_results(report: dict, allow_notes: bool = False, api_url: str | None = None, token: str | None = None, role: str | None = None) -> None:
    summary = _summary_text(report.get("summary", {}))
    risk_flags = report.get("risk_flags", [])
    image_metadata = report.get("image_metadata") or {}
    image_quality_notes = report.get("image_quality_notes") or []
    ocr_text = report.get("ocr_text") or ""

    if report.get("file_type") == "image":
        st.subheader("Image Preview and Metadata")
        file_path = report.get("file_path")
        if file_path and Path(file_path).exists():
            st.image(str(file_path), caption=report.get("file_name"), use_container_width=True)
        if image_metadata:
            st.json(image_metadata)
        if image_quality_notes:
            st.warning("Quality issue detected: " + " | ".join(image_quality_notes))
        else:
            st.success("No basic image quality warning detected.")

    st.subheader("Risk Flags")
    for flag in risk_flags:
        _risk_card(flag)

    st.subheader("Doctor/Nurse Clinical Summary")
    st.write(summary.get("summary_text") or summary.get("patient_background") or "No summary available.")

    sections = {
        "Patient Background": summary.get("patient_background"),
        "Known Conditions": ", ".join(summary.get("known_conditions", [])) if isinstance(summary.get("known_conditions"), list) else summary.get("known_conditions"),
        "Current Risk Factors": ", ".join(summary.get("current_risk_factors", [])) if isinstance(summary.get("current_risk_factors"), list) else summary.get("current_risk_factors"),
        "Medication History": summary.get("medication_history"),
        "Allergy Alerts": summary.get("allergy_alerts"),
        "Image Quality Notes": " | ".join(summary.get("image_quality_notes", [])) if isinstance(summary.get("image_quality_notes"), list) else summary.get("image_quality_notes"),
        "OCR Findings": summary.get("ocr_findings"),
        "Previous Hospital Visits": summary.get("previous_hospital_visits"),
        "Possible Emergency Concerns": ", ".join(summary.get("possible_emergency_concerns", [])) if isinstance(summary.get("possible_emergency_concerns"), list) else summary.get("possible_emergency_concerns"),
        "Triage Notes": summary.get("triage_notes"),
    }
    for title, content in sections.items():
        with st.expander(title):
            st.write(content or "No keyword mention detected.")

    with st.expander("Recommended Questions for Doctor/Nurse"):
        for question in summary.get("recommended_questions", []):
            st.write(f"- {question}")

    with st.expander("Extracted Text / OCR Preview"):
        extracted = report.get("extracted_text") or ""
        st.text(extracted[:3000] if extracted else "No extracted text preview available.")
        if ocr_text:
            st.markdown("**OCR Text**")
            st.text(ocr_text[:3000])
        elif report.get("file_type") == "image":
            st.info("OCR is not available. Image preview and metadata analysis completed.")

    if allow_notes and api_url and role in ["nurse", "doctor", "admin", "super_admin"]:
        st.subheader("Clinical Notes")
        doctor_notes = st.text_area("Doctor Notes", value=report.get("doctor_notes") or "", disabled=role == "nurse")
        nurse_notes = st.text_area("Nurse Notes", value=report.get("nurse_notes") or "")
        if st.button("Save Notes", use_container_width=True):
            payload = {
                "doctor_notes": doctor_notes if role in ["doctor", "admin", "super_admin"] else None,
                "nurse_notes": nurse_notes,
            }
            response = requests.put(
                f"{api_url}/historical-reports/{report.get('id')}/notes",
                json=payload,
                headers=_headers(token),
                timeout=30,
            )
            if response.status_code == 200:
                st.success("Notes saved.")
            else:
                st.error(response.json().get("detail", response.text))


def render_local_patient_demo_reports(role: str, username: str) -> None:
    st.info("Running local Patient Demo Mode. Reports are analyzed locally and not saved to the backend database.")
    with st.form("local_patient_demo_history"):
        patient_id = st.text_input("Patient ID")
        patient_name = st.text_input("Patient Name", value=username if role == "patient" else "")
        report_type = st.selectbox("Report Type", REPORT_TYPES)
        notes = st.text_area("Notes")
        uploaded_file = st.file_uploader("Report or Image File", type=["pdf", "txt", "csv", "docx", "jpg", "jpeg", "png"])
        analyze = st.form_submit_button("Analyze Report/Image", use_container_width=True)

    if analyze:
        if uploaded_file is None or not patient_id.strip() or not patient_name.strip():
            st.error("Patient ID, patient name, and report/image file are required.")
            return
        out_dir = Path("reports/historical_reports")
        out_dir.mkdir(parents=True, exist_ok=True)
        safe_name = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in uploaded_file.name)
        path = out_dir / f"demo_{patient_id}_{safe_name}"
        path.write_bytes(uploaded_file.getvalue())
        suffix = path.suffix.lower()
        image_assessment = {}
        if suffix in IMAGE_EXTENSIONS:
            st.image(uploaded_file.getvalue(), caption=uploaded_file.name, use_container_width=True)
            image_assessment = assess_medical_image(path)
            text = image_assessment.get("ocr_text") or image_assessment.get("ocr_message", "")
        else:
            ok, text = extract_text_from_file(path)
            if not ok:
                st.error(text)
                return
        analysis = analyze_medical_history("\n".join([text, notes]))
        flags = generate_risk_flags(analysis)
        if image_assessment:
            flags.extend(generate_image_risk_flags(report_type, notes, image_assessment.get("ocr_text", ""), image_assessment.get("image_quality_notes", [])))
        summary = generate_clinical_summary(patient_name, report_type, analysis, flags, notes, image_assessment)
        report = {
            "id": "local-demo",
            "summary": {"clinical_summary": summary},
            "risk_flags": flags,
            "extracted_text": text,
            "ocr_text": image_assessment.get("ocr_text", ""),
            "image_metadata": image_assessment.get("image_metadata", {}),
            "image_quality_notes": image_assessment.get("image_quality_notes", []),
            "file_type": "image" if image_assessment else "document",
            "doctor_notes": "",
            "nurse_notes": "",
        }
        show_report_results(report)
