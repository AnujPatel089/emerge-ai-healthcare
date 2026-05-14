import pandas as pd
import streamlit as st

from frontend.api_client import api_request, json_or_error


def render_patient_timeline(auth_headers):
    st.title("Patient Journey Timeline")
    patient_id = st.text_input("Patient ID")
    if not patient_id:
        return
    data = json_or_error(api_request("GET", f"/api/patient-timeline/{patient_id}", headers=auth_headers()), {})
    events = data.get("events", []) if data else []
    if events:
        st.dataframe(pd.DataFrame(events), use_container_width=True)
    else:
        st.info("No timeline events found.")
    with st.expander("Add Timeline Event"):
        event_type = st.selectbox("Event Type", ["registration", "approval", "prediction", "queue_entry", "nurse_assignment", "triage_started", "upload_added", "sent_to_doctor", "doctor_review", "pdf_generated", "completed"])
        title = st.text_input("Title")
        desc = st.text_area("Description")
        if st.button("Add Event") and title:
            payload = {"patient_id": patient_id, "event_type": event_type, "event_title": title, "event_description": desc}
            st.json(json_or_error(api_request("POST", "/api/patient-timeline/event", headers=auth_headers(), json=payload), {}))
