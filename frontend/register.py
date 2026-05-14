"""Registration UI helpers for the EmergAI Streamlit app."""

REGISTERABLE_ROLES = ["patient", "doctor", "nurse", "admin"]

REGISTER_ROLE_LABELS = {
    "patient": "Patient",
    "doctor": "Emergency Doctor",
    "nurse": "Triage Nurse",
    "admin": "Hospital Admin",
}


def registration_role_label(role: str) -> str:
    return REGISTER_ROLE_LABELS.get(role, role.replace("_", " ").title())
