"""Login UI helpers for the EmergAI Streamlit app.

The production login screen is rendered from frontend.app so session state,
API configuration, and navigation stay centralized.
"""

LOGIN_ROLE_OPTIONS = ["super_admin", "admin", "doctor", "nurse", "patient"]

LOGIN_ROLE_LABELS = {
    "super_admin": "Super Admin",
    "admin": "Hospital Admin",
    "doctor": "Emergency Doctor",
    "nurse": "Triage Nurse",
    "patient": "Patient",
}


def role_label(role: str) -> str:
    return LOGIN_ROLE_LABELS.get(role, role.replace("_", " ").title())
