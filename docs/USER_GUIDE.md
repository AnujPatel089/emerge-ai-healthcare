# User Guide

## Start the System

Backend:

```powershell
cd backend
uvicorn main:app --host 0.0.0.0 --port 9000
```

Frontend:

```powershell
streamlit run frontend/app.py
```

Set `API_URL` if the backend is not local:

```powershell
$env:API_URL="https://your-backend.example.com"
```

## Login

Demo roles are available for development:

- Admin: `admin`
- Doctor: `doctor`
- Nurse: `nurse`

Use the project password values configured in `src/auth.py` for local demo accounts. For production, replace demo auth with database-backed hashed passwords.

## Admin Workflow

1. Open Admin pages from the sidebar.
2. Review Admin Overview metrics.
3. Manage staff status and departments.
4. Create shifts.
5. Review billing, inventory, incidents, and hospital analytics.

## Doctor Workflow

1. Run triage prediction or review existing patient history.
2. Open SHAP Explainability to understand model factors.
3. Assign a nurse to a prediction record.
4. Add a Doctor Review with diagnosis and treatment plan.
5. Order labs or referrals when needed.
6. Assign beds and create discharge summaries.

## Nurse Workflow

1. View assigned patients.
2. Add vitals in Nurse Patient Care.
3. Create or update nurse tasks.
4. Update medication status.
5. Review alerts, bed status, and limited patient history.

## Patient Flow

1. Register patient.
2. Run triage prediction.
3. Track status through waiting, nurse care, doctor review, treatment, admitted, or discharged.
4. Capture vitals, tasks, medications, and doctor review.
5. Generate discharge PDF when care is complete.

## Safety Reminder

EmergeAI Healthcare is an educational project. AI predictions and explanations are demonstrations only and do not replace clinician judgment.
