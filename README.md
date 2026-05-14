# EmergeAI Healthcare - AI Emergency Triage & Hospital Management System

EmergeAI Healthcare is an educational capstone project that combines emergency department triage, AI-assisted clinical decision support, and hospital workflow management in one full-stack Python application.

The project includes:

- A FastAPI backend for authentication, prediction, workflow APIs, reports, and analytics.
- A Streamlit frontend for doctors, nurses, and admins.
- PostgreSQL persistence through SQLAlchemy models.
- XGBoost-based ESI-style emergency triage prediction.
- NLP symptom extraction from patient descriptions.
- Classical computer vision and optional deep-learning image analysis.
- SHAP explainability for model interpretation.
- PDF report generation and optional email delivery.
- Hospital operations workflows such as patients, beds, nurses, shifts, labs, medications, billing, inventory, incidents, and alerts.

> Important medical disclaimer: This project is for learning, demonstration, and portfolio use only. It is not a medical device. It must not be used for real diagnosis, treatment, emergency decisions, or clinical care.

## Table of Contents

- [Project Goals](#project-goals)
- [How The System Works](#how-the-system-works)
- [Main Features](#main-features)
- [Role-Based Access](#role-based-access)
- [Technology Stack](#technology-stack)
- [Setup](#setup)
- [Running The App](#running-the-app)
- [Demo Login Accounts](#demo-login-accounts)
- [Important API Areas](#important-api-areas)
- [Database Tables](#database-tables)
- [AI And ML Components](#ai-and-ml-components)
- [Synthetic Emergency Dataset And ESI Upgrade](#synthetic-emergency-dataset-and-esi-upgrade)
- [Patient Demo Mode Registration And Admin Approval](#patient-demo-mode-registration-and-admin-approval)
- [Hierarchical Approval System](#hierarchical-approval-system)
- [Historical Medical Report Upload](#historical-medical-report-upload)
- [Full Folder Structure](#full-folder-structure)
- [Important Files Explained](#important-files-explained)
- [Testing Workflow](#testing-workflow)
- [Deployment Notes](#deployment-notes)
- [Known Limitations](#known-limitations)

## Project Goals

Emergency departments need quick prioritization, clear patient documentation, and smooth coordination between doctors, nurses, and administrators. This project demonstrates how a hospital-facing system can combine:

- AI triage prediction.
- Rule-based safety checks.
- Human clinical feedback.
- Explainable AI.
- Patient and staff workflow tracking.
- Operational analytics.
- Report generation.

The app is designed as a complete learning project, not just a standalone ML script.

## How The System Works

1. A user logs in through the Streamlit frontend as `admin`, `doctor`, or `nurse`.
2. The frontend sends authenticated API requests to the FastAPI backend.
3. The backend validates role permissions using JWT authentication.
4. Patient demographics, vitals, symptoms, text notes, or images are submitted.
5. The triage model predicts an ESI-style severity level.
6. Safety rules adjust or explain the prediction when dangerous vitals or symptoms are present.
7. The prediction, confidence, reasons, and explanations are saved to PostgreSQL.
8. Staff can assign nurses, add vitals, create doctor reviews, update patient status, manage beds, order labs, create alerts, and generate discharge summaries.
9. Dashboards and analytics summarize patient flow, high-risk cases, acuity, workload, and AI feedback.

## Main Features

### Authentication And Security

- JWT login through `/login`.
- Role-based access control for `admin`, `doctor`, and `nurse`.
- Protected backend endpoints using FastAPI dependencies.
- Staff activation/deactivation support for admins.
- Audit logs for important actions such as login, prediction, and workflow changes.
- Environment-based configuration using `.env`.

### AI Emergency Triage

- ESI-style triage prediction using trained XGBoost model files.
- New modular Emergency Severity Index classifier:
  - ESI 1 = Immediate / Critical.
  - ESI 2 = Emergency.
  - ESI 3 = Urgent.
  - ESI 4 = Semi-Urgent.
  - ESI 5 = Non-Urgent.
- Input support for:
  - Age.
  - Gender.
  - Race.
  - Ethnicity.
  - Arrival mode.
  - Heart rate.
  - Blood pressure.
  - Respiratory rate.
  - Oxygen saturation.
  - Temperature.
  - Chief complaint flags.
- Prediction confidence output.
- Safety rules that can highlight dangerous symptoms or vital signs.
- ICU risk and readmission risk estimates in the enhanced Streamlit dashboard.
- Color-coded triage badges and critical-patient alert cards.
- Clinical explanation text for each prediction.
- Prediction history saved to PostgreSQL.

### NLP Symptom Extraction

- Extracts symptoms from free-text patient descriptions.
- Cleans text and creates an LLM-ready summary field.
- Detects matched symptom terms.
- Detects emergency keywords.
- Helps convert patient descriptions into structured triage context.

### Computer Vision And Image Analysis

- Classical image analysis endpoint through `/analyze-image`.
- Deep-learning image analysis endpoint through `/v2/analyze-image-dl`.
- Supports uploaded medical or wound images.
- Returns wound class, infection severity, severity score, confidence values, and image metadata when available.
- Image analysis can be linked into multi-modal triage.

### Multi-Modal Triage

- `/v2/triage` can combine:
  - Existing prediction log data.
  - Inline vitals.
  - Structured symptoms.
  - NLP findings.
  - Linked image analysis.
- Produces:
  - ESI level.
  - ESI label.
  - Composite risk score.
  - Component-level contributions.
  - Red flags.
  - Saved triage log.

### SHAP Explainability

- SHAP visualization support for model interpretation.
- `/shap` generates explanation output.
- `/shap/image` returns the latest SHAP image.
- SHAP assets are saved under `shap_outputs/`.

### Human-In-The-Loop Feedback

- Doctors and admins can accept or override AI predictions.
- Overrides can include:
  - New clinician-selected ESI level.
  - Clinical notes.
  - Override reason.
- Feedback history supports AI governance and model monitoring.

### Patient Management

- Add patient profiles.
- View all patients.
- View one patient by ID.
- Update patient demographics and history.
- Link patients to prediction records.
- Store allergies, chronic diseases, past medical history, emergency contacts, phone, and address.

### Nurse Workflow

- Add nurses.
- List nurses.
- Assign nurses to prediction records.
- View all nurse assignments.
- View assignments for a specific prediction.
- Record nurse vitals.
- Create nurse tasks.
- Track task priority and completion status.
- Support assigned patient care workflows.

### Doctor Workflow

- Add doctor review.
- View doctor review by prediction.
- Record diagnosis.
- Record treatment plan.
- Add medication notes.
- Mark follow-up requirement.
- Track admit status.
- Create lab orders.
- Create specialist referrals.
- Create discharge summaries and PDFs.

### Emergency Queue

- View active emergency queue.
- Prioritize cases.
- Track estimated wait time.
- Mark critical status.
- Link queue records to prediction logs.

### Bed Management

- Add hospital beds.
- View all beds.
- Assign available beds to patients/predictions.
- Release occupied beds.
- Track ward type and occupancy.

### Medication Records

- Add medication records.
- Track dosage, route, scheduled time, status, side effects, and notes.
- Update medication status as scheduled, given, missed, or cancelled.

### Clinical Alerts

- Add alerts for prediction records.
- View active alerts.
- Resolve alerts.
- Track alert type, severity, message, and resolution time.

### Discharge Summary And Reports

- Create discharge summaries.
- View discharge summaries by prediction.
- Download discharge summary PDF.
- Generate v2 clinical reports using `/v2/report/{log_id}`.
- Send report emails through `/send-report-email` when SMTP is configured.

### Staff And Admin Operations

- View staff directory.
- Update staff department and active status.
- View workload summaries.
- Manage staff shifts.
- View audit logs.
- Access admin monitoring dashboard.

### Hospital Operations

- Lab test orders and lab result updates.
- Specialist referrals and referral status updates.
- Patient consent records.
- Incident reports and incident status tracking.
- Billing records and payment status updates.
- Inventory records, item updates, and low-stock view.
- Follow-up appointments.
- Family or contact notifications.

### Analytics And Dashboards

- Hospital summary analytics.
- Company-level health dashboard.
- High-risk patient watchlist.
- Admin monitoring dashboard.
- Dashboard summary by time window.
- Charts for acuity, arrival mode, risk groups, hourly volume, feedback, and workload.

## Role-Based Access

### Admin

Admins have the widest access. They can manage staff, audit logs, workload, shifts, inventory, billing, incidents, dashboards, and most clinical workflows.

Typical admin pages:

- Admin Overview.
- Staff Management.
- Admin Monitor.
- Company Health Dashboard.
- Shift Management.
- Billing.
- Inventory Management.
- Hospital Analytics.
- Incident Reports.

### Doctor

Doctors focus on clinical review, triage decisions, patient history, explainability, treatment, referrals, labs, and discharge.

Typical doctor pages:

- Live Prediction.
- SHAP Explainability.
- Patient History.
- Doctor Review.
- Clinical Feedback Dashboard.
- Lab Tests.
- Specialist Referral.
- Discharge Summary.
- Company Health Dashboard.

### Nurse

Nurses focus on patient care execution, vitals, tasks, medications, queue awareness, and assigned patients.

Typical nurse pages:

- Live Prediction.
- Limited History.
- Nurse Patient Care.
- Nurse Management.
- Patient Status.
- Emergency Queue.
- Medication Records.
- Clinical Alerts.
- Risk Watchlist.

## Technology Stack

### Backend

- FastAPI.
- Uvicorn.
- SQLAlchemy.
- PostgreSQL.
- Pydantic.
- JWT authentication with `python-jose`.
- Environment loading with `python-dotenv`.

### Frontend

- Streamlit.
- Plotly.
- pandas.
- requests.
- streamlit-extras.
- Optional audio recorder and speech recognition packages.

### AI / ML

- XGBoost.
- scikit-learn.
- joblib.
- SHAP.
- pandas and NumPy.

### Image And Reports

- Pillow.
- OpenCV.
- Optional PyTorch and Torchvision.
- ReportLab for PDFs.
- Matplotlib for SHAP/image outputs.

## Setup

### 1. Clone Or Open The Project

```powershell
cd "E:\Personal Project\Sem 2\emergency-triage-ai"
```

### 2. Create A Virtual Environment

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

For the complete app:

```powershell
pip install -r requirements.txt
```

For backend-only deployment:

```powershell
pip install -r backend\requirements.txt
```

For frontend-only deployment:

```powershell
pip install -r frontend\requirements.txt
```

### 4. Configure Environment Variables

Copy the example file:

```powershell
Copy-Item .env.example .env
```

Update `.env` with your local values:

```text
DATABASE_URL=postgresql://<user>:<password>@<host>:5432/<database>
SECRET_KEY=replace-with-a-long-random-secret
ACCESS_TOKEN_EXPIRE_MINUTES=60
HOST=0.0.0.0
PORT=8000
RELOAD=false
CORS_ORIGINS=http://localhost:8501,http://127.0.0.1:8501
API_URL=http://127.0.0.1:8000
```

Optional email settings:

```text
SMTP_EMAIL=
SMTP_PASSWORD=
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=465
SMTP_USE_SSL=true
SMTP_TIMEOUT=60
```

Do not commit `.env`. It contains secrets and local machine settings.

### 5. Prepare PostgreSQL

Create a PostgreSQL database and point `DATABASE_URL` to it.

Example:

```text
postgresql://postgres:password@localhost:5432/emergeai
```

The backend calls `Base.metadata.create_all(bind=engine)` on startup, so missing tables are created automatically for development.

## Running The App

### Start Backend

From the project root:

```powershell
venv\Scripts\Activate.ps1
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

Backend URLs:

```text
API root: http://127.0.0.1:8000
Swagger docs: http://127.0.0.1:8000/docs
Health check: http://127.0.0.1:8000/health
```

### Start Frontend

Open a second terminal from the project root:

```powershell
venv\Scripts\Activate.ps1
$env:API_URL="http://127.0.0.1:8000"
streamlit run frontend/app.py
```

For the standalone local-demo dashboard that does not require FastAPI:

```powershell
streamlit run app.py
```

If Streamlit shows `WinError 10061` or “connection refused”, the backend is not running on port 8000. Start the backend command above, then verify `http://127.0.0.1:8000/health` returns `{"status":"ok", ...}` before logging in.

Streamlit usually opens at:

```text
http://localhost:8501
```

### Run The Enhanced Local ESI Dashboard

The root Streamlit app can run without the FastAPI backend. It uses the new emergency model when available and falls back to clinical rules if the model has not been trained yet.

```powershell
venv\Scripts\Activate.ps1
python -m src.generate_emergency_data
python -m src.train_emergency_model
streamlit run app.py
```

The enhanced local dashboard includes username/password login, local secure registration, Patient Demo Mode, role-based tabs, and local admin approval for newly registered admin requests.

### Generate Synthetic Emergency Data

```powershell
venv\Scripts\Activate.ps1
python -m src.generate_emergency_data --records 10000
```

Outputs:

- `data/emergency_synthetic_data.csv`
- `reports/emergency_synthetic_summary.txt`
- `reports/emergency_synthetic_distributions.png`

### Train The New Emergency ESI Model

```powershell
venv\Scripts\Activate.ps1
python -m src.train_emergency_model
```

Outputs:

- `models/emergency_triage_model.pkl`
- `models/emergency_feature_columns.pkl`
- `reports/emergency_model_metrics.txt`
- `reports/emergency_confusion_matrix.png`
- `reports/emergency_feature_importance.png`

## Demo Login Accounts

The demo users are defined in `src/auth.py`.

| Username | Password | Allowed Roles |
| --- | --- | --- |
| `anuj` | `anuj123` | admin, doctor, nurse |
| `chintan` | `chintan123` | admin, doctor, nurse |
| `admin` | `admin123` | admin |
| `doctor` | `doctor123` | doctor |
| `nurse` | `nurse123` | nurse |

These are development/demo credentials only. A production version should replace them with hashed database-backed users.

## Important API Areas

The backend has many endpoints. See [docs/API_ENDPOINTS.md](docs/API_ENDPOINTS.md) for a dedicated endpoint guide.

Major groups include:

- System: `/`, `/health`.
- Auth: `/login`.
- AI prediction: `/predict`, `/extract-symptoms`, `/analyze-image`, `/shap`, `/shap/image`.
- v2 AI: `/v2/analyze-image-dl`, `/v2/triage`, `/v2/report/{log_id}`, `/v2/dashboard/summary`.
- History: `/history`, `/history/limited`.
- Nurses: `/nurses/add`, `/nurses`, `/api/assignments/assign-nurse`, `/assignments`.
- Vitals and tasks: `/nurse-vitals/add`, `/nurse-tasks/add`, `/nurse-tasks/{task_id}/status`.
- Doctor review: `/doctor-review/add`, `/doctor-review/{prediction_id}`.
- Patient status: `/patient-status/{prediction_id}`.
- Admin: `/admin/staff`, `/admin/workload`, `/admin/workload-summary`, `/admin/audit-logs`.
- Queue: `/emergency-queue`, `/emergency-queue/{prediction_id}/priority`.
- Beds: `/beds/add`, `/beds`, `/beds/assign`, `/beds/release/{bed_id}`.
- Medications: `/medications/add`, `/medications/{prediction_id}`, `/medications/{medication_id}/status`.
- Alerts: `/alerts`, `/alerts/add`, `/alerts/{alert_id}/resolve`.
- Discharge: `/discharge-summary/create`, `/discharge-summary/{prediction_id}`, `/discharge-summary/{prediction_id}/pdf`.
- Patients: `/patients/add`, `/patients`, `/patients/{patient_id}`.
- Labs: `/labs/order`, `/labs/{prediction_id}`, `/labs/{lab_order_id}/result`.
- Referrals: `/referrals/add`, `/referrals`, `/referrals/{prediction_id}`, `/referrals/{referral_id}/status`.
- Consents: `/consents/add`, `/consents/{patient_id}`.
- Incidents: `/incidents/add`, `/incidents`, `/incidents/{incident_id}/status`.
- Billing: `/billing/create`, `/billing/{patient_id}`, `/billing/{billing_id}/status`.
- Inventory: `/inventory/add`, `/inventory`, `/inventory/low-stock`, `/inventory/{item_id}`.
- Appointments: `/appointments/add`, `/appointments`, `/appointments/{patient_id}`, `/appointments/{appointment_id}/status`.
- Notifications: `/notifications/send`, `/notifications/{patient_id}`.
- Analytics: `/analytics/hospital-summary`, `/v2/company-health-dashboard`, `/v2/watchlist`.
- Feedback: `/feedback`, `/clinical-feedback`.
- Retraining: `/v2/retrain-model`.
- Email: `/send-report-email`.

## Database Tables

The SQLAlchemy table models are defined in `src/models.py`.

Main tables:

- `prediction_logs`: Stores patient inputs, ML prediction, final prediction, confidence, explanations, NLP fields, image analysis, feedback status, and timestamps.
- `clinical_feedback`: Stores doctor/admin acceptance or override of AI predictions.
- `nurses`: Stores nurse directory records.
- `nurse_assignments`: Links nurses to prediction records.
- `nurse_vitals`: Stores nurse-recorded vitals.
- `nurse_tasks`: Stores nursing tasks and completion state.
- `doctor_reviews`: Stores diagnosis, treatment plan, follow-up, and admit status.
- `patient_statuses`: Stores patient status timeline.
- `staff_members`: Stores staff account metadata and active status.
- `emergency_queue`: Stores emergency queue priority and wait-time data.
- `beds`: Stores hospital bed inventory and assignment state.
- `medication_records`: Stores medication administration records.
- `clinical_alerts`: Stores clinical warning or alert records.
- `discharge_summaries`: Stores discharge summary details.
- `staff_shifts`: Stores staff scheduling records.
- `patients`: Stores patient demographics and medical history.
- `patient_prediction_links`: Links registered patients to prediction records.
- `lab_orders`: Stores lab orders and results.
- `referrals`: Stores specialist referrals.
- `consents`: Stores patient consent records.
- `incident_reports`: Stores safety or operational incidents.
- `billing_records`: Stores billing and payment status.
- `inventory_items`: Stores supplies, equipment, and low-stock information.
- `follow_up_appointments`: Stores follow-up appointment scheduling.
- `notification_logs`: Stores family/contact notification logs.
- `audit_logs`: Stores user actions for traceability.

## AI And ML Components

### Model Files

The backend looks for models in `backend/models/` first and then in root `models/`.

Important model files:

- `triage_xgboost_balanced.pkl`: Main balanced XGBoost model used by the backend when available.
- `triage_xgboost_balanced_model.pkl`: Balanced model artifact.
- `triage_xgboost_model.pkl`: XGBoost model artifact.
- `triage_baseline_model.pkl`: Baseline model artifact.
- `esi_label_encoder.pkl`: Converts encoded class labels back to ESI labels.
- `retrained_feature_columns.pkl`: Feature column list used after retraining.

### Model Input Columns

The backend prepares model input in `backend/main.py` using:

- Numeric vitals and chief complaint flags.
- One-hot columns for gender, race, ethnicity, and arrival mode.
- A fixed expected column order so prediction matches training.

### Safety Rules

`src/safety_rules.py` checks the model result against clinical danger signals such as severe vitals or emergency symptoms. The final prediction can include safety-rule reasoning, not just raw ML output.

### Explainability

`src/shap_explainer.py` and `src/shap_visualizer.py` create clinical explanations and SHAP visualization output.

### Retraining

`backend/retrain_model.py` and `/v2/retrain-model` support an admin-triggered retraining workflow. Existing model backups are stored in `models/backups/` and `backend/models/backups/`.

## Synthetic Emergency Dataset And ESI Upgrade

The project now includes a modular synthetic emergency department dataset and ESI model pipeline. The generator creates at least 10,000 patient records with clinically linked vitals, symptoms, comorbidities, ICU risk, wait time, length of stay, and readmission risk.

### Dataset Columns

The generated CSV includes:

- `Patient_ID`
- `Age`
- `Gender`
- `Blood_Pressure`
- `Systolic_BP`
- `Diastolic_BP`
- `Heart_Rate`
- `Respiratory_Rate`
- `Oxygen_Saturation`
- `Temperature`
- `Diabetes`
- `Hypertension`
- `Smoking`
- `Chest_Pain`
- `Shortness_of_Breath`
- `Fever`
- `Symptom_Description`
- `Triage_Level`
- `ICU_Required`
- `Wait_Time_Minutes`
- `Hospital_Stay_Days`
- `Readmission_Risk`

Additional engineered model features include `Mean_Arterial_Pressure`, `Pulse_Pressure`, `Shock_Index`, `Comorbidity_Count`, `Abnormal_Vitals_Count`, `NEWS2_Score`, and `Critical_Vital_Flag`.

### ESI Logic

The rule engine in `src/triage_rules.py` maps acuity to:

- ESI 1: Immediate / Critical
- ESI 2: Emergency
- ESI 3: Urgent
- ESI 4: Semi-Urgent
- ESI 5: Non-Urgent

Clinical relationships built into the data and rules:

- Low oxygen saturation increases severity.
- Chest pain increases triage priority.
- High heart rate increases severity.
- Older age, diabetes, and hypertension increase readmission risk.
- ICU patients are concentrated among severe vitals and high NEWS2 scores.
- Critical patients receive shorter wait times.
- Non-urgent patients receive longer wait times.

### New Modular Files

- `src/generate_emergency_data.py`: synthetic emergency data generation.
- `src/triage_rules.py`: ESI, ICU risk, readmission risk, wait-time logic.
- `src/feature_engineering.py`: derived medical features and model columns.
- `src/data_validation.py`: dataset validation checks.
- `src/train_emergency_model.py`: XGBoost or RandomForest training pipeline.

### Enhanced Dashboard

The root `app.py` includes:

- Patient input form for vitals, symptoms, and comorbidities.
- Predicted ESI level.
- ICU risk and readmission risk.
- Color-coded triage badges.
- Emergency alert card for critical patients.
- ESI distribution chart.
- ICU required distribution chart.
- Average wait time by triage level.
- Readmission risk by age group.
- Feature importance chart after model training.

## Patient Demo Mode Registration And Admin Approval

### Patient Demo Mode

Both Streamlit entry points now support a patient-friendly demo flow.

On the login screen, click `Continue as Patient`.

The app stores the demo session as:

- `username = "Patient Demo User"`
- `role = "patient"`
- `is_authenticated = True`

Patient Demo Mode can view synthetic dashboards, triage prediction, analytics, sample synthetic data, and demo report generation. It cannot access admin approvals, user management, settings, delete actions, or database-editing workflows.

Patient Demo Mode disclaimer:

```text
Patient Demo Mode is for educational and demonstration purposes only. Data shown is synthetic and not real patient information.
```

### User Registration

The login page includes a `Register` action. Users register with:

- Full name
- Username
- Email
- Password
- Confirm password
- Requested role: `patient`, `doctor`, `nurse`, or `admin`

Validation checks include required fields, unique username, unique email, matching passwords, and secure password hashing. Passwords are never stored as plain text.

### Registration Approval

All registrations are always created with:

```text
account_status = "pending"
```

The user sees:

```text
Your account is pending approval.
```

Authorized approvers can open `Admin Approvals`, review pending requests in their scope, and approve or reject them. Triage nurses can open the same area to view newly registered patients only.

Approval sets:

- `status = "active"`
- `role = requested_role`
- `approved_by`
- `approved_at`

Rejection sets:

- `status = "rejected"`
- `rejected_by`
- `rejected_at`
- `rejected_reason`
- login blocked

### Backend Auth Endpoints

The FastAPI backend now includes:

- `POST /register`
- `GET /admin/approvals`
- `POST /admin/approvals/{username}/approve`
- `POST /admin/approvals/{username}/reject`

Database-backed registered users live in the new `app_users` table. Existing demo users in `src/auth.py` still work, so the current login flow is preserved.

## Realistic Hospital Hierarchy

The dashboard uses these real-world role labels while keeping stable backend role keys:

- `super_admin`: Super Admin, system owner and central authority.
- `admin`: Hospital Admin, hospital operations administrator.
- `doctor`: Emergency Doctor, ER physician and clinical reviewer.
- `nurse`: Triage Nurse, intake and queue workflow staff.
- `patient`: Patient portal/demo user.

Super admins are pre-created manually and are not approved through public registration.

## Hierarchical Approval System

All newly registered accounts now start with:

```text
account_status = "pending"
```

Users cannot log in until an authorized higher-level role approves the account. Pending users see:

```text
Your account is pending approval.
```

### Approval Hierarchy

- `super_admin`: can approve `admin`.
- `admin`: can approve `doctor`, `nurse`, and `patient`.
- `doctor`: can approve `patient`.
- `nurse`: can view newly registered patients only; cannot approve or reject.
- `patient`: has no approval permissions.

The demo super admin account is:

```text
username: superadmin
password: superadmin123
role: super_admin
```

### Approval Queues

Approval requests route by requested role:

- Patient requests: doctor or hospital admin approval queue.
- Nurse requests: hospital admin approval queue.
- Doctor requests: hospital admin approval queue.
- Admin requests: super admin approval queue.

Nurses receive a **New Patients** view-only table for awareness and workflow preparation. The backend still blocks nurse approval/rejection even if someone calls the API directly.

### Approval Dashboard

Approval panels show:

- Full name
- Username
- Email
- Requested role
- Registration date
- Status
- Approved by
- Approved at
- Rejected reason

Actions:

- Approve
- Reject
- View details

Status colors:

- `pending`: yellow
- `active`: green
- `rejected`: red
- `suspended`: gray

### Backend Security

The backend validates approval permissions on every approval request. Lower hierarchy roles cannot approve higher roles, users cannot approve or reject themselves, and super-admin-only role/status management is available through:

```text
PUT /admin/accounts/{username}
```

## Historical Medical Report Upload And Image Assessment

The system includes a `Historical Reports & Image Assessment` page for uploading previous medical documents, medical images, and screenshots. It generates decision-support summaries for emergency triage context.

Important disclaimer shown in the UI:

```text
This tool analyzes uploaded historical medical reports and medical images for educational and decision-support purposes only. It does not provide a final diagnosis. A licensed healthcare professional must verify all findings.
```

### Supported Upload Types

- PDF
- TXT
- CSV
- DOCX
- JPG
- JPEG
- PNG

Upload fields:

- Patient ID
- Patient name
- Report type
- Upload date
- Notes
- Report file
- Medical image file, when applicable

Report types include blood tests, X-ray, MRI, CT scan, prescriptions, discharge summaries, previous diagnosis, emergency visits, surgery reports, allergy reports, medical image, and other.

### Extraction And Analysis

Text extraction is handled by:

- `src/report_text_extractor.py`
- `PyPDF2` for PDF files
- `python-docx` for DOCX files
- `pandas` for CSV files
- standard text reading for TXT files
- `Pillow`, `opencv-python-headless`, and optional `pytesseract` for image metadata, quality checks, and OCR

Image assessment includes:

- image preview
- file name and size
- image width and height
- image format
- upload date
- blur warning
- low resolution warning
- dark image warning
- very bright image warning
- optional OCR text when `pytesseract` and the Tesseract executable are available

If OCR is unavailable, the app shows:

```text
OCR is not available. Image preview and metadata analysis completed.
```

Medical history analysis is rule-based and detects possible historical indicators such as diabetes, hypertension, heart risk, respiratory risk, allergies, kidney risk, previous surgeries, medications, abnormal labs, emergency visits, smoking history, infection history, pregnancy notes, and neurological symptoms.

### Risk Flags

The system generates color-coded risk cards:

- High Risk: red
- Medium Risk: orange/yellow
- Low Risk: green
- Allergy Alert: blue

Examples:

- Chest pain plus heart disease mentions create a possible high cardiac risk flag.
- Diabetes plus high glucose mentions create possible metabolic risk.
- COPD/asthma plus low oxygen mentions create possible respiratory risk.
- Allergy mentions create an allergy alert.
- ICU history mentions create a possible high emergency risk flag.
- Poor image quality creates an image quality warning.
- X-ray, CT scan, MRI, or medical image uploads create a clinician imaging review flag.
- Notes or OCR mentioning fracture, pneumonia, tumor, bleeding, stroke, infection, or abnormal create a review-needed flag.

### Clinical Summary

The generated doctor/nurse summary includes:

- Patient background
- Known condition mentions
- Current risk factors
- Medication history
- Allergy alerts
- Previous hospital visits
- Possible emergency concerns
- Image quality notes
- OCR findings if available
- Recommended follow-up questions
- Triage notes

These findings are supporting context only and do not override ML prediction or clinician judgment.

### Backend Storage

Uploaded files are stored under:

```text
reports/historical_reports/
```

Database table:

```text
historical_medical_reports
```

Important fields:

- `patient_id`
- `patient_name`
- `uploaded_by`
- `uploaded_by_role`
- `report_type`
- `file_name`
- `file_path`
- `file_type`
- `extracted_text`
- `ocr_text`
- `image_metadata`
- `image_quality_notes`
- `summary`
- `risk_flags`
- `upload_date`
- `doctor_notes`
- `nurse_notes`

### Historical Report API

- `POST /historical-reports/upload`
- `GET /historical-reports`
- `GET /historical-reports/{report_id}`
- `PUT /historical-reports/{report_id}/notes`
- `DELETE /historical-reports/{report_id}`
- `GET /historical-reports/{report_id}/export`

RBAC:

- Admin and super admin: upload, view, delete, export.
- Doctor: upload, view, export, add doctor/nurse notes.
- Nurse: upload, view summaries, add nurse notes.
- Patient: upload own reports and view own summaries only.

## Full Folder Structure

Generated cache folders such as `__pycache__/` and the local `venv/` environment are intentionally omitted because they are not source code.

```text
emergency-triage-ai/
|-- .env                         # Local environment variables; do not commit secrets
|-- .env.example                 # Example environment configuration
|-- .gitignore                   # Git ignore rules
|-- README.md                    # Main project documentation
|-- app.py                       # Root-level Streamlit/app entry or legacy launcher
|-- convert_rdata.py             # Converts R data file into usable project data
|-- features.docx                # Feature planning/documentation file
|-- init_db.py                   # Helper script for database initialization
|-- requirements.txt             # Full project dependency list
|-- New Text Document.txt        # Scratch/project note file
|
|-- backend/
|   |-- main.py                  # Main FastAPI app with core API endpoints
|   |-- retrain_model.py         # Model retraining script
|   |-- requirements.txt         # Backend dependency list
|   |-- run.py                   # Backend startup helper
|   |
|   |-- models/
|   |   |-- triage_xgboost_balanced.pkl
|   |   |-- retrained_feature_columns.pkl
|   |   |
|   |   |-- backups/
|   |       |-- triage_xgboost_backup_20260510_131643.pkl
|   |
|   |-- reports/
|       |-- 37.pdf               # Generated backend report example/output
|
|-- data/
|   |-- 5v_cleandf.rdata         # Original/converted source dataset from R
|   |-- triage.csv               # CSV dataset used for training or inspection
|
|-- docs/
|   |-- API_ENDPOINTS.md         # Detailed API endpoint documentation
|   |-- DEPLOYMENT_GUIDE.md      # Deployment notes for hosting
|   |-- TESTING_CHECKLIST.md     # Manual testing checklist
|   |-- USER_GUIDE.md            # User-facing usage guide
|
|-- frontend/
|   |-- app.py                   # Main Streamlit UI
|   |-- analytics.py             # Frontend analytics helpers
|   |-- requirements.txt         # Frontend dependency list
|
|-- logs/
|   |-- prediction_logs.csv      # CSV prediction log/output file
|
|-- models/
|   |-- esi_label_encoder.pkl
|   |-- retrained_feature_columns.pkl
|   |-- triage_baseline_model.pkl
|   |-- triage_xgboost_balanced.pkl
|   |-- triage_xgboost_balanced_model.pkl
|   |-- triage_xgboost_model.pkl
|   |
|   |-- backups/
|       |-- triage_xgboost_backup_20260510_133627.pkl
|       |-- triage_xgboost_backup_20260510_133914.pkl
|       |-- triage_xgboost_backup_20260510_231254.pkl
|
|-- reports/
|   |-- 1.pdf
|   |-- 17.pdf
|   |-- 27.pdf
|   |-- 28.pdf
|   |-- 29.pdf
|   |-- 31.pdf
|
|-- shap_outputs/
|   |-- latest_shap_bar.png      # Latest generated SHAP bar chart
|
|-- src/
    |-- __init__.py
    |-- api_extensions.py        # v2 multi-modal routes and dashboard summary API
    |-- audit_logger.py          # Prediction/audit logging helpers
    |-- auth.py                  # Demo users, JWT creation, current-user validation
    |-- check_data.py            # Data checking helper script
    |-- dashboard.py             # Dashboard helper module
    |-- database.py              # SQLAlchemy engine, Base, session, get_db
    |-- dl_module.py             # Deep-learning image analyzer helper
    |-- email_service.py         # SMTP email report delivery
    |-- feature_schema.py        # Feature schema definitions for ML inputs
    |-- image_analyzer.py        # Classical computer vision image analysis
    |-- models.py                # SQLAlchemy database table models
    |-- predict_baseline.py      # Baseline model prediction script
    |-- report_generator.py      # PDF clinical report generator
    |-- safety_rules.py          # Clinical safety rule layer
    |-- shap_explainer.py        # Clinical explanation generation
    |-- shap_visualizer.py       # SHAP plot initialization/rendering
    |-- symptom_extractor.py     # NLP symptom extraction
    |-- train_baseline.py        # Baseline model training
    |-- train_xgboost.py         # XGBoost training script
    |-- train_xgboost_balanced.py # Balanced XGBoost training script
    |-- triage_engine.py         # Multi-modal triage scoring engine
```

## Important Files Explained

### `backend/main.py`

This is the main FastAPI backend. It:

- Creates the FastAPI app.
- Loads environment variables.
- Loads ML models and label encoder.
- Configures CORS.
- Creates database tables.
- Registers v2 routes.
- Defines request schemas.
- Protects endpoints with role checks.
- Implements prediction, SHAP, history, nurse, doctor, patient, admin, queue, bed, medication, alert, discharge, lab, referral, consent, incident, billing, inventory, appointment, notification, analytics, feedback, retraining, and email endpoints.

### `frontend/app.py`

This is the main Streamlit interface. It:

- Renders login.
- Stores JWT and user role in session state.
- Shows role-based navigation.
- Calls FastAPI endpoints with `requests`.
- Displays prediction forms, dashboards, charts, tables, reports, and workflow panels.

### `src/models.py`

Defines all SQLAlchemy ORM models and relationships used by the database.

### `src/database.py`

Creates the SQLAlchemy engine and session utilities. Most backend database work depends on this file.

### `src/auth.py`

Handles demo users, password checks, JWT creation, and current-user validation.

### `src/api_extensions.py`

Adds the v2 APIs:

- Deep-learning image analysis.
- Multi-modal triage scoring.
- PDF report download.
- Dashboard summary.

### `src/triage_engine.py`

Computes multi-modal triage results from vitals, symptoms, NLP findings, and image findings.

### `src/image_analyzer.py`

Performs classical image analysis for uploaded medical images.

### `src/dl_module.py`

Provides the optional deep-learning image analyzer.

### `src/report_generator.py`

Generates clinical PDF reports.

### `src/email_service.py`

Sends report emails through SMTP settings from `.env`.

### `src/symptom_extractor.py`

Extracts symptoms and emergency keywords from patient text.

### `src/safety_rules.py`

Applies clinical safety logic on top of raw model predictions.

### `src/shap_explainer.py` And `src/shap_visualizer.py`

Generate AI explanation text and SHAP visual outputs.

### Training Scripts

The following scripts are used for model development:

- `src/train_baseline.py`
- `src/train_xgboost.py`
- `src/train_xgboost_balanced.py`
- `src/predict_baseline.py`
- `backend/retrain_model.py`

## Testing Workflow

Use [docs/TESTING_CHECKLIST.md](docs/TESTING_CHECKLIST.md) for the full manual checklist.

Recommended smoke test:

1. Start PostgreSQL.
2. Start the FastAPI backend.
3. Open `http://127.0.0.1:8000/health`.
4. Start the Streamlit frontend.
5. Login as `admin`.
6. Add or view staff.
7. Add a patient.
8. Run a live triage prediction.
9. View prediction history.
10. Generate SHAP explanation.
11. Assign a nurse.
12. Add nurse vitals and tasks.
13. Add doctor review.
14. Update patient status.
15. Add medication record.
16. Add or resolve a clinical alert.
17. Assign and release a bed.
18. Create a discharge summary.
19. Download a PDF report.
20. Check hospital analytics and company dashboard.

## Deployment Notes

See [docs/DEPLOYMENT_GUIDE.md](docs/DEPLOYMENT_GUIDE.md) for hosting guidance.

Production backend command:

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

Production frontend command:

```bash
streamlit run frontend/app.py
```

Deployment reminders:

- Set `DATABASE_URL` to a managed PostgreSQL database.
- Set a strong `SECRET_KEY`.
- Configure `CORS_ORIGINS` for the deployed frontend URL.
- Upload required model files.
- Configure SMTP only if email reports are needed.
- Do not deploy the demo `.env` file.
- Replace demo users before any real-world use.

## Known Limitations

- Demo users are hardcoded in `src/auth.py`.
- Passwords are plain text for development demonstration.
- There is no Alembic migration setup yet.
- Generated reports and model artifacts are stored locally.
- Some AI modules are educational approximations and require clinical validation before any real healthcare use.
- The application is not HIPAA/PHIPA/GDPR compliant as-is.
- The model should not be trusted for real emergency triage.

## Future Improvements

- Replace demo users with database-backed hashed accounts.
- Add Alembic migrations.
- Add automated API tests with `pytest` and `httpx`.
- Add CI checks for linting, imports, and endpoint smoke tests.
- Add cloud object storage for generated PDFs and uploaded images.
- Add stronger model monitoring and drift detection.
- Add model cards and dataset documentation.
- Add stricter clinical validation workflow.
## Registration Approvals and Triage Uploads

New dashboard registrations are saved immediately with `account_status="pending"` and cannot log in until approved. Approval visibility follows the role hierarchy:

- `super_admin`: hospital admin approval requests, plus system-level user oversight
- `admin`: doctor, nurse, patient requests
- `doctor`: patient requests
- `nurse`: view newly registered patient records only
- `patient`: no approval access

Approvers must be active users, cannot approve themselves, and lower roles cannot approve higher roles. Nurses can see a **New Patients** table with patient ID, full name, username, email, registration date, account status, and a details view, but nurses do not receive Approve or Reject actions. Patient approvals are handled only by doctors or hospital admins. The frontend shows a pending approval badge for approvers and a new patient badge for nurses.

The Live Prediction page now includes an optional **Upload Past Medical Report or Medical Image Optional** section after patient vitals/symptoms. Supported files are `pdf`, `txt`, `csv`, `docx`, `jpg`, `jpeg`, and `png`. Uploaded files are stored under `reports/triage_uploads/reports/` or `reports/triage_uploads/images/`, analyzed as supporting context, and shown alongside ESI prediction, ICU/readmission risk, risk flags, OCR/image metadata, and doctor/nurse summary. Upload analysis is educational decision support only and does not provide a final diagnosis or override the ML prediction.

Test approval hierarchy:

1. Start backend: `uvicorn backend.main:app --reload --port 8000`
2. Start frontend: `streamlit run frontend/app.py`
3. Register a patient, nurse, doctor, or admin from the login page.
4. Log in as an allowed higher role and open **Admin Approvals**.
5. Approve or reject the pending request, then try logging in as that user.
6. Log in as `nurse` and open **Admin Approvals** to confirm the **New Patients** table is view-only and has no Approve or Reject buttons.

Test report or image upload:

1. Log in as patient, nurse, doctor, admin, or super admin.
2. Open **Live Prediction**.
3. Enter patient information, vitals, and symptoms.
4. Upload a supported report/image, choose report type, add notes, and select **Analyze Uploaded Context**.
5. Run prediction and review **Historical Context**, **Risk Flags**, **OCR Preview**, **Image Metadata**, and **Doctor/Nurse Summary**.

## End-To-End Hospital Workflow

The complete realistic emergency department flow:

1. Patient registers via the login page and selects role `patient`.
2. Account is created with `account_status = "pending"`.
3. Doctor or hospital admin approves the patient via **Admin Approvals**.
4. Patient account becomes `active`.
5. A clinician opens **Live Prediction** and enters patient vitals and symptoms.
6. Optional past medical report or image is uploaded for context.
7. AI predicts ESI level, ICU risk, and readmission risk.
8. Patient is inserted into the emergency waiting queue.
9. Auto nurse assignment runs immediately after prediction is saved.
10. If active nurses exist, a nurse is assigned within seconds.
11. Nurse opens **My Patients** and sees their assigned patients.
12. Nurse starts triage, records vitals, adds tasks, and adds notes.
13. Nurse sends patient to doctor review from the status flow panel.
14. Doctor opens **Doctor Review Queue** and reviews AI prediction, nurse notes, vitals, and risk flags.
15. Doctor adds diagnosis and treatment plan.
16. Doctor marks case as completed or adjusts admit status.
17. PDF clinical report is generated via **PDF Reports**.
18. Dashboards and analytics update automatically.

## Seed Demo Data

Run this once to create active demo nurses so auto-assignment works immediately:

```powershell
venv\Scripts\Activate.ps1
python scripts/seed_demo_data.py
```

This creates:

- Three active nurses: `nurse_sarah`, `nurse_emily`, `nurse_michael` (password: `nurse123`)
- Demo doctor: `demo_doctor` (password: `doctor123`)
- Demo admin: `demo_admin` (password: `admin123`)
- Three sample patients with waiting queue entries

If auto-assignment is still showing `Unassigned` after prediction, the most common cause is that no active Nurse records exist in the database. Running the seed script fixes this.

## Nurse Assignment Troubleshooting

**Symptom**: Patients show `assignment_status = waiting` and `assigned_nurse = Unassigned` even after running Live Prediction.

**Root cause**: The auto-assignment engine calls `sync_active_nurse_users` which creates Nurse records from `app_users` rows where `role = nurse` and `account_status = active`. If the database has no such rows (the demo users in `src/auth.py` are in-memory only, not in the database), no nurses are found and auto-assignment returns None.

**Fix**:

```powershell
python scripts/seed_demo_data.py
```

This creates AppUser records for three nurses and ensures they have corresponding Nurse table records. After seeding, any new prediction will be auto-assigned immediately.

**Manual fix without seed script**: Go to Nurse Management and add at least one nurse with an email address. Then run Auto Assign All Waiting from the Waiting Queue page.

## Patient-Nurse Assignment System

EmergAI now models a realistic emergency-room assignment flow:

1. Patient registers and is approved.
2. Patient enters the emergency triage workflow.
3. Running **Live Prediction** creates a prediction log and emergency queue entry.
4. The backend auto-assigns the least-busy available triage nurse when nurse records exist.
5. ESI 1 or ESI 2 patients prefer experienced, senior, or critical-care nurses.
6. Assigned nurses see only their assigned patients in the queue/dashboard.
7. Doctors or hospital admins can manually assign/reassign nurses.
8. Nurses perform triage and send patients to doctor review.

Assignment statuses:

- `waiting`
- `assigned`
- `in_triage`
- `doctor_review`
- `completed`
- `critical`

Queue priority is ordered by ESI severity:

- ESI 1: red, critical, top of queue
- ESI 2: orange, high priority
- ESI 3: yellow, urgent
- ESI 4: blue, semi-urgent
- ESI 5: green, non-urgent

Backend assignment endpoints:

- `GET /api/assignments/waiting-queue`
- `GET /api/assignments/nurse-workload`
- `POST /api/assignments/assign-nurse`
- `POST /api/assignments/reassign-nurse`
- `POST /api/assignments/update-status`

Backend assignment endpoints:

- `POST /api/assignments/auto-assign`
- `POST /api/assignments/auto-assign-all`
- `GET /api/assignments/my-patients`
- `GET /api/assignments/patient/{patient_id}`

Test assignment workflow:

1. Start backend: `python -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000`
2. Seed demo data: `python scripts/seed_demo_data.py`
3. Start frontend: `streamlit run frontend/app.py`
4. Log in as `admin` or `doctor`.
5. Open **Waiting Queue** page and click **Auto Assign All Waiting Patients**.
6. Confirm patients now show assigned nurse names.
7. Open **Nurse Workload** to see workload cards for each nurse.
8. Log in as `nurse_sarah` (password: `nurse123`, role: `nurse`).
9. Open **My Patients** to see only Sarah's assigned patients.
10. Click **Start Triage** to advance a patient to In Triage status.
11. Record vitals and add a nurse task.
12. Click **Send to Doctor** to advance to Doctor Review.
13. Log in as `doctor` or `demo_doctor`.
14. Open **Doctor Review Queue** to see patients awaiting review.
15. Add diagnosis and treatment plan.
16. Mark case as completed.
17. Open **PDF Reports** to generate a clinical report.

## New Pages Added

| Page | Route Key | Access |
|------|-----------|--------|
| Waiting Queue | `Waiting Queue` | admin, doctor, nurse (own), super_admin |
| Nurse Workload | `Nurse Workload` | admin, super_admin, doctor, nurse |
| My Patients | `My Patients` | nurse (own), admin, doctor |
| Doctor Review Queue | `Doctor Review Queue` | doctor, admin, super_admin |

All pages are available in the sidebar navigation under their respective workspace groups.

## Backend Commands

```powershell
# Start backend (from project root, venv activated)
python -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000

# Start frontend (second terminal)
$env:API_URL="http://127.0.0.1:8000"
streamlit run frontend/app.py

# Seed demo data
python scripts/seed_demo_data.py

# Train model (if not already done)
python -m src.generate_emergency_data
python -m src.train_emergency_model
```

## Frontend Connection Error

If the frontend shows a connection error:

1. Confirm the backend is running: `python -m uvicorn backend.main:app --reload --port 8000`
2. Visit `http://127.0.0.1:8000/health` in your browser. It should return `{"status": "ok"}`.
3. Make sure `API_URL` is set to `http://127.0.0.1:8000` (not `localhost` without port).
4. On Windows, use `python -m uvicorn` instead of `uvicorn` directly if the command is not found.
