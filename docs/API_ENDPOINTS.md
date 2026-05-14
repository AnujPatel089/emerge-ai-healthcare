# API Endpoints

Base URL locally: `http://127.0.0.1:9000`

Most endpoints require `Authorization: Bearer <token>` from `POST /login`.

## Core

- `GET /` - backend metadata
- `GET /health` - health check
- `GET /docs` - OpenAPI documentation
- `POST /login` - login with `username` and `password`

## AI Triage

- `POST /predict` - run triage prediction
- `POST /extract-symptoms` - extract symptoms from text
- `POST /analyze-image` - basic image analysis
- `POST /shap` - SHAP explanation
- `POST /shap/image` - SHAP image output
- `GET /history` - prediction history
- `GET /history/limited` - nurse-safe limited history
- `POST /feedback` - prediction feedback
- `POST /clinical-feedback` - doctor/admin feedback
- `GET /clinical-feedback` - clinical feedback records

## V2 Routes

- `POST /v2/analyze-image-dl` - deep learning image analysis
- `POST /v2/triage` - multi-modal triage
- `GET /v2/report/{prediction_id}` - report file
- `GET /v2/dashboard/summary` - dashboard summary
- `GET /v2/watchlist` - risk watchlist
- `POST /v2/retrain-model` - model retraining
- `GET /v2/company-health-dashboard` - company dashboard

## Patients

- `POST /patients/add`
- `GET /patients`
- `GET /patients/{patient_id}`
- `PUT /patients/{patient_id}`
- `PUT /patient-status/{prediction_id}`
- `GET /patient-status/{prediction_id}`

## Nurses and Care

- `POST /nurses/add`
- `GET /nurses`
- `POST /assign-nurse`
- `GET /assignments`
- `GET /assignments/{prediction_id}`
- `POST /nurse-vitals/add`
- `GET /nurse-vitals/{prediction_id}`
- `POST /nurse-tasks/add`
- `GET /nurse-tasks/{prediction_id}`
- `PUT /nurse-tasks/{task_id}/status`

## Doctor Review

- `POST /doctor-review/add`
- `GET /doctor-review/{prediction_id}`

## Operations

- `GET /emergency-queue`
- `PUT /emergency-queue/{prediction_id}/priority`
- `POST /beds/add`
- `GET /beds`
- `POST /beds/assign`
- `PUT /beds/release/{bed_id}`
- `POST /medications/add`
- `GET /medications/{prediction_id}`
- `PUT /medications/{medication_id}/status`
- `GET /alerts`
- `POST /alerts/add`
- `PUT /alerts/{alert_id}/resolve`

## Clinical Admin

- `POST /labs/order`
- `GET /labs/{prediction_id}`
- `PUT /labs/{lab_order_id}/result`
- `POST /referrals/add`
- `GET /referrals`
- `GET /referrals/{prediction_id}`
- `PUT /referrals/{referral_id}/status`
- `POST /consents/add`
- `GET /consents/{patient_id}`
- `POST /incidents/add`
- `GET /incidents`
- `PUT /incidents/{incident_id}/status`

## Billing, Inventory, Scheduling

- `POST /billing/create`
- `GET /billing/{patient_id}`
- `PUT /billing/{billing_id}/status`
- `POST /inventory/add`
- `GET /inventory`
- `GET /inventory/low-stock`
- `PUT /inventory/{item_id}`
- `POST /appointments/add`
- `GET /appointments`
- `GET /appointments/{patient_id}`
- `PUT /appointments/{appointment_id}/status`
- `POST /notifications/send`
- `GET /notifications/{patient_id}`

## Discharge and Shifts

- `POST /discharge-summary/create`
- `GET /discharge-summary/{prediction_id}`
- `GET /discharge-summary/{prediction_id}/pdf`
- `POST /shifts/add`
- `GET /shifts`
- `GET /shifts/current`

## Admin

- `GET /admin/staff`
- `PUT /admin/staff/{username}`
- `GET /admin/workload`
- `GET /admin/workload-summary`
- `GET /admin/audit-logs`
- `GET /analytics/hospital-summary`
