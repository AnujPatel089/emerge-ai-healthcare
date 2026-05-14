# Testing Checklist

Use this checklist after installing dependencies and setting `.env`.

## Startup

- [ ] Backend starts with `uvicorn main:app --host 0.0.0.0 --port 9000`
- [ ] `GET /health` returns success
- [ ] `GET /docs` loads
- [ ] Streamlit starts with `streamlit run frontend/app.py`
- [ ] Frontend `API_URL` points to the backend

## Authentication

- [ ] Admin can log in
- [ ] Doctor can log in
- [ ] Nurse can log in
- [ ] Invalid credentials are rejected
- [ ] Nurse cannot access admin-only pages

## Core Clinical Workflow

- [ ] Add patient
- [ ] Run triage prediction
- [ ] Confirm prediction response contains `prediction_id`
- [ ] View prediction history
- [ ] Open SHAP explanation
- [ ] Upload image for analysis if needed

## Nurse Workflow

- [ ] Add nurse
- [ ] Assign nurse to `prediction_id`
- [ ] View assignments
- [ ] Add vitals
- [ ] Add nurse task
- [ ] Update task status

## Doctor Workflow

- [ ] Add doctor review
- [ ] View doctor review by `prediction_id`
- [ ] Update patient status
- [ ] Order lab test
- [ ] Add referral
- [ ] Generate discharge summary
- [ ] Download discharge PDF

## Operations Workflow

- [ ] View emergency queue
- [ ] Update queue priority
- [ ] Add bed
- [ ] Assign bed
- [ ] Release bed
- [ ] Add medication record
- [ ] Update medication status
- [ ] Add alert
- [ ] Resolve alert

## Admin Workflow

- [ ] View staff by role
- [ ] Activate/deactivate staff
- [ ] Assign staff department
- [ ] View workload dashboard
- [ ] Add/update inventory
- [ ] View low-stock inventory
- [ ] View hospital analytics

## Deployment Smoke Test

- [ ] Public backend `/health` works
- [ ] Public backend `/docs` works
- [ ] Public frontend loads
- [ ] Frontend login succeeds against deployed backend
- [ ] CORS permits frontend domain only
- [ ] Model files load on deployed backend
- [ ] Database tables exist

## Known Migration Note

If a request fails because a column or table is missing, the database schema is older than the code. Use migrations or reset the development database after backing up data.
