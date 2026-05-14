# Deployment Guide

This guide covers deployment preparation for the FastAPI backend and Streamlit frontend.

## Required Files

Keep these files available in the deployed backend package:

- `backend/main.py`
- `src/`
- `backend/models/` or project-root `models/`
- `requirements.txt` or `backend/requirements.txt`
- `.env` values configured in the hosting platform

The backend resolves model files with absolute `pathlib.Path` paths:

1. `backend/models/<filename>`
2. `models/<filename>`

Required model artifact:

- `esi_label_encoder.pkl`

Expected model artifacts:

- `triage_xgboost_balanced.pkl`
- `triage_xgboost.pkl` or matching fallback model present in the project

## Environment Variables

Backend:

- `DATABASE_URL`
- `SECRET_KEY`
- `ACCESS_TOKEN_EXPIRE_MINUTES`
- `CORS_ORIGINS`
- `HOST`
- `PORT`
- `RELOAD`
- SMTP variables if email reports are used

Frontend:

- `API_URL`

Example `CORS_ORIGINS`:

```text
https://your-streamlit-app.onrender.com,https://your-frontend.azurewebsites.net
```

## Backend Command

Run from the `backend/` directory:

```bash
uvicorn main:app --host 0.0.0.0 --port 9000
```

Some hosts provide a dynamic port. If they expose `$PORT`, use:

```bash
uvicorn main:app --host 0.0.0.0 --port $PORT
```

## Frontend Command

Run from the project root:

```bash
streamlit run frontend/app.py
```

Set `API_URL` to the public backend URL.

## Render Backend

1. Create a Web Service.
2. Connect the repository.
3. Set root directory to the project root, or configure build/start commands with paths.
4. Build command:

```bash
pip install -r backend/requirements.txt
```

5. Start command:

```bash
cd backend && uvicorn main:app --host 0.0.0.0 --port $PORT
```

6. Add environment variables in Render dashboard.
7. Use Render PostgreSQL or an external PostgreSQL database and set `DATABASE_URL`.

## Render Streamlit

1. Create another Web Service for Streamlit.
2. Build command:

```bash
pip install -r frontend/requirements.txt
```

3. Start command:

```bash
streamlit run frontend/app.py --server.port $PORT --server.address 0.0.0.0
```

4. Set `API_URL` to the deployed backend URL.

## Azure App Service Backend

1. Create an App Service with Python runtime.
2. Add app settings for all backend environment variables.
3. Configure startup command:

```bash
cd backend && uvicorn main:app --host 0.0.0.0 --port 8000
```

Azure commonly routes traffic to port `8000` for Python containers. If using a custom container or different setting, match the platform port.

## Azure Streamlit Frontend

Use a separate App Service or container:

```bash
streamlit run frontend/app.py --server.port 8000 --server.address 0.0.0.0
```

Set `API_URL` to the backend App Service URL and add the frontend URL to backend `CORS_ORIGINS`.

## Database Migration Note

The current app creates missing tables with SQLAlchemy `Base.metadata.create_all()`. This does not modify existing columns or constraints. For production, add Alembic migrations. For a capstone demo database, a reset can be simpler, but only after backing up any data you need.

## Production Checklist

- Replace demo auth users with database-backed hashed passwords.
- Use a long random `SECRET_KEY`.
- Do not commit `.env`.
- Restrict `CORS_ORIGINS` to deployed frontend domains.
- Use managed PostgreSQL.
- Confirm model files are included in deployment.
- Confirm `/health` and `/docs` load after deployment.
- Disable `RELOAD` in production.
