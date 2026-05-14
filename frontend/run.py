import os
import subprocess

port = os.getenv("PORT", "8501")

subprocess.run([
    "streamlit",
    "run",
    "frontend/app.py",
    "--server.port", port,
    "--server.address", "0.0.0.0"
])