def classify_error(message: str) -> dict:
    text = (message or "").lower()
    if "database" in text or "sql" in text or "connection" in text:
        return {"incident_type": "database_failure", "severity": "critical", "service": "database"}
    if "model" in text or "xgboost" in text or "label encoder" in text:
        return {"incident_type": "model_failure", "severity": "critical", "service": "model"}
    if "predict" in text:
        return {"incident_type": "prediction_failure", "severity": "warning", "service": "prediction"}
    if "upload" in text or "ocr" in text or "image" in text:
        return {"incident_type": "upload_failure", "severity": "warning", "service": "upload"}
    if "queue" in text or "assignment" in text or "nurse" in text:
        return {"incident_type": "queue_assignment_failure", "severity": "warning", "service": "queue"}
    if "latency" in text or "timeout" in text:
        return {"incident_type": "high_latency", "severity": "warning", "service": "api"}
    return {"incident_type": "backend_failure", "severity": "warning", "service": "backend"}
