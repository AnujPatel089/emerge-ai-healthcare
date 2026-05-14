from fastapi import APIRouter, Depends
from sqlalchemy import or_

from src.database import get_db
from src.mlops.model_monitor import model_health
from src.models import EmergencyQueue, MLPredictionMonitoring, Nurse, PlatformIncident, PredictionLog
from src.platform.system_health import health_summary


def build_router(require_role, model_loaded_getter, label_encoder_loaded_getter):
    router = APIRouter(prefix="/api/command-center", tags=["Command Center"])

    @router.get("/summary")
    def summary(current_user=Depends(require_role(["doctor", "admin", "super_admin"])), db=Depends(get_db)):
        return {
            "active_emergency_queue_count": db.query(EmergencyQueue).count(),
            "critical_patients": db.query(PredictionLog).filter(or_(PredictionLog.final_prediction.ilike("%1%"), PredictionLog.final_prediction.ilike("%2%"))).count(),
            "active_nurses": db.query(Nurse).filter(Nurse.available_status.is_(True)).count(),
            "nurse_workload": [{"id": n.id, "name": n.name, "active_patient_count": n.active_patient_count} for n in db.query(Nurse).limit(50).all()],
            "failed_predictions": db.query(MLPredictionMonitoring).filter(MLPredictionMonitoring.failed.is_(True)).count(),
            "active_incidents": db.query(PlatformIncident).filter(PlatformIncident.status == "open").count(),
            "model_health": model_health(db),
            "backend_health": health_summary(db, model_loaded=model_loaded_getter(), label_encoder_loaded=label_encoder_loaded_getter()),
        }

    return router
