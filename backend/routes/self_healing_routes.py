from fastapi import APIRouter, Depends, HTTPException

from src.database import get_db
from src.platform.self_healing import attempt_recovery, resolve_self_healing_incident, run_checks_and_log, self_healing_status


def build_router(require_role, model_loaded_getter):
    router = APIRouter(prefix="/api/platform/self-healing", tags=["Self Healing"])

    @router.get("/status")
    def status(current_user=Depends(require_role(["doctor", "admin", "super_admin"])), db=Depends(get_db)):
        return self_healing_status(db, model_loaded=model_loaded_getter())

    @router.post("/run-checks")
    def run_checks(current_user=Depends(require_role(["admin", "super_admin"])), db=Depends(get_db)):
        return run_checks_and_log(db, model_loaded=model_loaded_getter())

    @router.post("/recover")
    def recover(current_user=Depends(require_role(["admin", "super_admin"])), db=Depends(get_db)):
        return attempt_recovery(db)

    @router.get("/incidents")
    def incidents(current_user=Depends(require_role(["admin", "super_admin"])), db=Depends(get_db)):
        return self_healing_status(db, model_loaded=model_loaded_getter()).get("incidents", [])

    @router.post("/resolve-incident/{incident_id}")
    def resolve(incident_id: int, current_user=Depends(require_role(["admin", "super_admin"])), db=Depends(get_db)):
        result = resolve_self_healing_incident(db, incident_id, current_user["username"])
        if not result:
            raise HTTPException(status_code=404, detail="Incident not found.")
        return result

    return router
