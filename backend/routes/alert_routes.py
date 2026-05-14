from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from src.database import get_db
from src.platform.alert_manager import create_alert, list_alerts, resolve_alert


class AlertInput(BaseModel):
    alert_type: str
    severity: str
    title: str
    message: str


def build_router(require_role):
    router = APIRouter(prefix="/api/platform/alerts", tags=["Platform Alerts"])

    @router.get("")
    def get_alerts(include_resolved: bool = False, current_user=Depends(require_role(["doctor", "admin", "super_admin"])), db=Depends(get_db)):
        return {"alerts": list_alerts(db, include_resolved=include_resolved)}

    @router.post("/create")
    def create(payload: AlertInput, current_user=Depends(require_role(["admin", "super_admin"])), db=Depends(get_db)):
        return create_alert(db, payload.alert_type, payload.severity, payload.title, payload.message)

    @router.post("/{alert_id}/resolve")
    def resolve(alert_id: int, current_user=Depends(require_role(["admin", "super_admin"])), db=Depends(get_db)):
        result = resolve_alert(db, alert_id, current_user["username"])
        if not result:
            raise HTTPException(status_code=404, detail="Alert not found.")
        return result

    return router
