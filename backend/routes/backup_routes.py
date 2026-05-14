from fastapi import APIRouter, Depends

from src.database import get_db
from src.platform.backup_manager import backup_metadata, export_backup


def build_router(require_role):
    router = APIRouter(prefix="/api/platform/backup", tags=["Backup Export"])

    @router.get("/metadata")
    def metadata(current_user=Depends(require_role(["admin", "super_admin"])), db=Depends(get_db)):
        return backup_metadata(db)

    @router.post("/export")
    def export(current_user=Depends(require_role(["admin", "super_admin"])), db=Depends(get_db)):
        return export_backup(db)

    return router
