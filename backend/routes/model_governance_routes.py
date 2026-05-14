from fastapi import APIRouter, Depends, HTTPException

from src.database import get_db
from src.mlops.model_card_generator import get_model_card
from src.mlops.model_registry import get_current_production_model, list_models


def build_router(require_role):
    router = APIRouter(prefix="/api/mlops/governance", tags=["Model Governance"])

    @router.get("")
    def governance(current_user=Depends(require_role(["doctor", "admin", "super_admin"])), db=Depends(get_db)):
        production = get_current_production_model(db)
        return {
            "production_model_version": production.get("model_version"),
            "model_status": production.get("status"),
            "training_date": production.get("training_date"),
            "metrics": {
                "accuracy": production.get("accuracy"),
                "precision": production.get("precision"),
                "recall": production.get("recall"),
                "f1_score": production.get("f1_score"),
                "recall_esi_1_2": production.get("recall_esi_1_2"),
            },
            "promotion_history": list_models(db),
            "retraining_history": [row for row in list_models(db) if row.get("status") in {"candidate", "staging"}],
            "limitations": "Educational/demo AI-supported triage only. Clinician review required.",
            "safety_disclaimer": "Possible risk only. This is not a final medical diagnosis.",
        }

    @router.get("/model-card/{version}")
    def model_card(version: str, current_user=Depends(require_role(["doctor", "admin", "super_admin"])), db=Depends(get_db)):
        card = get_model_card(db, version)
        if not card:
            raise HTTPException(status_code=404, detail="Model card not found.")
        return card

    return router
