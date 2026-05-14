"""Emergency queue helpers for ESI-prioritized ER workflow."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any

from src.models import EmergencyQueue, PredictionLog


ESI_PRIORITY = {
    1: ("Critical", "Red", 0, True),
    2: ("High", "Orange", 10, True),
    3: ("Medium", "Yellow", 45, False),
    4: ("Low", "Blue", 90, False),
    5: ("Low", "Green", 150, False),
}


def extract_esi_level(value: Any, oxygen: int | None = None) -> int:
    text = str(value or "")
    match = re.search(r"\b([1-5])\b", text)
    if match:
        return int(match.group(1))
    lowered = text.lower()
    if "critical" in lowered:
        return 1
    if "emergency" in lowered or "high" in lowered:
        return 2
    if "urgent" in lowered:
        return 3
    if oxygen is not None and oxygen < 90:
        return 1
    if oxygen is not None and oxygen < 92:
        return 2
    return 3


def queue_priority_from_prediction(prediction: PredictionLog) -> dict:
    esi_level = extract_esi_level(prediction.final_prediction, prediction.triage_vital_o2)
    priority, color, wait_time, critical = ESI_PRIORITY.get(esi_level, ESI_PRIORITY[3])
    return {
        "esi_level": esi_level,
        "priority": priority,
        "color": color,
        "estimated_wait_time": wait_time,
        "critical_status": critical,
    }


def ensure_emergency_queue_entry(db, prediction: PredictionLog) -> EmergencyQueue:
    priority = queue_priority_from_prediction(prediction)
    entry = db.query(EmergencyQueue).filter(EmergencyQueue.prediction_id == prediction.id).first()
    if not entry:
        entry = EmergencyQueue(
            prediction_id=prediction.id,
            esi_severity=f"ESI {priority['esi_level']}",
            priority=priority["priority"],
            estimated_wait_time=priority["estimated_wait_time"],
            critical_status=priority["critical_status"],
            assignment_status="waiting",
            arrival_time=prediction.created_at or datetime.utcnow(),
        )
        db.add(entry)
    else:
        entry.esi_severity = f"ESI {priority['esi_level']}"
        entry.priority = priority["priority"]
        entry.estimated_wait_time = priority["estimated_wait_time"]
        entry.critical_status = priority["critical_status"]
        if entry.assignment_status is None:
            entry.assignment_status = "waiting"
        entry.updated_at = datetime.utcnow()
    return entry


def ordered_queue_query(db):
    return (
        db.query(EmergencyQueue)
        .order_by(
            EmergencyQueue.critical_status.desc(),
            EmergencyQueue.esi_severity.asc(),
            EmergencyQueue.arrival_time.asc(),
        )
    )
