"""
Multi-Modal Triage Scoring Engine

Fuses:
  - Vitals (HR, BP, RR, SpO2, Temp) → modified NEWS2 score
  - Structured symptoms (severity-weighted)
  - NLP-extracted findings from free text (from symptom_extractor.py)
  - Image analysis (severity_score from dl_module + cv_module flags)

Output: ESI-style triage level (1=resuscitation … 5=non-urgent) with
component breakdown so SHAP / clinicians can see where the score came from.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
import math


# ---------- Inputs ----------
@dataclass
class Vitals:
    heart_rate: Optional[float] = None         # bpm
    systolic_bp: Optional[float] = None        # mmHg
    respiratory_rate: Optional[float] = None   # /min
    spo2: Optional[float] = None               # %
    temperature: Optional[float] = None        # °C
    consciousness: Optional[str] = None        # 'alert' | 'voice' | 'pain' | 'unresponsive'


@dataclass
class TriageInput:
    vitals: Optional[Vitals] = None
    structured_symptoms: List[Dict[str, Any]] = field(default_factory=list)
    # e.g. [{"name": "chest_pain", "severity": "severe"}, ...]
    nlp_findings: Optional[Dict[str, Any]] = None
    # output of symptom_extractor.py
    image_analysis: Optional[Dict[str, Any]] = None
    # output of dl_module.analyze() merged with cv_module
    age: Optional[int] = None
    sex: Optional[str] = None


# ---------- Scoring components ----------
def _news2_vitals(v: Vitals) -> Dict[str, Any]:
    """Modified NEWS2: each vital → 0–3 risk points; sum is the vitals score."""
    if v is None:
        return {"score": 0, "components": {}, "missing": True}

    comps = {}

    def band(value, ranges):
        # ranges: list of (low, high, points), inclusive
        if value is None:
            return None
        for lo, hi, pts in ranges:
            if lo <= value <= hi:
                return pts
        return 3  # out of all bands → highest risk

    comps["respiratory_rate"] = band(v.respiratory_rate, [
        (12, 20, 0), (9, 11, 1), (21, 24, 2),
    ])
    comps["spo2"] = band(v.spo2, [
        (96, 100, 0), (94, 95, 1), (92, 93, 2),
    ])
    comps["systolic_bp"] = band(v.systolic_bp, [
        (111, 219, 0), (101, 110, 1), (91, 100, 2),
    ])
    comps["heart_rate"] = band(v.heart_rate, [
        (51, 90, 0), (41, 50, 1), (91, 110, 1), (111, 130, 2),
    ])
    comps["temperature"] = band(v.temperature, [
        (36.1, 38.0, 0), (35.1, 36.0, 1), (38.1, 39.0, 1), (39.1, 41.0, 2),
    ])
    consc_map = {"alert": 0, "voice": 3, "pain": 3, "unresponsive": 3}
    comps["consciousness"] = consc_map.get(
        (v.consciousness or "alert").lower(), 0
    )

    valid = [c for c in comps.values() if c is not None]
    return {
        "score": sum(valid),
        "components": comps,
        "missing": False,
    }


_SEVERITY_WEIGHT = {"mild": 1, "moderate": 2, "severe": 3, "critical": 4}

_RED_FLAG_SYMPTOMS = {
    "chest_pain", "shortness_of_breath", "loss_of_consciousness",
    "stroke_symptoms", "severe_bleeding", "anaphylaxis",
}


def _structured_symptom_score(symptoms: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not symptoms:
        return {"score": 0, "red_flags": [], "count": 0}

    score = 0
    red_flags = []
    for s in symptoms:
        name = (s.get("name") or "").lower()
        sev = (s.get("severity") or "mild").lower()
        score += _SEVERITY_WEIGHT.get(sev, 1)
        if name in _RED_FLAG_SYMPTOMS:
            red_flags.append(name)
            score += 3

    return {"score": score, "red_flags": red_flags, "count": len(symptoms)}


def _nlp_score(nlp: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """NLP findings come from the existing symptom_extractor.py."""
    if not nlp:
        return {"score": 0, "entities": 0}

    entities = nlp.get("entities", []) or nlp.get("symptoms", [])
    urgency_terms = nlp.get("urgency_terms", [])
    negations = nlp.get("negations", [])

    score = 0.5 * max(0, len(entities) - len(negations)) + 2 * len(urgency_terms)
    return {
        "score": round(score, 2),
        "entities": len(entities),
        "urgency_terms": urgency_terms,
        "negations": len(negations),
    }


def _image_score(img: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not img:
        return {"score": 0, "present": False}

    sev_score = float(img.get("severity_score", 0.0))   # 0..1 from DL head
    cv_flags = img.get("cv_flags", {}) or {}
    extra = 0
    if cv_flags.get("active_bleeding"):
        extra += 3
    if cv_flags.get("possible_infection"):
        extra += 2
    if cv_flags.get("large_area"):
        extra += 1

    score = sev_score * 5 + extra  # 0..~10
    return {
        "score": round(score, 2),
        "severity_score": sev_score,
        "infection_severity": img.get("infection_severity"),
        "wound_class": img.get("wound_class"),
        "cv_flags": cv_flags,
        "present": True,
    }


# ---------- Fusion ----------
def _esi_level(total: float, red_flags: List[str], vitals_score: int) -> int:
    """Map fused score → ESI 1..5."""
    if vitals_score >= 7 or "loss_of_consciousness" in red_flags or total >= 18:
        return 1   # resuscitation
    if vitals_score >= 5 or red_flags or total >= 12:
        return 2   # emergent
    if total >= 7:
        return 3   # urgent
    if total >= 3:
        return 4   # less urgent
    return 5       # non-urgent


_ESI_LABEL = {
    1: "RESUSCITATION — immediate, life-threatening",
    2: "EMERGENT — high risk, see within 10 min",
    3: "URGENT — see within 30 min",
    4: "LESS URGENT — see within 60 min",
    5: "NON-URGENT — see within 120 min",
}


# Logistic squashing for a 0–100 'composite_risk' indicator.
def _composite_risk(total: float) -> float:
    return round(100 / (1 + math.exp(-(total - 8) / 3)), 1)


def compute_triage(inp: TriageInput) -> Dict[str, Any]:
    vitals_part = _news2_vitals(inp.vitals)
    sym_part = _structured_symptom_score(inp.structured_symptoms)
    nlp_part = _nlp_score(inp.nlp_findings)
    img_part = _image_score(inp.image_analysis)

    total = (
        vitals_part["score"]
        + sym_part["score"]
        + nlp_part["score"]
        + img_part["score"]
    )

    level = _esi_level(total, sym_part["red_flags"], vitals_part["score"])

    # Per-component contribution (for SHAP-style display)
    contributions = {
        "vitals": vitals_part["score"],
        "structured_symptoms": sym_part["score"],
        "nlp_findings": nlp_part["score"],
        "image_analysis": img_part["score"],
    }

    return {
        "esi_level": level,
        "esi_label": _ESI_LABEL[level],
        "composite_risk": _composite_risk(total),  # 0..100
        "total_score": round(total, 2),
        "contributions": contributions,
        "components": {
            "vitals": vitals_part,
            "structured_symptoms": sym_part,
            "nlp_findings": nlp_part,
            "image_analysis": img_part,
        },
        "red_flags": sym_part["red_flags"],
        "input_summary": {
            "age": inp.age,
            "sex": inp.sex,
            "vitals_provided": inp.vitals is not None,
            "symptoms_provided": bool(inp.structured_symptoms),
            "nlp_provided": bool(inp.nlp_findings),
            "image_provided": bool(inp.image_analysis),
        },
    }


# Convenience: build TriageInput from raw dicts coming out of the API.
def from_payload(payload: Dict[str, Any]) -> TriageInput:
    v = payload.get("vitals")
    return TriageInput(
        vitals=Vitals(**v) if v else None,
        structured_symptoms=payload.get("structured_symptoms", []),
        nlp_findings=payload.get("nlp_findings"),
        image_analysis=payload.get("image_analysis"),
        age=payload.get("age"),
        sex=payload.get("sex"),
    )
