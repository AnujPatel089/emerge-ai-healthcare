"""
NLP Symptom Extractor
Emergency Triage AI
"""

import re


SYMPTOM_KEYWORDS = {
    "cc_chestpain": [
        "chest pain",
        "chest tightness",
        "heart pain",
        "pressure in chest",
        "chest pressure",
        "pain in chest"
    ],

    "cc_shortnessofbreath": [
        "shortness of breath",
        "difficulty breathing",
        "breathing problem",
        "trouble breathing",
        "can't breathe",
        "cannot breathe",
        "sob"
    ],

    "cc_headache": [
        "headache",
        "head pain",
        "migraine",
        "pain in head"
    ],

    "cc_fever": [
        "fever",
        "high temperature",
        "chills",
        "hot body",
        "temperature"
    ],

    "cc_abdominalpain": [
        "abdominal pain",
        "stomach pain",
        "belly pain",
        "pain in stomach",
        "abdomen pain"
    ],

    "cc_dizziness": [
        "dizzy",
        "dizziness",
        "lightheaded",
        "light headed",
        "feeling faint"
    ],

    "cc_syncope": [
        "syncope",
        "fainting",
        "fainted",
        "passed out",
        "loss of consciousness",
        "unconscious"
    ],

    "cc_weakness": [
        "weakness",
        "weak",
        "numbness",
        "fatigue",
        "unable to move"
    ]
}


EMERGENCY_KEYWORDS = [
    "severe chest pain",
    "heart attack",
    "can't breathe",
    "cannot breathe",
    "difficulty breathing",
    "not breathing",
    "stroke",
    "seizure",
    "unconscious",
    "not responding",
    "loss of consciousness",
    "heavy bleeding",
    "blue lips",
    "suicidal",
    "overdose"
]


def preprocess_clinical_text(text: str) -> str:
    """
    Clean problem description text for NLP / LLM-ready preprocessing.
    """

    if not text:
        return ""

    text = text.lower()
    text = text.replace("can’t", "can't")
    text = text.replace("cannot", "cannot")

    text = re.sub(r"[^a-zA-Z0-9\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def extract_symptoms(text: str) -> dict:
    """
    Extract symptom checkbox values and emergency keywords
    from typed/audio patient description.
    """

    cleaned_text = preprocess_clinical_text(text)

    extracted_symptoms = {}
    matched_terms = []

    for symptom_field, keywords in SYMPTOM_KEYWORDS.items():
        found = False

        for keyword in keywords:
            if keyword in cleaned_text:
                found = True
                matched_terms.append(keyword)

        extracted_symptoms[symptom_field] = 1 if found else 0

    emergency_keywords = []

    for keyword in EMERGENCY_KEYWORDS:
        if keyword in cleaned_text:
            emergency_keywords.append(keyword)

    return {
        "cleaned_text": cleaned_text,
        "extracted_symptoms": extracted_symptoms,
        "matched_terms": sorted(list(set(matched_terms))),
        "emergency_keywords": sorted(list(set(emergency_keywords))),
        "has_emergency_keyword": len(emergency_keywords) > 0,
        "llm_ready_text": cleaned_text
    }