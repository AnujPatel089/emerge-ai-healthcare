"""
PDF Clinical Report Generator

Builds a clinician-facing PDF summarising a single prediction_id:
  - Patient header
  - Vitals table
  - Symptoms (structured + NLP)
  - Image analysis (with embedded image if available)
  - Multi-modal triage score + component breakdown
  - SHAP top contributors
  - Audit footer (prediction_id, timestamp, clinician)
"""

import io
import os
from datetime import datetime
from typing import Dict, Any, Optional
from xml.sax.saxutils import escape

from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image as RLImage, PageBreak,
)


# ---------- Styling ----------
def _styles():
    base = getSampleStyleSheet()
    base.add(ParagraphStyle(
        name="ReportTitle", fontName="Helvetica-Bold",
        fontSize=18, leading=22, textColor=colors.HexColor("#0b3d91"),
        spaceAfter=12,
    ))
    base.add(ParagraphStyle(
        name="Section", fontName="Helvetica-Bold",
        fontSize=12, leading=16, textColor=colors.HexColor("#0b3d91"),
        spaceBefore=12, spaceAfter=6,
    ))
    base.add(ParagraphStyle(
        name="Body", fontName="Helvetica", fontSize=10, leading=13,
    ))
    base.add(ParagraphStyle(
        name="Mono", fontName="Courier", fontSize=8, leading=10,
        textColor=colors.HexColor("#444444"),
    ))
    return base


_TRIAGE_COLOR = {
    1: colors.HexColor("#b00020"),   # red
    2: colors.HexColor("#e65100"),   # orange
    3: colors.HexColor("#f9a825"),   # amber
    4: colors.HexColor("#2e7d32"),   # green
    5: colors.HexColor("#1565c0"),   # blue
}


def _kv_table(data: Dict[str, Any], col_widths=(2.0 * inch, 4.0 * inch)) -> Table:
    rows = [[str(k), "—" if v is None else str(v)] for k, v in data.items()]
    t = Table(rows, colWidths=col_widths)
    t.setStyle(TableStyle([
        ("FONT", (0, 0), (-1, -1), "Helvetica", 9),
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f0f4fa")),
        ("FONT", (0, 0), (0, -1), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cccccc")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return t


def _triage_banner(triage: Dict[str, Any], styles) -> Table:
    level = triage["esi_level"]
    color = _TRIAGE_COLOR.get(level, colors.grey)
    text = (
        f"<b>ESI Level {level}</b> &nbsp;|&nbsp; "
        f"{triage['esi_label']} <br/>"
        f"Composite Risk: <b>{triage['composite_risk']}</b> / 100 &nbsp;|&nbsp; "
        f"Total Score: {triage['total_score']}"
    )
    p = Paragraph(text, ParagraphStyle(
        name="Banner", fontName="Helvetica", fontSize=12, leading=16,
        textColor=colors.white,
    ))
    t = Table([[p]], colWidths=[6.5 * inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), color),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("RIGHTPADDING", (0, 0), (-1, -1), 12),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
    ]))
    return t


# ---------- Public API ----------
def generate_report(
    prediction_id: str,
    patient: Dict[str, Any],
    vitals: Optional[Dict[str, Any]],
    symptoms: Optional[Dict[str, Any]],
    image_analysis: Optional[Dict[str, Any]],
    triage: Dict[str, Any],
    shap_values: Optional[Dict[str, float]] = None,
    image_bytes: Optional[bytes] = None,
    historical_context: Optional[Dict[str, Any]] = None,
    clinician: str = "system",
    output_path: Optional[str] = None,
) -> str:
    """
    Render the report PDF. Returns the file path.
    If output_path is None, writes to ./reports/<prediction_id>.pdf
    """
    output_path = output_path or os.path.join("reports", f"{prediction_id}.pdf")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    styles = _styles()
    doc = SimpleDocTemplate(
        output_path, pagesize=LETTER,
        leftMargin=0.6 * inch, rightMargin=0.6 * inch,
        topMargin=0.6 * inch, bottomMargin=0.6 * inch,
        title=f"EmergeAI Clinical Report {prediction_id}",
    )
    story = []

    # Header
    story.append(Paragraph("EmergeAI Healthcare — Clinical Report", styles["ReportTitle"]))
    story.append(Paragraph(
        f"Generated {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')} &nbsp;|&nbsp; "
        f"Prediction ID: <b>{prediction_id}</b>",
        styles["Body"],
    ))
    story.append(Spacer(1, 0.15 * inch))

    # Triage banner up top so clinicians see the headline first
    story.append(_triage_banner(triage, styles))
    story.append(Spacer(1, 0.15 * inch))

    # Patient
    story.append(Paragraph("Patient", styles["Section"]))
    story.append(_kv_table({
        "Name": patient.get("name"),
        "MRN": patient.get("mrn"),
        "Age": patient.get("age"),
        "Sex": patient.get("sex"),
        "Arrival": patient.get("arrival_time"),
    }))

    # Vitals
    story.append(Paragraph("Vitals", styles["Section"]))
    if vitals:
        story.append(_kv_table({
            "Heart Rate (bpm)": vitals.get("heart_rate"),
            "Systolic BP (mmHg)": vitals.get("systolic_bp"),
            "Respiratory Rate": vitals.get("respiratory_rate"),
            "SpO2 (%)": vitals.get("spo2"),
            "Temperature (C)": vitals.get("temperature"),
            "Consciousness": vitals.get("consciousness"),
        }))
    else:
        story.append(Paragraph("<i>No vitals recorded.</i>", styles["Body"]))

    # Symptoms
    story.append(Paragraph("Symptoms & NLP Findings", styles["Section"]))
    if symptoms:
        struct = symptoms.get("structured", []) or []
        nlp = symptoms.get("nlp", {}) or {}
        if struct:
            rows = [["Symptom", "Severity"]]
            rows += [[s.get("name", ""), s.get("severity", "")] for s in struct]
            t = Table(rows, colWidths=[3 * inch, 3 * inch])
            t.setStyle(TableStyle([
                ("FONT", (0, 0), (-1, -1), "Helvetica", 9),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0b3d91")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cccccc")),
            ]))
            story.append(t)
            story.append(Spacer(1, 0.1 * inch))
        if nlp:
            urgency = nlp.get("urgency_terms") or []
            story.append(Paragraph(
                f"<b>NLP entities:</b> {len(nlp.get('entities', []))} &nbsp;|&nbsp; "
                f"<b>Urgency terms:</b> {', '.join(urgency) or '—'} "
                f"&nbsp;|&nbsp; <b>Negations:</b> {len(nlp.get('negations', []))}",
                styles["Body"],
            ))
    else:
        story.append(Paragraph("<i>No symptoms captured.</i>", styles["Body"]))

    # Image
    story.append(Paragraph("Image Analysis", styles["Section"]))
    if image_analysis:
        if image_bytes:
            try:
                img_buf = io.BytesIO(image_bytes)
                story.append(RLImage(img_buf, width=2.6 * inch, height=2.6 * inch))
                story.append(Spacer(1, 0.08 * inch))
            except Exception:
                pass  # corrupt image — skip thumbnail

        # FIX: use 'or 0' to handle None values stored explicitly in the dict
        wound_conf = image_analysis.get("wound_confidence") or 0
        severity_score = image_analysis.get("severity_score") or 0

        story.append(_kv_table({
            "Wound class": image_analysis.get("wound_class") or "—",
            "Wound confidence": f"{wound_conf:.2%}",
            "Infection severity": image_analysis.get("infection_severity") or "—",
            "Severity score (0-1)": f"{severity_score:.3f}",
            "Model": image_analysis.get("model") or "—",
        }))
    else:
        story.append(Paragraph("<i>No image submitted.</i>", styles["Body"]))

    # Triage breakdown
    story.append(Paragraph("Multi-Modal Triage Breakdown", styles["Section"]))
    contribs = triage.get("contributions", {})
    if contribs:
        rows = [["Modality", "Score Contribution"]]
        rows += [[k.replace("_", " ").title(), f"{v:.2f}"] for k, v in contribs.items()]
        t = Table(rows, colWidths=[3 * inch, 3 * inch])
        t.setStyle(TableStyle([
            ("FONT", (0, 0), (-1, -1), "Helvetica", 9),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0b3d91")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cccccc")),
        ]))
        story.append(t)
    else:
        story.append(Paragraph("<i>No triage breakdown available.</i>", styles["Body"]))

    if triage.get("red_flags"):
        story.append(Spacer(1, 0.08 * inch))
        story.append(Paragraph(
            f"<b>Red flags:</b> {', '.join(triage['red_flags'])}",
            ParagraphStyle("rf", parent=styles["Body"],
                           textColor=colors.HexColor("#b00020"),
                           fontName="Helvetica-Bold"),
        ))

    # SHAP
    if shap_values:
        story.append(Paragraph("Top SHAP Contributors", styles["Section"]))
        sorted_items = sorted(
            shap_values.items(), key=lambda kv: abs(kv[1]), reverse=True
        )[:8]
        rows = [["Feature", "SHAP Value"]]
        rows += [[k, f"{v:+.3f}"] for k, v in sorted_items]
        t = Table(rows, colWidths=[3.5 * inch, 2.5 * inch])
        t.setStyle(TableStyle([
            ("FONT", (0, 0), (-1, -1), "Helvetica", 9),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0b3d91")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cccccc")),
        ]))
        story.append(t)

    # Uploaded history context
    if historical_context:
        story.append(Paragraph("Uploaded Report / Image Context", styles["Section"]))
        risk_flags = historical_context.get("risk_flags", []) or []
        summary = historical_context.get("clinical_summary", {}) or {}
        metadata = historical_context.get("image_metadata", {}) or {}
        story.append(_kv_table({
            "Uploaded file": historical_context.get("file_name"),
            "Report/image type": historical_context.get("report_type"),
            "File type": historical_context.get("file_type"),
            "Risk flags": ", ".join(flag.get("label", "") for flag in risk_flags) or "No major flag detected",
            "Image quality": " | ".join(historical_context.get("image_quality_notes", []) or []) or "No image quality warning detected",
        }))
        story.append(Spacer(1, 0.08 * inch))
        story.append(Paragraph(
            escape(summary.get("summary_text") or summary.get("patient_background") or "No uploaded context summary available."),
            styles["Body"],
        ))
        if historical_context.get("ocr_text"):
            story.append(Paragraph("OCR Findings", styles["Section"]))
            story.append(Paragraph(escape(str(historical_context.get("ocr_text"))[:1000]), styles["Body"]))
        if metadata:
            story.append(Paragraph("Image Metadata", styles["Section"]))
            story.append(_kv_table({k: v for k, v in metadata.items()}))

    # Footer / audit
    story.append(Spacer(1, 0.3 * inch))
    story.append(Paragraph(
        f"Audit: prediction_id={prediction_id} · generated_by={clinician} · "
        f"engine=EmergeAI v1 · This report is decision support, not a diagnosis.",
        styles["Mono"],
    ))

    doc.build(story)
    return output_path
