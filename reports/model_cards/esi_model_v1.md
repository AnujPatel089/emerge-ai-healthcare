# Model Card: esi_model_v1

## Purpose
This model supports educational AI-supported triage by estimating possible risk and an ESI-style priority level. Clinician review required.

## Training Data
- Dataset: Existing project training data
- Training date: 2026-05-14T15:51:35.195746
- Feature columns: E:\Personal Project\Sem 2\emergency-triage-ai\models\emergency_feature_columns.pkl

## Features
Age, arrival mode, vital signs, and chief complaint flags are used where available. The model must not be used for final diagnosis or treatment decisions.

## Metrics
- accuracy: not available
- precision: not available
- recall: not available
- f1_score: not available
- recall_esi_1_2: not available

## Limitations
This is educational/demo healthcare software. It may be affected by synthetic data quality, missing values, workflow bias, and changes in patient mix.

## Safety Disclaimer
Outputs describe possible risk only. AI-supported triage requires clinician review and must not replace emergency medical judgment.

## Bias Considerations
Monitor performance across demographic and arrival-mode groups. Review override patterns and low-confidence predictions for potential inequity.

## Clinical Limitations
The model does not produce a final medical diagnosis. It does not interpret all clinical context, labs, imaging, or bedside examination findings.

## Monitoring Plan
Track prediction volume, confidence, clinician overrides, ESI distribution, latency, failures, and data drift.

## Retraining Trigger
Consider retraining when drift is warning/critical, override rate increases, low-confidence predictions rise, or recall for ESI 1/2 decreases.

## Render Deployment Notes
Render service disks can be ephemeral. Keep registry metadata, monitoring logs, drift reports, and model cards in PostgreSQL. Local files under `models/registry` and `reports/model_cards` are fallback artifacts only.

Generated at 2026-05-14T15:51:35.205756 UTC.
