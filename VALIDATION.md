# Verification report

- Environment: Python 3.12.14
- Test command: `python -m pytest -q`
- Result: **40 passed**, 1 upstream Starlette/httpx deprecation warning, 8.19 seconds.
- Frontend: Streamlit AppTest exercised all six pages, clinical form submission over HTTP, result/warning display, history and clear-history behavior.
- Backend: real saved artifact inference and invalid/missing request cases verified; unavailable artifacts produce 503 readiness and prediction-service errors.
- Reports: CSV, JSON and TXT import verified, including unsupported content and upload-size rejection. Text-based PDF support uses pypdf; no real hospital PDF was supplied for layout validation. Scanned images/PDFs are unsupported.
- Reproducibility: saved metrics independently recalculated on the same held-out patients; saved imputer medians compared with training-only medians.
- Patient overlap: 0.
- Confusion matrix [TN FP; FN TP]: [[9287, 29], [210, 0]].
- Accuracy: 0.974911; precision: 0.000000; recall: 0.000000; F1: 0.000000; ROC-AUC: 0.645121.
- Majority-class baseline accuracy: 0.977955.

The application runs, but the trained model has **zero measured sepsis recall** at its default classification threshold. All 210 positive test observations were missed. This model is unsuitable for clinical decisions; the UI and API warn about this limitation. No target success rate was fabricated or obtained through test-set tuning.

No clinical validation, public deployment, GitHub push, arbitrary hospital-document extraction, or visual browser screenshot review is claimed. The original project-report document was not available. Source and methodology follow the written specification.
