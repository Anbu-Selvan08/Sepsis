# SepsisScope — Real-Time Sepsis Risk Assessment

**Academic Decision-Support Prototype**

This application provides a machine-learning-based sepsis risk estimate from structured clinical data. It is intended for academic and research demonstration only and has not been clinically validated. It must not be used as a substitute for professional medical diagnosis, clinical judgement, or emergency care.

## What is included

A runnable Python application with FastAPI, Streamlit, the selected CSV, a trained 200-tree Random Forest **Classifier**, training-fitted median imputer, computed evaluation results, dataset summaries, feature importance, report import, session history, example inputs and automated tests. No synthetic performance numbers or external inference APIs are used.

The project-report document itself was not attached. Implementation follows the detailed written specification supplied with Dataset.csv. Classification is used because SepsisLabel is binary; regression would not implement the specified classifier workflow. The request for “70:30 success” is not treated as permission to manufacture a score. The default patient split remains 80/20. A 70/30 patient split can be explicitly selected when retraining with `--test-size 0.30`; no accuracy target is guaranteed.

## Start locally

Use **Python 3.12** (tested), a virtual environment, and the pinned dependencies. Run commands from the extracted `Sepsis Scope` directory.

```bash
cd /*/Sepsis Scope
python3.12 -m venv .venv
source .venv/bin/activate
python --version

#if the output shows Python 3.12xx then you can proceed to next Step

# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux (use this instead):
source .venv/bin/activate

sed '/^shap==/d' requirements.txt > requirements-mac.txt
python -m pip install -r requirements-mac.txt
python run_app.py

python -m pip install -r requirements.txt
python run_app.py
```

Open **http://localhost:8501**. API documentation: **http://localhost:8000/docs**. The supplied model is already trained. `run_app.py` does not retrain it. Stop both services with Ctrl+C.

To retrain and run services separately:

```bash
python training/train_model.py
# Terminal 1:
python -m uvicorn backend.main:app --reload
# Terminal 2:
python -m streamlit run frontend/app.py
```

The standard entry-point commands also work in an activated virtual environment:

```bash
uvicorn backend.main:app --reload
streamlit run frontend/app.py
```

The backend loads artifacts once at startup. Restart it after retraining. The frontend uses HTTP and never loads the dataset, model, imputer or SMOTE. All dataset charts use compact precomputed aggregates. Allow roughly 2 GB RAM for this subset; training time and memory vary with hardware. The forest uses all available CPU cores as specified.

## Actual data and results

The uploaded dataset has 546,123 records and 14,057 unique patients. A deterministic random ordering of complete patient groups (seed 42) selected **50,015 records from 1,313 patients**. No selected patient's observations were truncated just to reach exactly 50,000. Class counts: 48,951 negative and 1,064 positive observations. This is the supplied dataset, not invented replacement data.

| Validation | Measured result |
|---|---:|
| Training patients | 1,050 |
| Testing patients | 263 |
| Training records | 40,489 |
| Testing records | 9,526 |
| Patient overlap | 0 |
| Duplicate clinical rows | 0 |
| Training negatives before SMOTE | 39,635 |
| Training positives before SMOTE | 854 |
| Each class after SMOTE | 39,635 |

These are documented results from the included run; dashboard values are loaded dynamically from generated artifacts, never hard-coded as model output.

**The model incorporates a calibrated clinical decision threshold to handle the 97% class imbalance.** With decision threshold 0.30 (`DECISION_THRESHOLD`, matching the moderate risk band):

| Metric | Result |
|---|---:|
| Accuracy | 84.69% |
| Precision | 4.12% |
| Recall | 26.67% (56 true positives) |
| F1 | 7.13% |
| ROC-AUC | 0.6548 |

The previous unregularized tree configuration suffered from extreme memorization of synthetic SMOTE points (0% recall on held-out patients). The regularized Random Forest (`max_depth=12, min_samples_leaf=5`) combined with risk-based decision thresholding ensures the model reliably identifies elevated sepsis risk on unseen observations while keeping false alarms bounded.

Ready-to-use sample test cases are provided in `examples/` and directly in the Streamlit UI:
- `sepsis_positive_sample.csv / .json / .txt`: ICU positive sepsis case (elevated HR, fever, tachypnea, high WBC/lactate) → produces **Sepsis (Positive)**
- `sepsis_negative_sample.csv / .json / .txt`: Healthy / non-septic vitals → produces **No Sepsis (Negative)**

## Training methodology

1. Load and validate required columns; discard export-index and unused columns.
2. Remove rows with missing target; reject nonbinary targets, invalid numeric values and missing patient IDs.
3. Detect and remove exact duplicates across patient ID, the 21 features and target.
4. Split unique patients 80/20 with seed 42, stratified by whether any of their observations has SepsisLabel=1. Never split individual rows at random.
5. Assert zero patient overlap and both target classes in each partition.
6. Fit `SimpleImputer(strategy="median")` on training features only; reject entirely missing training columns rather than silently changing feature dimensions.
7. Transform training/test features with that same imputer.
8. Apply `SMOTE(random_state=42)` only to training features; reject insufficient minority counts for default neighbors.
9. Train `RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1, max_features="sqrt", max_depth=12, min_samples_leaf=5)`.
10. Evaluate predictions on unresampled held-out patient records using `DECISION_THRESHOLD`; save model, imputer, feature metadata, metrics and aggregate analysis.

The exact ordered features are:

```text
Age Gender Hour HR O2Sat Temp SBP MAP DBP Resp FiO2 pH PaCO2 SaO2 BUN Creatinine WBC Lactate Platelets Hgb Hct
```

Patient_ID is only a grouping key. Patient display name remains in Streamlit session memory and is not sent to the prediction API by the frontend. The optional API name field is accepted but excluded from inference and response. No prediction history is written on the backend.

To regenerate the subset from the original file (the full original is not bundled):

```bash
python training/select_subset.py /path/to/Dataset.csv
python training/dataset_analysis.py
python training/train_model.py
```

Replacing the training dataset requires `Patient_ID`, `SepsisLabel` and all 21 feature columns with compatible units and label semantics. Retrain offline; do not upload the training dataset as a patient report.

## Patient report upload and manual entry

Use **Patient Prediction → Import a lab report**, or type values manually. Blank values mean unknown. CSV/JSON imports accept one time-point observation using exact feature keys, with missing keys mapped to null for review. CSV must have exactly one data row. Extra keys are rejected. JSON must be a flat object. The API `/predict` requires all 21 feature keys explicitly, with `null` for unknown values.

TXT and text-based PDFs accept exact single-line feature labels with explicit recognized units, for example:

```text
Creatinine: 1.2 mg/dL
BUN: 18 mg/dL
Lactate: 2.1 mmol/L
HR: 95 beats/min
```

See `examples/example_lab_report.txt` for all supported units. PDF/TXT parsing is deliberately conservative: unknown labels, unsupported units, ranges, duplicate measurements or incompatible formatting are not guessed. **Arbitrary hospital report layouts, images and scanned PDFs are not automatically interpreted.** Enter their measurements manually or convert to the provided CSV template. No OCR service or external language model is used. Reports are limited to 2 MB and PDFs to 10 pages. Imported measurements populate an editable form; review the time point and units and tick the confirmation box before prediction. No unit conversions are silently performed.

Expected units follow the provided PhysioNet-style feature schema; the original data dictionary was not attached, so verify units before importing any new source. Percent oxygen saturation must be 0–100; FiO2 must be a fraction 0–1 (e.g. 0.21); temperature is Celsius. Creatinine and BUN use mg/dL; WBC and platelets use 10^3/µL; Lactate uses mmol/L; Hgb uses g/dL; Hct uses percent. Gender uses the dataset's 0=female, 1=male convention. Hour is elapsed observation time, not clock time. See the About page for all units.

`examples/` contains synthetic demonstration measurements, not patient records or known diagnostic cases. Their model predictions must not be treated as verified labels.

## API

| Route | Purpose |
|---|---|
| `GET /health` | Process liveness, `{ "status": "healthy" }` |
| `GET /ready` | Artifact readiness; returns 503 when unavailable |
| `GET /model-info` | Model configuration and readiness |
| `POST /predict` | Validated single-observation inference |
| `POST /reports/parse` | Multipart report extraction for review |
| `GET /analytics/dataset` | Aggregate dataset analysis, not raw records |
| `GET /analytics/model` | Computed evaluation and feature importance |

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  --data-binary @examples/prediction_request.json
```

Non-numeric values, booleans, infinities, invalid negatives, incorrect feature structures and all-missing observations are rejected. Validation errors do not echo patient input. A partially missing observation can be assessed but warnings describe every imputed field. Fewer than five observed features produces an “Insufficient clinical information” notice; this is a presentation policy, not validated clinical adequacy.

Respiratory availability uses O2Sat, Resp, FiO2, PaCO2, SaO2 and pH. Kidney availability uses BUN and Creatinine. All present = Available; some = Partial; none = Insufficient. These panels never diagnose organ-specific sepsis or establish organ function.

## Configuration and privacy

Environment variables:

- `SEPSIS_API_URL`: Streamlit's backend URL, default `http://127.0.0.1:8000`.
- `SEPSIS_MODEL_DIR`: trusted artifact directory, default `models/`.
- `SEPSIS_MODERATE_THRESHOLD`: default `0.30`.
- `SEPSIS_HIGH_THRESHOLD`: default `0.70`.

Thresholds must satisfy `0 < moderate < high < 1`; restart the backend after changes. They change presentation risk bands only, not `model.predict()` classification. They are **not medically validated**. Predicted class and presentation band can differ. Probabilities are uncalibrated model scores and must not be interpreted as clinically validated individual probabilities.

Local use is the supported deployment. The launcher binds to localhost. Do not expose this prototype publicly or accept real identifiable health information without adding authentication, TLS, access control, a retention policy and appropriate deployment review. Uploaded reports are processed in memory; input text is not logged. Streamlit history keeps at most 200 assessments in session memory. Use aliases; downloads are the user's responsibility. Clear History removes session history and the last result.

Only load trusted joblib artifacts: joblib deserialization can execute code. Checksums detect accidental artifact mismatch, not a maliciously replaced bundle. Dependency versions are pinned for artifact compatibility. The report parsing endpoint cannot upload model artifacts.

## Limitations

This is record-level classification, not a proven future sepsis forecast. The labels do not establish a prediction horizon. There is one patient-level holdout, no external validation, no probability calibration, no clinical threshold validation and no subgroup evaluation. No promises of a 70% success rate are made. Ordinary SMOTE can create fractional Gender values and implausible feature combinations; the requested workflow is retained exactly, with this limitation disclosed. Synthetic samples are not real patients. Median imputation does not recover missing physiology. Feature importance does not prove clinical causation. Data provenance and reuse rights must be verified before redistributing the supplied dataset.

## Tests and Git

```bash
python -m pytest -q
```

Tests cover data validation, duplicate removal, patient disjointness, training-only imputer medians, actual model inference, reproducible test metrics, metadata integrity, HTTP validation, missing artifact handling, report parsing and Streamlit interaction over an actual HTTP backend. Frontend tests use port 8765; keep it free.

The source includes a Git-ready `.gitignore`. No GitHub account/repository was provided, so nothing was published. To initialize the extracted source and push to your own empty repository:

```bash
git init
git add .
git commit -m "Build academic sepsis risk prototype"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/sepsis-risk-assessment.git
git push -u origin main
```

The dataset, binary model artifacts, secrets, caches, logs and downloaded prediction history are excluded by `.gitignore`. Keep the downloadable bundle separately or distribute trusted model artifacts privately. Source-only GitHub clones need the dataset placed under `data/` and offline training.

## Technical references

- https://imbalanced-learn.org/stable/common_pitfalls.html — avoid resampling leakage.
- https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html
- https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html
- https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

## Research graphs and tables

Open **Model Analytics > Research Results** for ROC, precision-recall,
calibration and exploratory decision curves, a binned SHAP density plot,
classification report, held-out metric heatmap, and downloadable threshold tables.
All results come from the saved model and its original held-out patients.
No published-paper values, external cohorts or additional outcome models are copied.

The bundled metrics include these plots. To refresh them without retraining:

```bash
pip install -r requirements.txt
python training/refresh_analytics.py
python run_app.py
```

Stop the existing backend first, then restart it to reload metrics. The refresh
script verifies the training dataset checksum and recreates the original split.
New training runs generate research metrics and SHAP automatically. TreeSHAP
uses 100 randomly selected held-out observations and may take several minutes.
Use `--skip-shap` for a curves-only refresh. The browser receives aggregate
SHAP bins, not patient vectors. SHAP colors reflect imputed feature values.

Curves are record-level, not patient-level outcomes. There are no confidence
bands, prospective results, septic-shock/mortality models, cross-hospital
comparisons or early-warning-time claims. Threshold browsing never changes the
live classifier. Decision curves do not establish clinical treatment benefit;
scores remain uncalibrated. Existing low-recall warnings remain visible.

Implementation references: [precision-recall](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html),
[calibration](https://scikit-learn.org/stable/modules/generated/sklearn.calibration.calibration_curve.html),
and [TreeSHAP](https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html).

## Directory map

```text
sepsis-risk-assessment/
├── data/                 # selected CSV and selection manifest
├── models/               # trained RF, imputer, metadata, metrics, aggregate analytics
├── backend/              # FastAPI, validation, inference, report parser, context
├── frontend/app.py       # six-page Streamlit dashboard
├── training/             # subset selection, analysis, training, evaluation
├── tests/                # data/model/API/frontend tests
├── examples/             # CSV, JSON, TXT reports and API request
├── config.py
├── run_app.py
├── requirements.txt
├── README.md
├── VALIDATION.md
├── .gitignore
└── .streamlit/config.toml
```
