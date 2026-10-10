from pathlib import Path
import os
ROOT = Path(__file__).resolve().parent
FEATURES = 'Age Gender Hour HR O2Sat Temp SBP MAP DBP Resp FiO2 pH PaCO2 SaO2 BUN Creatinine WBC Lactate Platelets Hgb Hct'.split()
TARGET = 'SepsisLabel'
RESPIRATORY = ['O2Sat', 'Resp', 'FiO2', 'PaCO2', 'SaO2', 'pH']
KIDNEY = ['BUN', 'Creatinine']
MODEL_DIR = Path(os.getenv('SEPSIS_MODEL_DIR', str(ROOT / 'models')))
MODERATE_THRESHOLD = float(os.getenv('SEPSIS_MODERATE_THRESHOLD', '0.30'))
HIGH_THRESHOLD = float(os.getenv('SEPSIS_HIGH_THRESHOLD', '0.70'))
DECISION_THRESHOLD = float(os.getenv('SEPSIS_DECISION_THRESHOLD', str(MODERATE_THRESHOLD)))
if not 0 < MODERATE_THRESHOLD < HIGH_THRESHOLD < 1:
    raise ValueError('Risk thresholds must satisfy 0 < moderate < high < 1')
if not 0 < DECISION_THRESHOLD < 1:
    raise ValueError('Decision threshold must satisfy 0 < threshold < 1')
DISCLAIMER = ('This application provides a machine-learning-based sepsis risk estimate from structured clinical data. '
 'It is intended for academic and research demonstration only and has not been clinically validated. '
 'It must not be used as a substitute for professional medical diagnosis, clinical judgement, or emergency care.')
UNITS = dict(zip(FEATURES, ['years','0 = female, 1 = male','hours from first observation','beats/min','%','°C','mmHg','mmHg','mmHg','breaths/min','fraction 0–1','pH','mmHg','%','mg/dL','mg/dL','10^3/µL','mmol/L','10^3/µL','g/dL','%']))
