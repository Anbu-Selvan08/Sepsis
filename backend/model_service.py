import hashlib, json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from config import FEATURES, TARGET, MODEL_DIR, MODERATE_THRESHOLD, HIGH_THRESHOLD, DECISION_THRESHOLD
from backend.utils import input_context, risk_level

class ModelService:
    def __init__(self, directory=MODEL_DIR):
        self.directory=directory
        self.decision_threshold=DECISION_THRESHOLD
        self.metadata=json.loads((directory/'feature_metadata.json').read_text())
        if self.metadata['features']!=FEATURES or self.metadata['target']!=TARGET:
            raise ValueError('Model feature metadata mismatch')
        for name in ['random_forest.joblib','imputer.joblib']:
            if hashlib.sha256((directory/name).read_bytes()).hexdigest()!=self.metadata['artifact_sha256'][name]:
                raise ValueError('Artifact checksum mismatch')
        # Only load trusted locally trained artifacts; joblib can execute code.
        self.model=joblib.load(directory/'random_forest.joblib')
        self.imputer=joblib.load(directory/'imputer.joblib')
        if not isinstance(self.model,RandomForestClassifier) or not isinstance(self.imputer,SimpleImputer):
            raise ValueError('Unexpected artifact type')
        if list(self.imputer.feature_names_in_)!=FEATURES or self.model.n_features_in_!=21 or list(self.model.classes_)!=[0,1]:
            raise ValueError('Artifact schema mismatch')
        if self.model.n_estimators!=200 or self.imputer.strategy!='median':
            raise ValueError('Artifact configuration mismatch')
        self.metrics=json.loads((directory/'metrics.json').read_text())
        self.analysis=json.loads((directory/'dataset_analysis.json').read_text())

    def predict(self, values):
        if set(values)!=set(FEATURES): raise ValueError('Exactly 21 clinical feature keys are required')
        frame=pd.DataFrame([[np.nan if values[f] is None else values[f] for f in FEATURES]],columns=FEATURES)
        transformed=self.imputer.transform(frame)
        probability=float(self.model.predict_proba(transformed)[0,1])
        predicted=int(probability>=self.decision_threshold)
        context=input_context(values)
        if self.metrics.get('recall', 0.0)<0.5:
            context['warnings'].insert(0,f"Held-out sepsis recall is {self.metrics['recall']:.1%}. A low score or No Sepsis output must not be used to rule out sepsis.")
        return dict(prediction='Sepsis' if predicted else 'No Sepsis',predicted_class=predicted,
                    probability=probability,risk_level=risk_level(probability),**context,
                    thresholds={'moderate':MODERATE_THRESHOLD,'high':HIGH_THRESHOLD,'decision':self.decision_threshold},model_version=self.metadata['created_at'])
