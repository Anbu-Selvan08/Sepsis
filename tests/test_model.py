import json,shutil
import numpy as np
import pandas as pd
import pytest
from config import FEATURES,ROOT
from backend.model_service import ModelService
from backend.utils import risk_level
from training.dataset_analysis import load_dataset
from training.train_model import patient_split
from training.evaluate_model import evaluate

def test_model_loading(model):
    assert model.model.n_estimators==200 and model.model.n_features_in_==21
    assert model.imputer.strategy=='median'

def test_feature_order(model,values):
    assert model.predict(values)==model.predict(dict(reversed(list(values.items()))))
    with pytest.raises(ValueError):model.predict({**values,'Patient_ID':'123'})

def test_prediction(model,values):
    result=model.predict(values)
    assert result['prediction'] in ['Sepsis','No Sepsis'] and 0<=result['probability']<=1
    transformed=model.imputer.transform(pd.DataFrame([values],columns=FEATURES))
    assert result['probability']==model.model.predict_proba(transformed)[0,1]
    assert result['predicted_class']==model.model.predict(transformed)[0]

def test_missing_context(model,values):
    result=model.predict({**values,'Creatinine':None})
    assert result['kidney_status']=='Partial' and 'Creatinine' in result['missing_features']
    result=model.predict({**values,'Creatinine':None,'BUN':None})
    assert result['kidney_status']=='Insufficient'
    assert result['lung_status']=='Available'

def test_corrupt_artifact(tmp_path):
    shutil.copytree(ROOT/'models',tmp_path/'models')
    path=tmp_path/'models/feature_metadata.json';data=json.loads(path.read_text());data['features'].reverse();path.write_text(json.dumps(data))
    with pytest.raises(ValueError):ModelService(tmp_path/'models')

def test_missing_artifacts(tmp_path):
    with pytest.raises(FileNotFoundError):ModelService(tmp_path)

def test_thresholds():
    assert risk_level(.29)=='Low' and risk_level(.30)=='Moderate' and risk_level(.70)=='High'

def test_metrics_reproducible(model):
    _,test=patient_split(load_dataset(ROOT/'data/sepsis_selected.csv'))
    result=evaluate(model.model,model.imputer.transform(test[FEATURES]),test.SepsisLabel)
    for key in ['accuracy','precision','recall','f1','roc_auc','confusion_matrix']:assert result[key]==model.metrics[key]
