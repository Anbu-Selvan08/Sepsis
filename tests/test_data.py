import numpy as np
import pandas as pd
import pytest
from config import ROOT, FEATURES, TARGET
from training.dataset_analysis import load_dataset, clean_dataset, analyze
from training.train_model import patient_split

@pytest.fixture(scope='module')
def data():return load_dataset(ROOT/'data/sepsis_selected.csv')

def test_loading(data):
    assert 49000<len(data)<51000
    assert set(data)==set(FEATURES+[TARGET,'Patient_ID'])
    assert data.Patient_ID.dtype.name=='string'

def test_target_validation(data):
    bad=data.head(10).copy();bad.loc[0,TARGET]=2
    with pytest.raises(ValueError):clean_dataset(bad)

def test_missing_target(data):
    bad=data.head(10).copy();bad.loc[0,TARGET]=np.nan
    clean,checks=clean_dataset(bad)
    assert len(clean)==9 and checks['removed_missing_target']==1

def test_duplicates(data):
    duplicate=pd.concat([data.head(10),data.head(1)],ignore_index=True)
    clean,checks=clean_dataset(duplicate)
    assert checks['duplicate_rows']==1 and len(clean)==10

def test_patient_leakage(data):
    train,test=patient_split(data)
    assert set(train.Patient_ID).isdisjoint(test.Patient_ID)
    assert len(train)+len(test)==len(data)
    assert abs(test.Patient_ID.nunique()/data.Patient_ID.nunique()-.2)<.01

def test_missing_patient(data):
    bad=data.head(10).copy();bad.loc[0,'Patient_ID']=pd.NA
    with pytest.raises(ValueError):clean_dataset(bad)

def test_analysis_missing(data):
    result=analyze(data)
    assert result['missing_values']==int(data[FEATURES].isna().sum().sum())
    assert len(result['missing'])==21

def test_imputer_fitted_train_only(data,model):
    train,test=patient_split(data)
    np.testing.assert_allclose(model.imputer.statistics_,train[FEATURES].median().values)
    assert model.metrics['split']['testing_records']==len(test)
    assert model.metrics['before_smote']=={str(int(k)):int(v) for k,v in train.SepsisLabel.value_counts().items()}
    assert len(set(model.metrics['after_smote'].values()))==1
