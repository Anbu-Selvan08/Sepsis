import pytest
from fastapi.testclient import TestClient
from backend.main import app
from config import FEATURES,ROOT

@pytest.fixture(scope='module')
def client():
    with TestClient(app) as c:yield c

def test_health_info(client):
    assert client.get('/health').json()=={'status':'healthy'}
    assert client.get('/model-info').json()['features']==21
    assert client.get('/ready').status_code==200

def test_prediction(client,values):
    result=client.post('/predict',json={'features':values,'patient_name':'Demo alias'})
    assert result.status_code==200
    assert 'Demo alias' not in result.text

@pytest.mark.parametrize('field,value',[('Age',-1),('HR','fast'),('Gender',2),('FiO2',40),('O2Sat',101),('HR',True)])
def test_invalid(client,values,field,value):
    response=client.post('/predict',json={'features':{**values,field:value}})
    assert response.status_code==422 and 'input' not in response.text

def test_missing_structure(client):
    assert client.post('/predict',json={}).status_code==422
    assert client.post('/predict',json={'features':{'Age':34}}).status_code==422
    assert client.post('/predict',json={'features':{f:None for f in FEATURES}}).status_code==422

def test_unexpected_feature(client,values):
    assert client.post('/predict',json={'features':{**values,'Patient_ID':'1'}}).status_code==422

def test_missing_allowed(client,values):
    result=client.post('/predict',json={'features':{**values,'Creatinine':None}})
    assert result.status_code==200 and result.json()['kidney_status']=='Partial'

def test_failure(client):
    original=app.state.service
    try:
        app.state.service=None
        assert client.get('/ready').status_code==503
        assert client.get('/health').status_code==200
        assert client.get('/analytics/model').status_code==503
    finally:app.state.service=original

def test_startup_artifact_failure(monkeypatch,tmp_path):
    import backend.main as main
    monkeypatch.setattr(main,'MODEL_DIR',tmp_path)
    with TestClient(app) as c:assert c.get('/ready').status_code==503
    # Restore shared app service after nested lifespan.
    from backend.model_service import ModelService
    app.state.service=ModelService(ROOT/'models')

@pytest.mark.parametrize('extension',['csv','json','txt'])
def test_report(client,extension):
    content=(ROOT/f'examples/example_lab_report.{extension}').read_bytes()
    response=client.post('/reports/parse',files={'file':('lab.'+extension,content)})
    assert response.status_code==200 and response.json()['features']['Creatinine']==1.1
    assert response.json()['review_required']

def test_report_bad(client):
    for name,content in [('lab.txt',b'Creatinine: 12 bananas'),('lab.csv',b'Age\n20\n30'),('lab.json',b'{"unknown":1}')]:
        assert client.post('/reports/parse',files={'file':(name,content)}).status_code==422
    assert client.post('/reports/parse',files={'file':('lab.txt',b'x'*(2*1024*1024+1))}).status_code==413

def test_analytics(client):
    assert client.get('/analytics/dataset').json()['total_records']==50015
    assert client.get('/analytics/model').json()['split']['patient_overlap']==0

def test_positive_and_negative_api(client):
    import json
    pos_data=json.loads((ROOT/'examples/sepsis_positive_sample.json').read_text())
    neg_data=json.loads((ROOT/'examples/sepsis_negative_sample.json').read_text())
    res_pos=client.post('/predict',json={'features':pos_data})
    res_neg=client.post('/predict',json={'features':neg_data})
    assert res_pos.status_code==200
    assert res_pos.json()['prediction']=='Sepsis' and res_pos.json()['predicted_class']==1
    assert res_neg.status_code==200
    assert res_neg.json()['prediction']=='No Sepsis' and res_neg.json()['predicted_class']==0
