"""Streamlit integration tests against an actual local HTTP backend."""
import os,subprocess,sys,time
import requests
import pytest
from streamlit.testing.v1 import AppTest
from config import ROOT

@pytest.fixture(scope='module',autouse=True)
def server():
    process=subprocess.Popen([sys.executable,'-m','uvicorn','backend.main:app','--port','8765','--host','127.0.0.1'],cwd=ROOT,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    original=os.environ.get('SEPSIS_API_URL');os.environ['SEPSIS_API_URL']='http://127.0.0.1:8765'
    try:
        for _ in range(200):
            if process.poll() is not None:raise RuntimeError('Test backend failed to start')
            try:
                if requests.get('http://127.0.0.1:8765/ready',timeout=1).ok:break
            except requests.RequestException:pass
            time.sleep(.1)
        else:raise RuntimeError('Test backend not ready')
        yield
    finally:
        process.terminate();process.wait(timeout=10)
        if original is None:os.environ.pop('SEPSIS_API_URL',None)
        else:os.environ['SEPSIS_API_URL']=original

@pytest.mark.parametrize('page',['Dashboard','Dataset Analysis','Model Analytics','Prediction History','About'])
def test_pages(page):
    at=AppTest.from_file(str(ROOT/'frontend/app.py'),default_timeout=30).run()
    at.sidebar.radio[0].set_value(page).run()
    assert not at.exception
    if page in ['Dashboard','Model Analytics']:assert len(at.metric)>0
    if page == 'Model Analytics':
        assert any(t.label == 'Evaluation curves' for t in at.tabs)
        assert len(at.get('plotly_chart')) >= 10
        assert any(t.label == 'SHAP summary' for t in at.tabs)

def test_form_prediction_history():
    at=AppTest.from_file(str(ROOT/'frontend/app.py'),default_timeout=30).run()
    at.sidebar.radio[0].set_value('Patient Prediction').run()
    assert len(at.number_input)==20
    at.number_input(key='value_Age').set_value(54.0)
    at.number_input(key='value_HR').set_value(96.0)
    at.checkbox[0].check()
    [b for b in at.button if b.label=='Predict Sepsis Risk'][0].click().run()
    assert not at.exception
    assert any(m.label=='Sepsis probability' for m in at.metric)
    assert len(at.warning)>0
    at.sidebar.radio[0].set_value('Prediction History').run()
    assert len(at.dataframe)==1
    [b for b in at.button if b.label=='Clear History'][0].click().run()
    assert len(at.dataframe)==0
