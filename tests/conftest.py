import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json
import pytest
from config import ROOT
from backend.model_service import ModelService

@pytest.fixture(scope='session')
def values():return json.loads((ROOT/'examples/example_lab_report.json').read_text())

@pytest.fixture(scope='session')
def model():return ModelService(ROOT/'models')
