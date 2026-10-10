"""Rebuild research plots for trusted saved artifacts without retraining."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import hashlib
import json
from backend.model_service import ModelService
from config import ROOT, FEATURES, TARGET
from training.dataset_analysis import load_dataset, clean_dataset
from training.train_model import patient_split
from training.evaluate_model import evaluate
from training.research_analysis import shap_summary


def refresh(data, directory, with_shap=True):
    service = ModelService(Path(directory))
    if hashlib.sha256(Path(data).read_bytes()).hexdigest() != service.metadata['dataset_sha256']:
        raise ValueError('Dataset differs from training data; refusing to mislabel a test cohort.')
    clean, _ = clean_dataset(load_dataset(data))
    _, test = patient_split(clean, service.metadata['test_size'])
    X = service.imputer.transform(test[FEATURES])
    print('Evaluating saved model on original held-out patients...', flush=True)
    metrics = {**service.metrics, **evaluate(service.model, X, test[TARGET])}
    if with_shap:
        print('Computing exact TreeSHAP on 100 held-out observations...', flush=True)
        metrics['shap_summary'] = shap_summary(service.model, X, FEATURES)
    output = Path(directory) / 'metrics.json'
    temporary = output.with_suffix('.tmp')
    temporary.write_text(json.dumps(metrics, indent=2, allow_nan=False))
    temporary.replace(output)
    print('Research analytics saved. Restart the backend to load them.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default=str(ROOT / 'data/sepsis_selected.csv'))
    parser.add_argument('--models', default=str(ROOT / 'models'))
    parser.add_argument('--skip-shap', action='store_true')
    args = parser.parse_args()
    refresh(args.data, args.models, not args.skip_shap)
