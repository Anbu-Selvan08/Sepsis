import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
import argparse
import numpy as np
import pandas as pd
from config import FEATURES, TARGET, ROOT

def load_dataset(path):
    df = pd.read_csv(path, dtype={'Patient_ID': 'string'})
    required = FEATURES + [TARGET, 'Patient_ID']
    absent = sorted(set(required) - set(df.columns))
    if absent:
        raise ValueError(f'Missing dataset columns: {absent}')
    return df[required].copy()

def clean_dataset(df):
    missing_target = int(df[TARGET].isna().sum())
    df = df.dropna(subset=[TARGET]).copy()
    if df.empty or not df[TARGET].isin([0, 1]).all():
        raise ValueError('SepsisLabel must contain only 0 and 1, with at least one labeled row')
    if df.Patient_ID.isna().any() or df.Patient_ID.str.strip().eq('').any():
        raise ValueError('Patient_ID is required for every labeled record')
    for feature in FEATURES:
        df[feature] = pd.to_numeric(df[feature], errors='raise')
        if np.isinf(df[feature].dropna()).any():
            raise ValueError(f'Infinite values in {feature}')
    duplicates = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)
    df[TARGET] = df[TARGET].astype(int)
    return df, {'removed_missing_target': missing_target, 'duplicate_rows': duplicates}

def analyze(df):
    missing = df[FEATURES].isna().sum()
    hist = {}
    for f in FEATURES:
        values = df[f].dropna()
        counts, edges = np.histogram(values, bins=30) if len(values) else ([], [])
        hist[f] = {'counts': list(map(int, counts)), 'edges': list(map(float, edges))}
    result = {'total_records':len(df), 'unique_patients':int(df.Patient_ID.nunique()), 'features':len(FEATURES),
              'target':TARGET, 'missing_values':int(missing.sum()), 'duplicate_rows':int(df.duplicated().sum()),
              'target_distribution':{str(k):int(v) for k,v in df[TARGET].value_counts(dropna=False).items()},
              'missing':[{'feature':f,'count':int(missing[f]),'percent':float(missing[f]/len(df)*100)} for f in FEATURES],
              'feature_statistics':json.loads(df[FEATURES].describe().to_json()), 'histograms':hist,
              'patient_record_statistics':df.groupby('Patient_ID').size().describe().to_dict()}
    return result

if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',default=str(ROOT/'data/sepsis_selected.csv'));p.add_argument('--output',default=str(ROOT/'models/dataset_analysis.json'));a=p.parse_args()
    result=analyze(load_dataset(a.data));Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    Path(a.output).write_text(json.dumps(result,indent=2,allow_nan=False));print(json.dumps({k:result[k] for k in ['total_records','unique_patients','duplicate_rows']},indent=2))
