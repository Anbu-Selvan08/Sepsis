import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse, hashlib, json, logging, platform
from datetime import datetime, timezone
import joblib
import numpy as np
import sklearn, imblearn
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from config import FEATURES, TARGET, ROOT
from training.dataset_analysis import load_dataset, clean_dataset, analyze
from training.evaluate_model import evaluate
from training.research_analysis import shap_summary
log=logging.getLogger(__name__)

def patient_split(df, test_size=.2):
    patients=df.groupby('Patient_ID')[TARGET].max()
    if patients.value_counts().min()<2 or patients.nunique()!=2:
        raise ValueError('Need at least two patients in each patient-level target class')
    train_ids,test_ids=train_test_split(patients.index,test_size=test_size,random_state=42,stratify=patients)
    train=df[df.Patient_ID.isin(train_ids)].copy();test=df[df.Patient_ID.isin(test_ids)].copy()
    overlap=set(train.Patient_ID)&set(test.Patient_ID)
    if overlap: raise ValueError('Patient leakage detected')
    if train[TARGET].nunique()!=2 or test[TARGET].nunique()!=2:
        raise ValueError('Both classes must be present in training and test partitions')
    return train,test

def distribution(y): return {str(k):int(v) for k,v in zip(*np.unique(y,return_counts=True))}

def train(data, output, test_size=.2):
    raw=load_dataset(data);df,checks=clean_dataset(raw)
    training,test=patient_split(df,test_size)
    checks.update(training_patients=int(training.Patient_ID.nunique()),testing_patients=int(test.Patient_ID.nunique()),
                  training_records=len(training),testing_records=len(test),patient_overlap=0,duplicates_after_cleaning=int(df.duplicated().sum()))
    log.info('Split validation: %s',json.dumps(checks))
    X_train=training[FEATURES];X_test=test[FEATURES];y_train=training[TARGET];y_test=test[TARGET]
    empty=X_train.columns[X_train.isna().all()].tolist()
    if empty: raise ValueError(f'Cannot fit median for entirely missing training features: {empty}')
    log.info('Fitting median imputer on training records only')
    imputer=SimpleImputer(strategy='median');train_imputed=imputer.fit_transform(X_train);test_imputed=imputer.transform(X_test)
    if y_train.value_counts().min()<6: raise ValueError('Default SMOTE needs at least 6 training records in each class')
    log.info('Applying SMOTE only to training records')
    balanced,labels=SMOTE(random_state=42).fit_resample(train_imputed,y_train)
    model=RandomForestClassifier(n_estimators=200,random_state=42,n_jobs=-1,max_features='sqrt',max_depth=12,min_samples_leaf=5)
    log.info('Training 200 trees on %s resampled records',len(labels));model.fit(balanced,labels)
    log.info('Evaluating on untouched held-out patients');metrics=evaluate(model,test_imputed,y_test)
    log.info('Computing SHAP summary on 100 held-out observations')
    metrics['shap_summary']=shap_summary(model,test_imputed,FEATURES)
    metrics.update(split=checks,before_smote=distribution(y_train),after_smote=distribution(labels),
                   feature_importance=dict(zip(FEATURES,model.feature_importances_.tolist())),model_parameters=model.get_params())
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    joblib.dump(model,output/'random_forest.joblib',compress=3);joblib.dump(imputer,output/'imputer.joblib',compress=3)
    metadata={'features':FEATURES,'target':TARGET,'created_at':datetime.now(timezone.utc).isoformat(),
              'dataset_sha256':hashlib.sha256(Path(data).read_bytes()).hexdigest(),'records':len(df),'patients':int(df.Patient_ID.nunique()),
              'test_size':test_size,'random_state':42,'units_source':'PhysioNet-style schema; verify source data dictionary before other datasets',
              'versions':{'python':platform.python_version(),'scikit-learn':sklearn.__version__,'imbalanced-learn':imblearn.__version__},
              'artifact_sha256':{f:hashlib.sha256((output/f).read_bytes()).hexdigest() for f in ['random_forest.joblib','imputer.joblib']}}
    for name,obj in [('feature_metadata.json',metadata),('metrics.json',metrics),('dataset_analysis.json',analyze(raw))]:
        (output/name).write_text(json.dumps(obj,indent=2,allow_nan=False))
    log.info('Metrics: %s',json.dumps({k:metrics[k] for k in ['accuracy','precision','recall','f1','roc_auc']}))
    return metrics

if __name__=='__main__':
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    p=argparse.ArgumentParser();p.add_argument('--data',default=str(ROOT/'data/sepsis_selected.csv'));p.add_argument('--output',default=str(ROOT/'models'));p.add_argument('--test-size',type=float,default=.2);a=p.parse_args()
    if not 0<a.test_size<1:p.error('test-size must be between 0 and 1')
    train(a.data,a.output,a.test_size)
