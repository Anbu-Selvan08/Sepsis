"""Select approximately 50k records without truncating any selected patient's history."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse,json
import numpy as np
from config import ROOT
from training.dataset_analysis import load_dataset

def select(source,output,target=50000):
    df=load_dataset(source)
    if df.Patient_ID.isna().any():raise ValueError('Cannot select patient groups with missing IDs')
    groups=df.groupby('Patient_ID',sort=True).size()
    ids=np.random.default_rng(42).permutation(groups.index)
    cumulative=groups.loc[ids].cumsum()
    n=int(np.argmin(abs(cumulative.to_numpy()-target)))+1
    selected=df[df.Patient_ID.isin(ids[:n])]
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True);selected.to_csv(output,index=False)
    manifest={'source_records':len(df),'source_patients':int(df.Patient_ID.nunique()),'selected_records':len(selected),'selected_patients':int(selected.Patient_ID.nunique()),'selection':'Seed 42 random patient order, nearest whole-patient prefix to requested record count','target_counts':{str(k):int(v) for k,v in selected.SepsisLabel.value_counts().items()}}
    (output.parent/'selection_manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('--output',default=str(ROOT/'data/sepsis_selected.csv'));p.add_argument('--records',type=int,default=50000);a=p.parse_args()
    if a.records<1:p.error('records must be positive')
    select(a.source,a.output,a.records)
