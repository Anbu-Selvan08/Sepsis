"""Conservative extraction only; patient must review values and units before assessment."""
import io, json, re
import pandas as pd
from pypdf import PdfReader
from config import FEATURES, UNITS
from backend.schemas import ClinicalFeatures

# Text reports accept exact feature names with explicit units. No reference-range inference.
TEXT_UNITS={'Age':'years','Gender':'code','Hour':'hours','HR':'beats/min','O2Sat':'%','Temp':'C','SBP':'mmHg','MAP':'mmHg','DBP':'mmHg','Resp':'breaths/min','FiO2':'fraction','pH':'pH','PaCO2':'mmHg','SaO2':'%','BUN':'mg/dL','Creatinine':'mg/dL','WBC':'10^3/uL','Lactate':'mmol/L','Platelets':'10^3/uL','Hgb':'g/dL','Hct':'%'}

def parse_report(data, filename):
    extension=filename.lower().rsplit('.',1)[-1]
    values={f:None for f in FEATURES};warnings=[]
    if extension in ['csv','json']:
        if extension=='csv':
            frame=pd.read_csv(io.BytesIO(data))
            if len(frame)!=1:raise ValueError('Patient report CSV must contain exactly one row')
            if len(frame.columns)!=len(set(frame.columns)) or any('.' in x for x in frame.columns):
                raise ValueError('Duplicate or unexpected CSV headers')
            parsed=frame.iloc[0].where(frame.iloc[0].notna(),None).to_dict()
            parsed={k:None if pd.isna(v) else v for k,v in parsed.items()}
        else:
            parsed=json.loads(data)
            if not isinstance(parsed,dict):raise ValueError('JSON report must be one feature object')
        unknown=set(parsed)-set(FEATURES)
        if unknown:raise ValueError('Report contains unsupported feature keys; use the supplied template')
        for f,v in parsed.items():
            if v is not None:
                if isinstance(v,bool) or not isinstance(v,(int,float)):raise ValueError('Report values must be numeric or blank/null')
                if f=='Gender':
                    if v not in [0,1]:raise ValueError('Gender must be 0 or 1')
                    v=int(v)
            values[f]=v
        warnings.append('CSV/JSON units are not detected. Confirm every value uses the displayed units before prediction.')
    elif extension in ['txt','pdf']:
        if extension=='pdf':
            reader=PdfReader(io.BytesIO(data))
            if reader.is_encrypted:raise ValueError('Encrypted PDFs are not supported')
            if len(reader.pages)>10:raise ValueError('Use a report with at most 10 pages')
            text='\n'.join(page.extract_text() or '' for page in reader.pages)
        else:text=data.decode('utf-8')
        if not text.strip():raise ValueError('No readable text. Scanned PDFs and images are not supported; enter values manually or use CSV.')
        for f in FEATURES:
            pattern=rf'^\s*{re.escape(f)}\s*[:=]\s*(-?\d+(?:\.\d+)?)\s+{re.escape(TEXT_UNITS[f])}\s*$'
            matches=re.findall(pattern,text,flags=re.MULTILINE|re.IGNORECASE)
            if len(matches)==1:
                value=float(matches[0]);values[f]=int(value) if f=='Gender' and value in [0,1] else value
            elif len(matches)>1:warnings.append(f'{f}: multiple results found; enter the appropriate time-point value manually.')
        warnings.append('Conservative extraction: only exact feature labels and recognized units were imported. Unmatched values remain missing. Review before prediction.')
    else:raise ValueError('Supported reports: CSV, JSON, TXT, and text-based PDF')
    ClinicalFeatures.model_validate(values)
    return {'features':values,'units':UNITS,'warnings':warnings,'review_required':True}
