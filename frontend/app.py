import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import os, json
from datetime import datetime, timezone
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from config import FEATURES, DISCLAIMER, UNITS
from frontend.research_views import render_research

st.set_page_config(page_title='SepsisScope · Academic Risk Assessment',page_icon='🩺',layout='wide')
API=os.getenv('SEPSIS_API_URL','http://127.0.0.1:8000').rstrip('/')
st.markdown('''<style>
.stApp {background: #f4f7fb;} [data-testid="stSidebar"] {background:#10283f;}
[data-testid="stSidebar"] * {color:#f4f7fb;} h1,h2,h3 {color:#16334b;}
[data-testid="stMetric"] {background:white;border:1px solid #dce6ef;border-radius:8px;padding:16px;}
.stButton>button[kind="primary"] {background:#087f8c;border:0;}
.block-container {padding-top:2rem;} .eyebrow {color:#087f8c;letter-spacing:2px;font-size:12px;font-weight:700;}
</style>''',unsafe_allow_html=True)
st.sidebar.title('SepsisScope')
st.sidebar.caption('CLINICAL DATA · RESEARCH PROTOTYPE')
page=st.sidebar.radio('Workspace',['Dashboard','Patient Prediction','Dataset Analysis','Model Analytics','Prediction History','About'])
st.sidebar.divider();st.sidebar.caption('Random Forest · 200 trees\n\nMedian imputation + training-only SMOTE')
st.sidebar.warning('Academic use only. Not a diagnostic system.')
st.markdown('<div class="eyebrow">ACADEMIC DECISION-SUPPORT PROTOTYPE</div>',unsafe_allow_html=True)
st.caption(DISCLAIMER)
if 'history' not in st.session_state:st.session_state.history=[]

class APIError(Exception):pass

def api(path,method='GET',**kwargs):
    try:
        response=requests.request(method,API+path,timeout=60,**kwargs)
        if not response.ok:
            try:detail=response.json().get('detail','Request failed')
            except ValueError:detail='Request failed'
            raise APIError(f'HTTP {response.status_code}: {detail}')
        return response.json()
    except requests.RequestException as exc:raise APIError('Backend unavailable. Start FastAPI on the configured address.') from exc

def metric_row(items, percent=False):
    for col,(label,value) in zip(st.columns(len(items)),items):
        col.metric(label,f'{value:.1%}' if percent else f'{value:,}' if isinstance(value,int) else str(value))

def result_panel(result):
    st.subheader('Sepsis Risk Assessment')
    if result['prediction'] == 'Sepsis':
        st.error(f"🚨 Sepsis Assessment: POSITIVE ({result['risk_level']} Risk · Estimated Probability {result['probability']:.1%})")
    else:
        st.success(f"✅ Sepsis Assessment: NEGATIVE ({result['risk_level']} Risk · Estimated Probability {result['probability']:.1%})")
    a,b,c=st.columns(3);a.metric('Model prediction',result['prediction']);b.metric('Sepsis probability',f"{result['probability']:.1%}");c.metric('Presentation risk level',result['risk_level'])
    st.caption('A model class is not a diagnosis. A No Sepsis output does not rule out sepsis. Probability is an uncalibrated model score.')
    gauge=go.Figure(go.Indicator(mode='gauge+number',value=result['probability']*100,number={'suffix':'%'},gauge={'axis':{'range':[0,100]},'bar':{'color':'#087f8c'},'steps':[{'range':[0,result['thresholds']['moderate']*100],'color':'#d6ece9'},{'range':[result['thresholds']['moderate']*100,result['thresholds']['high']*100],'color':'#fff0cf'},{'range':[result['thresholds']['high']*100,100],'color':'#f8dada'}]}))
    gauge.update_layout(height=240,margin=dict(t=25,b=15,l=30,r=30));st.plotly_chart(gauge,width='stretch')
    st.caption(f"Risk bands: Low < {result['thresholds']['moderate']:.0%}, Moderate < {result['thresholds']['high']:.0%}, otherwise High. Decision threshold: {result['thresholds'].get('decision', result['thresholds']['moderate']):.0%}.")
    left,right=st.columns(2)
    with left:st.info('Respiratory Clinical Indicators: '+result['lung_status'])
    with right:st.info('Kidney-related Clinical Indicators: '+result['kidney_status'])
    st.caption('These panels describe measurement availability only. They do not assess organ function or identify a source of sepsis.')
    st.subheader('Data Quality Warnings');st.write(result['data_quality']);st.progress(result['observed_features']/21,text=f"{result['observed_features']} of 21 features observed")
    for warning in result['warnings']:st.warning(warning)

try:
    if page=='Dashboard':
        st.title('Real-Time Sepsis Risk Assessment')
        st.write('Explore structured clinical data, assess model-estimated risk, and inspect the evidence behind each result.')
        info=api('/model-info');data=api('/analytics/dataset')
        performance=api('/analytics/model')
        if performance['recall']<0.5:st.error(f"Model limitation: held-out sepsis recall is {performance['recall']:.1%}. This model misses many positive observations and is unsuitable for clinical use.")
        metric_row([('Selected records',data['total_records']),('Selected patients',data['unique_patients']),('Clinical features',21)])
        st.caption('Report reference dataset: 546,123 records · 14,057 patients. This application uses the smaller uploaded-data subset shown above.')
        a,b=st.columns([1.3,1])
        with a:
            st.subheader('One transparent workflow')
            st.write('1. Enter observations or import a lab report.\n2. Review measurements and missing information.\n3. Request a model estimate from FastAPI.\n4. Inspect probability, context and warnings.')
            st.info('Model: Random Forest + SMOTE · 80/20 split by patient · no patient overlap')
        with b:
            st.subheader('Selected dataset labels')
            counts=data['target_distribution'];fig=px.pie(names=['No Sepsis','Sepsis'],values=[counts.get('0.0',counts.get('0',0)),counts.get('1.0',counts.get('1',0))],hole=.65,color_discrete_sequence=['#087f8c','#e39b40']);fig.update_layout(height=280,margin=dict(t=5,b=5));st.plotly_chart(fig,width='stretch')
        st.warning('Missing does not mean normal. This system uses training medians for missing model inputs and reports their absence separately.')
    elif page=='Patient Prediction':
        st.title('Patient Prediction');st.write('Enter a single observation time point. Leave unknown values blank.')
        st.subheader('🧪 Quick Test Cases')
        st.caption('Pre-populate clinical measurements to verify both Sepsis Positive and Sepsis Negative assessments.')
        c_pos,c_mod,c_neg,c_clr=st.columns(4)
        if c_pos.button('🔴 Sepsis-Positive (High Risk)',use_container_width=True):
            sample_pos=json.loads((Path(__file__).resolve().parents[1]/'examples/sepsis_positive_sample.json').read_text())
            for f in FEATURES:st.session_state['value_'+f]=sample_pos[f]
            st.session_state['patient_name']='Positive Sepsis ICU Case'
            st.session_state['reviewed_case']=True
            st.session_state.pop('result',None)
            st.rerun()
        if c_mod.button('🟡 Moderate Risk Case',use_container_width=True):
            sample_mod={'Age':68.0,'Gender':1,'Hour':48.0,'HR':128.0,'O2Sat':90.0,'Temp':39.2,'SBP':82.0,'MAP':54.0,'DBP':44.0,'Resp':30.0,'FiO2':0.50,'pH':7.25,'PaCO2':30.0,'SaO2':91.0,'BUN':48.0,'Creatinine':3.2,'WBC':22.0,'Lactate':5.4,'Platelets':55.0,'Hgb':9.2,'Hct':27.5}
            for f in FEATURES:st.session_state['value_'+f]=sample_mod[f]
            st.session_state['patient_name']='Moderate Sepsis Risk Case'
            st.session_state['reviewed_case']=True
            st.session_state.pop('result',None)
            st.rerun()
        if c_neg.button('🟢 Sepsis-Negative (Normal)',use_container_width=True):
            sample_neg=json.loads((Path(__file__).resolve().parents[1]/'examples/sepsis_negative_sample.json').read_text())
            for f in FEATURES:st.session_state['value_'+f]=sample_neg[f]
            st.session_state['patient_name']='Negative Healthy Normal Patient'
            st.session_state['reviewed_case']=True
            st.session_state.pop('result',None)
            st.rerun()
        if c_clr.button('🧹 Clear Inputs',use_container_width=True):
            for f in FEATURES:st.session_state['value_'+f]=None
            st.session_state['patient_name']=''
            st.session_state['reviewed_case']=False
            st.session_state.pop('result',None)
            st.rerun()
        with st.expander('Import a lab report / download templates',expanded=False):
            st.caption('CSV or JSON: one patient observation using the feature names below. TXT and text-based PDF: exact labels and units, e.g. “Creatinine: 1.2 mg/dL”. Scanned PDFs/images are not supported. Maximum 2 MB.')
            d1,d2,d3=st.columns(3)
            d1.download_button('Download Sepsis-Positive CSV',(Path(__file__).resolve().parents[1]/'examples/sepsis_positive_sample.csv').read_bytes(),'sepsis_positive_sample.csv','text/csv')
            d2.download_button('Download Sepsis-Negative CSV',(Path(__file__).resolve().parents[1]/'examples/sepsis_negative_sample.csv').read_bytes(),'sepsis_negative_sample.csv','text/csv')
            d3.download_button('Download Blank Template CSV',pd.DataFrame([{f:None for f in FEATURES}]).to_csv(index=False),'patient_template.csv','text/csv')
            upload=st.file_uploader('Lab report',type=['csv','json','txt','pdf'])
            if st.button('Extract values for review',disabled=upload is None):
                extracted=api('/reports/parse','POST',files={'file':(upload.name,upload.getvalue(),upload.type)})
                for f in FEATURES:st.session_state['value_'+f]=extracted['features'][f]
                st.session_state.pop('result',None)
                for warning in extracted['warnings']:st.warning(warning)
                st.success('Values imported below. Review units and the observation time before assessment.')
        with st.form('patient_form'):
            name=st.text_input('Patient display name (optional; use an alias)',value=st.session_state.get('patient_name',''),max_chars=100)
            sections={'Demographics':['Age','Gender','Hour'],'Vital Signs':['HR','O2Sat','Temp','SBP','MAP','DBP','Resp'],'Respiratory / Blood Gas':['FiO2','pH','PaCO2','SaO2'],'Laboratory':['BUN','Creatinine','WBC','Lactate','Platelets','Hgb','Hct']}
            values={}
            for heading,features in sections.items():
                st.subheader(heading);columns=st.columns(3)
                for i,f in enumerate(features):
                    with columns[i%3]:
                        if f=='Gender':
                            values[f]=st.selectbox('Gender (dataset encoding)',options=[None,0,1],format_func=lambda x:'Missing' if x is None else 'Female (0)' if x==0 else 'Male (1)',key='value_'+f)
                        else:values[f]=st.number_input(f'{f} · {UNITS[f]}',value=None,min_value=0.0,format='%.3f',key='value_'+f,placeholder='Missing / not measured')
            reviewed=st.checkbox('I have reviewed the values, units and observation time. Missing values are unknown, not normal.',value=st.session_state.get('reviewed_case',False))
            submitted=st.form_submit_button('Predict Sepsis Risk',type='primary')
        if submitted:
            st.session_state.pop('result',None)
            if not reviewed:st.error('Review the measurements and tick the confirmation box first.')
            else:
                with st.spinner('Assessing with the saved Random Forest…'):result=api('/predict','POST',json={'features':values})
                st.session_state.result=result
                st.session_state.history.append({'Timestamp':datetime.now(timezone.utc).isoformat(),'Patient Display Name':name or 'Anonymous','Prediction':result['prediction'],'Probability':result['probability'],'Risk Level':result['risk_level'],'Respiratory Status':result['lung_status'],'Kidney Status':result['kidney_status'],'Warnings':'; '.join(result['warnings'])})
                st.session_state.history=st.session_state.history[-200:]
        if 'result' in st.session_state:
            st.caption('Last submitted assessment; edits to the form require a new prediction.')
            result_panel(st.session_state.result)
    elif page=='Dataset Analysis':
        st.title('Dataset Analysis');data=api('/analytics/dataset');model=api('/analytics/model')
        metric_row([('Total records',data['total_records']),('Unique patients',data['unique_patients']),('Features',data['features']),('Duplicate rows',data['duplicate_rows'])])
        st.caption(f"Target: {data['target']} · Missing feature cells: {data['missing_values']:,}. Duplicates are checked using patient ID, target and the 21 clinical features, excluding export-index columns.")
        tabs=st.tabs(['Class distribution','Missing data','Feature distributions','Patient split'])
        with tabs[0]:
            counts=data['target_distribution'];st.plotly_chart(px.bar(x=['No Sepsis','Sepsis'],y=[counts.get('0.0',counts.get('0',0)),counts.get('1.0',counts.get('1',0))],labels={'x':'Label','y':'Records'},color_discrete_sequence=['#087f8c']),width='stretch')
        with tabs[1]:
            missing=pd.DataFrame(data['missing']);st.plotly_chart(px.bar(missing,x='feature',y='percent',labels={'percent':'Missing (%)'},color_discrete_sequence=['#e39b40']),width='stretch');st.dataframe(missing,hide_index=True,width='stretch')
        with tabs[2]:
            feature=st.selectbox('Clinical feature',FEATURES);hist=data['histograms'][feature];edges=hist['edges']
            if edges:st.plotly_chart(px.bar(x=[(a+b)/2 for a,b in zip(edges[:-1],edges[1:])],y=hist['counts'],labels={'x':feature+' · '+UNITS[feature],'y':'Observed records'},color_discrete_sequence=['#087f8c']),width='stretch')
            st.caption('Histograms exclude missing values. Only aggregate counts are sent to the frontend.')
        with tabs[3]:st.dataframe(pd.DataFrame(model['split'].items(),columns=['Validation','Result']),hide_index=True)
    elif page=='Model Analytics':
        st.title('Model Analytics');data=api('/analytics/model')
        metric_row([(label,data[key]) for label,key in [('Accuracy','accuracy'),('Precision','precision'),('Recall','recall'),('F1 Score','f1'),('ROC-AUC','roc_auc')]],percent=True)
        st.caption(f"Held-out prevalence: {data['test_prevalence']:.2%} · Always predict No Sepsis baseline accuracy: {data['majority_baseline_accuracy']:.2%} · Average precision: {data['average_precision']:.3f}")
        st.warning('Accuracy alone can look strong when sepsis is rare. Recall measures detected positive observations; F1 balances precision and recall. These results do not establish clinical safety.')
        render_research(data)
        a,b=st.columns(2)
        with a:
            st.subheader('Confusion matrix');fig=px.imshow(data['confusion_matrix'],x=['No Sepsis','Sepsis'],y=['No Sepsis','Sepsis'],text_auto=True,color_continuous_scale='Teal',labels={'x':'Predicted','y':'Actual','color':'Records'});st.plotly_chart(fig,width='stretch')
            st.caption('Top-left TN · top-right FP · bottom-left FN · bottom-right TP')
        with b:
            st.subheader('Training class balance');rows=[{'Stage':stage,'Class':'Sepsis' if k=='1' else 'No Sepsis','Records':v} for stage,key in [('Before SMOTE','before_smote'),('After SMOTE','after_smote')] for k,v in data[key].items()];st.plotly_chart(px.bar(pd.DataFrame(rows),x='Stage',y='Records',color='Class',barmode='group',color_discrete_sequence=['#087f8c','#e39b40']),width='stretch')
            st.caption('SMOTE creates synthetic minority-class samples in feature space. These are not real patients. Test observations and live inputs are never resampled.')
        importance=pd.DataFrame(data['feature_importance'].items(),columns=['Feature','Importance']).sort_values('Importance')
        st.subheader('Random Forest feature importance');st.plotly_chart(px.bar(importance,x='Importance',y='Feature',orientation='h',color_discrete_sequence=['#087f8c']),width='stretch');st.caption('Impurity-based importance describes model behavior; it does not establish clinical causation.')
        with st.expander('Model configuration and split validation'):st.json(data['model_parameters']);st.json(data['split'])
        with st.expander('ROC curve'):st.plotly_chart(px.line(x=data['roc_curve']['fpr'],y=data['roc_curve']['tpr'],labels={'x':'False positive rate','y':'True positive rate'}),width='stretch')
    elif page=='Prediction History':
        st.title('Prediction History');st.caption('Session-only · latest 200 assessments · use aliases. The API does not persist patient data. Downloaded files may contain the aliases you entered.')
        if st.button('Clear History'):st.session_state.history=[];st.session_state.pop('result',None)
        history=pd.DataFrame(st.session_state.history)
        if history.empty:st.info('No predictions in this session yet.')
        else:
            st.dataframe(history,hide_index=True,width='stretch')
            safe=history.copy()
            for col in safe.select_dtypes('object'):
                safe[col]=safe[col].map(lambda x:"'"+x if isinstance(x,str) and x.startswith(('=','+','-','@','\t','\r')) else x)
            st.download_button('Download Prediction History',safe.to_csv(index=False),'prediction_history.csv','text/csv')
    else:
        st.title('About this academic prototype');st.info(DISCLAIMER)
        st.write('The application follows the supplied project specification: patient-level split → training-only median imputation → training-only SMOTE → 200-tree Random Forest Classifier → evaluation on held-out patients.')
        st.write('The CSV labels represent record-level classification. This model does not establish a future prediction horizon, diagnose sepsis, identify the infection source, or assess organ function.')
        st.write('The supplied project-report document was not attached. The implementation follows the detailed written requirements. The final request’s “regression” wording is implemented as classification to match the binary target and probability requirements.')
        st.warning('Limitations: single random patient holdout; uncalibrated probabilities after resampling; missingness and dataset shift; ordinary SMOTE can interpolate the binary Gender feature. The requested SMOTE workflow is retained, and no clinical-validity claim is made.')
        st.dataframe(pd.DataFrame({'Feature':FEATURES,'Expected units':[UNITS[f] for f in FEATURES]}),hide_index=True,width='stretch')
except APIError as exc:
    st.error(str(exc));st.info('From the project folder, start: python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000')
