from contextlib import asynccontextmanager
import logging
from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool
from backend.schemas import PredictionRequest, PredictionResponse
from backend.model_service import ModelService
from backend.report_parser import parse_report
from config import MODEL_DIR

@asynccontextmanager
async def lifespan(app):
    try:app.state.service=ModelService(MODEL_DIR)
    except Exception:
        app.state.service=None
        logging.getLogger(__name__).error('Model unavailable. Verify trusted artifacts or run offline training.')
    yield

app=FastAPI(title='Academic Sepsis Risk Assessment',version='1.0.0',lifespan=lifespan)

@app.exception_handler(RequestValidationError)
async def validation_handler(request,exc):
    # Avoid echoing names, measurements, or raw request data in validation errors.
    return JSONResponse(status_code=422,content={'detail':[{'loc':list(e['loc']),'message':e['msg'],'type':e['type']} for e in exc.errors()]})

def service(request):
    model=getattr(request.app.state,'service',None)
    if model is None:raise HTTPException(503,'Model unavailable. Ask the operator to train or restore valid model artifacts.')
    return model

@app.get('/health')
def health():return {'status':'healthy'}

@app.get('/ready')
def ready(request:Request):service(request);return {'status':'ready'}

@app.get('/model-info')
def model_info(request:Request):
    return {'model':'Random Forest','n_estimators':200,'features':21,'target':'SepsisLabel','ready':getattr(request.app.state,'service',None) is not None}

@app.post('/predict',response_model=PredictionResponse)
def predict(body:PredictionRequest,request:Request):
    model=service(request)
    try:return model.predict(body.features.model_dump())
    except Exception:raise HTTPException(503,'Prediction could not be generated. Check model artifacts and retry.')

@app.get('/analytics/dataset')
def dataset(request:Request):return service(request).analysis

@app.get('/analytics/model')
def model_analytics(request:Request):return service(request).metrics

@app.post('/reports/parse')
async def report(file:UploadFile=File(...)):
    data=await file.read(2*1024*1024+1)
    await file.close()
    if len(data)>2*1024*1024:raise HTTPException(413,'Report must be at most 2 MB')
    try:return await run_in_threadpool(parse_report,data,file.filename or '')
    except Exception:raise HTTPException(422,'Could not import this report. Use the one-row CSV/JSON template, or TXT/text PDF with exact feature names and documented units. Scanned PDFs, invalid values and ambiguous-only reports are not supported.')
