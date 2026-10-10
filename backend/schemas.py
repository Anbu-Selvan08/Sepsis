from typing import Annotated
from pydantic import BaseModel, ConfigDict, Field, create_model, model_validator
from config import FEATURES

class ClinicalBase(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True, allow_inf_nan=False)
    @model_validator(mode='after')
    def values_present(self):
        if all(getattr(self, f) is None for f in FEATURES):
            raise ValueError('At least one observed clinical measurement is required; all-missing input cannot be assessed')
        return self

# Every feature key is required; null explicitly means missing, never normal.
LIMITS={'Age':(0,130),'Gender':(0,1),'Hour':(0,None),'O2Sat':(0,100),'SaO2':(0,100),'FiO2':(0,1),'pH':(0,14),'Hct':(0,100),'Temp':(0,60)}
fields={}
for feature in FEATURES:
    low,high=LIMITS.get(feature,(0,None))
    numeric=Annotated[int,Field(ge=0,le=1)] if feature=='Gender' else Annotated[float,Field(ge=low,le=high)]
    fields[feature]=(numeric | None, ...)
ClinicalFeatures=create_model('ClinicalFeatures',__base__=ClinicalBase,**fields)

class PredictionRequest(BaseModel):
    model_config=ConfigDict(extra='forbid',strict=True)
    features: ClinicalFeatures
    patient_name: str | None = Field(default=None,max_length=100,description='Optional display only; not used by the model or retained by the API')

class PredictionResponse(BaseModel):
    prediction: str
    predicted_class: int
    probability: float
    risk_level: str
    lung_status: str
    kidney_status: str
    data_quality: str
    observed_features: int
    missing_features: list[str]
    warnings: list[str]
    thresholds: dict[str,float]
    model_version: str
