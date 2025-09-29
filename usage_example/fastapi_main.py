from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Union
import joblib
import pandas as pd

model = joblib.load("exoplanet_model.pkl")
features = [
    'koi_period', 'koi_duration', 'koi_depth', 'koi_prad',
    'koi_teq', 'koi_insol', 'koi_model_snr', 'koi_score',
    'koi_fpflag_nt', 'koi_fpflag_ss', 'koi_fpflag_co', 'koi_fpflag_ec',
    'koi_steff', 'koi_slogg', 'koi_srad'
]

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
async def predict(data: Union[dict, List[dict]]):
    """
    Accepts either a single JSON object or a list of objects
    """
    if isinstance(data, dict):
        df = pd.DataFrame([data], columns=features)
    else:
        df = pd.DataFrame(data, columns=features)

    df = df.fillna(-1)

    preds = model.predict(df)
    return {"predictions": preds.tolist()}
