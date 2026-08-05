from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Annotated
from model.inference import load_scaler, load_model, inference
from pathlib import Path
from env_settings import Settings
from database.postgre_connection import connection, text
import pandas as pd
from contextlib import asynccontextmanager

settings = Settings()
engine = connection(settings)
data = pd.read_parquet(settings.data_path)

ml_context = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        ml_context['scaler'] = load_scaler(settings.scaler_path)
    except:
        pass

    try:
        ml_context['model'] = load_model(settings.model_path, data)
    except:
        pass
    yield

app = FastAPI(lifespan=lifespan)

class prediction_params(BaseModel):
    id: Annotated[int, Query(title="Dataset ID", ge=0, le=190_000)]
    T: Annotated[int, Query(title="MC Dropout Passes", ge=30, le=50)]


@app.get("/health")
def health():
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except SQLAlchemyError:
        raise HTTPException(
            status_code=503,
            detail="Postgres Unavailable"
        )

    try:
        model = ml_context['model']
    except:
        raise HTTPException(
            status_code=503,
            detail="Model Not Found"
        )

    try:
        scaler = ml_context['scaler']
    except:
        raise HTTPException(
            status_code=503,
            detail="Scaler Not Found"
        )

    return {"status": "Everything is Ready"}



@app.post("/predict")
def prediction(params: prediction_params):
    try:
        results_dict = inference(params.id, params.T, ml_context["model"], ml_context["scaler"], data, engine)
        return results_dict
    except:
        raise HTTPException(status_code=500, detail="Could not predict, Check Health")



