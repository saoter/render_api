from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import Optional, List
import joblib
import numpy as np
import sqlite3
import pandas as pd

app = FastAPI()

DATABASE_URL = "db/db_penguins.db"


######################################################################
######################################################################

def fetch_penguins(island_id: Optional[int] = None, status_id: Optional[int] = None, species: Optional[str] = None, model_id: Optional[int] = None) -> list:
    conn = sqlite3.connect(DATABASE_URL)
    conditions = []
    
    if island_id is not None:
        conditions.append(f"island_id = {island_id}")
    if status_id is not None:
        conditions.append(f"status_id = {status_id}")
    if species is not None:
        conditions.append(f"species = '{species}'")
    if model_id is not None:
        conditions.append(f"model_id = {model_id}")

    query_conditions = " AND ".join(conditions)
    query = "SELECT * FROM PENGUINS"
    if query_conditions:
        query += f" WHERE {query_conditions}"

    try:
        data = pd.read_sql_query(query, conn)
        return data.to_dict(orient="records")
    finally:
        conn.close()

@app.get("/penguins/")
def get_penguins(island_id: Optional[int] = None, status_id: Optional[int] = None, species: Optional[str] = None, model_id: Optional[int] = None):
    try:
        penguins_data = fetch_penguins(island_id, status_id, species, model_id)
        return {"penguins": penguins_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

######################################################################
######################################################################

def fetch_model(model_id: Optional[int] = None) -> List[dict]:
    conn = sqlite3.connect(DATABASE_URL)
    try:
        if model_id is not None:
            # Use parameter substitution to safely insert the model_id into the query
            query = "SELECT * FROM MODEL WHERE model_id = ?"
            data = pd.read_sql_query(query, conn, params=(model_id,))
        else:
            query = "SELECT * FROM MODEL"
            data = pd.read_sql_query(query, conn)
        return data.to_dict(orient="records")
    except Exception as e:
        print(e)
        raise
    finally:
        conn.close()



@app.get("/model/")
def get_model(model_id: Optional[int] = Query(None, title="Model ID", description="The ID of the model to fetch")):
    try:
        model_data = fetch_model(model_id)
        return {"model": model_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
######################################################################
######################################################################
    
def fetch_status() -> List[dict]:
    conn = sqlite3.connect(DATABASE_URL)
    try:
        query = "SELECT * FROM STATUS"
        data = pd.read_sql_query(query, conn)
        return data.to_dict(orient="records")
    finally:
        conn.close()

@app.get("/status/")
def get_status():
    try:
        status_data = fetch_status()  # Corrected function call
        return {"status": status_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

######################################################################
######################################################################
    
class PredictionRequest(BaseModel):
    prediction_model_id: int
    bill_length_mm: float
    bill_depth_mm: float
    flipper_length_mm: float
    body_mass_g: float

def load_model(model_id: int):
    # Corrected to use the function parameter `model_id`
    model_path = f"models/model_v{model_id-100}.joblib"
    try:
        model = joblib.load(model_path)
        return model
    except FileNotFoundError:
        # Corrected variable name in the error message
        raise HTTPException(status_code=404, detail=f"Model with ID {model_id} not found")

@app.post("/predict/")
def predict(request: PredictionRequest):
    # Corrected to use `request.prediction_model_id`
    model = load_model(request.prediction_model_id)
    features = pd.DataFrame([[
        request.bill_length_mm,
        request.bill_depth_mm,
        request.flipper_length_mm,
        request.body_mass_g
    ]], columns=["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"])

    prediction = model.predict(features)
    return {"prediction": prediction.tolist()}
