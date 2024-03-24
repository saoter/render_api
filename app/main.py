from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import joblib
import numpy as np
import sqlite3
import pandas as pd

# We initiate the API app
app = FastAPI()

# We locate database
DATABASE_URL = "db/db_penguins.db"


######################################################################
######################################################################
# We create end point function to access penguin data
# We add 3 optional conditions
def fetch_penguins(island_id: Optional[int] = None, status_id: Optional[int] = None, species: Optional[str] = None) -> list:
    conn = sqlite3.connect(DATABASE_URL)
    conditions = []
    
    if island_id is not None:
        conditions.append(f"island_id = {island_id}")
    if status_id is not None:
        conditions.append(f"status_id = {status_id}")
    if species is not None:
        conditions.append(f"species = '{species}'")

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
def fetch_penguins(island_id: Optional[int] = None, status_id: Optional[int] = None, species: Optional[str] = None) -> list:
    try:
        penguins_data = fetch_penguins(island_id, status_id, species)
        return {"penguins": penguins_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

######################################################################
######################################################################
# We create end point function to access model data
# We add 1 optional conditions

def fetch_model(model_id: Optional[int] = None) -> list[dict]:
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
def get_model(model_id: Optional[int] = query(None, title="Model ID", description="The ID of the model to fetch")):
    try:
        model_data = fetch_model(model_id)
        return {"model": model_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
######################################################################
######################################################################
# We create end point function to access status data
# We do not add any conditions

def fetch_status() -> list[dict]:
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
# We want to make POST call for making predictions

# What we expect to receive from the call    
class PredictionRequest(BaseModel):
    model_id: int
    bill_length_mm: float
    bill_depth_mm: float
    flipper_length_mm: float
    body_mass_g: float

# Function to make prediction. First load teh model 
def load_model(model_id: int):
    model_path = f"models/model_v{model_id}.joblib"
    try:
        model = joblib.load(model_path)
        return model
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Model with ID {model_id} not found")

# Making the end point for prediction
@app.post("/predict/")
def predict(request: PredictionRequest):
    model = load_model(request.model_id)
    features = pd.DataFrame([[
        request.bill_length_mm,
        request.bill_depth_mm,
        request.flipper_length_mm,
        request.body_mass_g
    ]], columns=["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"])

    prediction = model.predict(features)
    return {"prediction": prediction.tolist()}
