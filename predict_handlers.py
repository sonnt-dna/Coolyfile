
# ---------------------- Import Library -------------------------

import joblib
import numpy as np
from pydantic import BaseModel
from typing import List
import pandas as pd
class PredictionRequest(BaseModel):
    GR: float
    LLD: float
    DEPTH: float
    NPHI: float
    RHOB: float

class PredictionResponse(BaseModel):
    predictions: List[float]
# ------------------------ Processing --------------------------------

def predict(full_df):
    df_ = pd.DataFrame(full_df)
    # Define the feature names as a list of strings
    feature_names = ["GR", "LLS", "LLD", "DEPTH", "NPHI", "RHOB"]
    # Filter the DataFrame to include only the feature columns
    df = df_[feature_names]
    print("Danh sách cột:", df.columns)
    # Load the model from the .joblib file
    model = joblib.load('LGBMRegressor.joblib')
 
    # Make predictions
    predictions = model.predict(df).tolist()
    return PredictionResponse(predictions=predictions)