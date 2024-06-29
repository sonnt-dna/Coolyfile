
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Header, APIRouter, HTTPException, FastAPI, UploadFile, File, Request, Form, Query
from pydantic import BaseModel
import pandas as pd
import json as json
import csv
from io import StringIO
from predict_handlers import predict
import pandas as pd
from typing import Union, Literal

app = FastAPI()

# Check if the server is running or not
@app.get("/hello")
async def read_main():
    return {"message": "Hello World", "Note": "This is a test"}

# Endpoint to predict
@app.post("/predict", status_code=201)
async def upload_csv(asc_file: UploadFile = File(...), query_params: str = Form(...)):

    def rename_col(name_old_column: str, name_new_column: str, df: pd.DataFrame) -> pd.DataFrame:
        return df.rename(columns={name_old_column: name_new_column}) if name_old_column in df.columns else df

    def read_asc(file: Union[str, object]) -> pd.DataFrame:
        try:
            if isinstance(file, str):
                with open(file, 'r') as f:
                    lines = f.readlines()
            else:
                lines = file.read().decode('utf-8').splitlines()
        except Exception as e:
            raise ValueError(f"Error reading file: {e}")

        cols = [col.strip() for col in lines[1].split()]
        well_name = [x.strip() for x in lines[0].split(" ")]

        rows = []
        for line in lines[4:]:
            row = [x.strip() for x in line.split()]
            if len(row) == len(cols):
                rows.append(row)

        if not rows:
            raise ValueError("No data in file")

        df = pd.DataFrame(rows, columns=cols).apply(pd.to_numeric, errors="coerce")

        # Replace negative values by NaN
        df = df.where(df >= 0)

        # Use UPPER CASE column names
        df.columns = [col.upper() for col in df.columns]

        # Add well name
        df['WELLBORE'] = well_name[2]
        date = " ".join(well_name[-2:])
        df['FIELD'] = date.strip()

        ######################
        # rename vcl to vwcl #
        ######################
        df = rename_col("VCL", "VWCL", df)

        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    # Cố gắng chuyển đổi chuỗi ngày tháng thành timestamp
                    df[col] = pd.to_datetime(df[col])
                    df[col] = df[col].astype('int64') // 10**9  # Chuyển đổi datetime thành timestamp
                except ValueError:
                    # Nếu không thể chuyển đổi, loại bỏ cột
                    df = df.drop(columns=[col])
        return df

    df = read_asc(asc_file.file)
    predicted_result = predict(df)
    return predicted_result