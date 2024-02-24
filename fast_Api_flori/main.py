import uuid
from io import StringIO
import os
from typing import Annotated

import numpy as np
from starlette.responses import FileResponse

import src.code.training.trainer as trainer
import src.code.inference.predict as predict

import pandas as pd
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

app = FastAPI()

current_model = None

@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}


@app.post("/files/")
async def create_file(file: Annotated[bytes, File()]):
    return {"file_size": len(file)}


@app.post("/upload-training-data/")
async def create_upload_file(file: UploadFile, final_metric: str):
    dataframe = pd.read_csv(file.file, delimiter=';')
    current_model = trainer.train_model(dataframe, final_metric)
    return {"filename": file.filename}


@app.post("/upload-data-for-predictions/")
async def create_upload_file(file: UploadFile):
    dataframe = pd.read_csv(file.file)
    predictions = predict.load_model_and_predict(dataframe, current_model)

    # Generate a unique file name
    file_name = f"predictions_{uuid.uuid4().hex}.csv"

    # Save predictions to a CSV file
    predictions.to_csv(file_name, index=False)
    file_path = f"./{file_name}"  # Adjust the path based on where you save files
    return FileResponse(path=file_path, filename=file_name, media_type='text/csv')

    # Return the path or name of the saved file to the client
    # Note: Adjust based on whether you want to automatically start the download or just provide a link
    # return {"file_name": unique_filename}
    #
    # # Convert DataFrame to JSON
    # predictions_json = predictions.to_json(orient="index")
    #
    # # Return the JSON response
    # return JSONResponse(content={"predictions": predictions_json})


@app.get("/download-predictions/{file_name}")
async def download_predictions(file_name: str):
    file_path = f"./{file_name}"  # Adjust the path based on where you save files
    return FileResponse(path=file_path, filename=file_name, media_type='text/csv')

