# from project_name.models.svm import run_pipeline
from fastapi import FastAPI, UploadFile, File, HTTPException
from starlette.responses import RedirectResponse
import joblib
import numpy as np
import cv2
from pydantic import BaseModel
from project_name.features.feature_extraction import FeatureExtraction
from project_name.data.preprocessing import resize, normalise


class Prediction(BaseModel):
    filename: str | None = None
    prediction: str


svm = joblib.load("project_name/models/svm_model.pkl")
scaler = joblib.load("project_name/models/scaler.pkl")

CLASS_NAMES = [
    "cumulus", "altocumulus", "cirrus",
    "clearsky", "stratocumulus", "cumulonimbus", "mixed"
]


app = FastAPI(
    title="Cloud Classifier",
    description="An API to classify cloud types from images."
)

@app.get("/", description="Redirects to documentation.")
async def root():
    return RedirectResponse(url='/docs')

# run_pipeline()


@app.post("/predict", description="Upload a cloud image to get a prediction.")
async def predict(image: UploadFile = File(...)):

    if not image.filename or not image.filename.endswith(('.jpg', '.jpeg', '.png')):
        raise HTTPException(status_code=415, detail="Type of the file is not supported. Only .jpg, .jpeg and .png are allowed.")
    

    contents = await image.read()
    np_arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    

    img = resize(np.array([img]))[0]
    img = normalise(np.array([img]))[0]
    

    feature_extractor = FeatureExtraction()
    feats = np.array([list(feature_extractor.run(img).values())])
    
    
    feats_scaled = scaler.transform(feats)
    
    #
    prediction = svm.predict(feats_scaled)[0]
    
    return Prediction(
        filename=image.filename,
        prediction=CLASS_NAMES[prediction]
    )