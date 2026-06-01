# from project_name.models.svm import run_pipeline
from fastapi import FastAPI, UploadFile, File, HTTPException
from starlette.responses import RedirectResponse
import joblib
import numpy as np
import cv2
from pydantic import BaseModel
from project_name.features.feature_extraction import FeatureExtraction
from project_name.data.preprocessing import IMAGE_SIZE, resize, normalise
from pathlib import Path
import tensorflow as tf

class Prediction(BaseModel):
    filename: str | None = None
    prediction: str
    confidence: float | None = None


svm = joblib.load("project_name/models/svm_model.pkl")
scaler = joblib.load("project_name/models/scaler.pkl")
cnn_model = None
CNN_MODEL_PATH = Path("project_name/models/cnn_model.keras")

CLASS_NAMES = [
    "cumulus", "altocumulus", "cirrus",
    "clearsky", "stratocumulus", "cumulonimbus", "mixed"
]


def validate_image_file(image: UploadFile) -> None:
    if not image.filename or not image.filename.endswith((".jpg", ".jpeg", ".png")):
        raise HTTPException(
            status_code=415,
            detail="Type of the file is not supported. Only .jpg, .jpeg and .png are allowed.",
        )


def get_cnn_model():
    global cnn_model
    if cnn_model is None:
        if not CNN_MODEL_PATH.exists():
            raise HTTPException(
                status_code=503,
                detail=f"CNN model file not found at {CNN_MODEL_PATH}. Train and save the CNN model first.",
            )
        cnn_model = tf.keras.models.load_model(CNN_MODEL_PATH)
    return cnn_model


def preprocess_for_cnn(contents: bytes) -> np.ndarray:
    image = tf.io.decode_image(contents, channels=3, expand_animations=False)
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.image.rgb_to_grayscale(image)
    image = tf.cast(image, tf.float32) / 255.0
    return tf.expand_dims(image, axis=0).numpy()


app = FastAPI(
    title="Cloud Classifier",
    description="An API to classify cloud types from images."
)

@app.get("/", description="Redirects to documentation.")
async def root():
    return RedirectResponse(url='/docs')

# run_pipeline()


@app.post("/predict", description="Upload a cloud image to get an SVM prediction.")
async def predict(image: UploadFile = File(...)):
    validate_image_file(image)
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

@app.post("/predict/cnn", description="Upload a cloud image to get a CNN prediction.")
async def predict_cnn(image: UploadFile = File(...)):
    validate_image_file(image)

    contents = await image.read()
    model_input = preprocess_for_cnn(contents)

    logits = get_cnn_model().predict(model_input, verbose=0)[0]
    probabilities = tf.nn.softmax(logits).numpy()
    prediction = int(np.argmax(probabilities))

    return Prediction(
        filename=image.filename,
        prediction=CLASS_NAMES[prediction],
        confidence=float(probabilities[prediction]),
    )