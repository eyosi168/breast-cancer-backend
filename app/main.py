from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.schemas import PredictionInput
from app.model_loader import get_model
import numpy as np

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"status": "Server is running!"}

@app.post("/predict")
def predict(data: PredictionInput):
    model = get_model(data.model_type)

    features = np.array([[
        data.radius_mean,
        data.texture_mean,
        data.perimeter_mean,
        data.area_mean,
        data.concavity_mean,
        data.concave_points_mean,
        data.radius_worst,
        data.concave_points_worst
    ]])

    prediction = model.predict(features)[0]

    result = "Malignant" if prediction == 1 else "Benign"

    return {
        "model_used": data.model_type,
        "prediction": result
    }
