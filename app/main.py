from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.schemas import PredictionInput
from app.model_loader import get_model
import numpy as np

app = FastAPI()

# 🔓 CORS CONFIGURATION (required for frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all during development
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
        data.mean_radius,
        data.mean_texture,
        data.mean_concave_points
    ]])

    prediction = model.predict(features)[0]

    result = "Malignant" if prediction == 1 else "Benign"

    return {
        "model_used": data.model_type,
        "prediction": result
    }
