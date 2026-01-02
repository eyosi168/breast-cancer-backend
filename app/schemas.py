from pydantic import BaseModel

class PredictionInput(BaseModel):
    model_type: str  # "LR" or "DT"
    mean_radius: float
    mean_texture: float
    mean_concave_points: float
