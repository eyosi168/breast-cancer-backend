from pydantic import BaseModel

class PredictionInput(BaseModel):
    model_type: str  # "LR" or "DT"

    radius_mean: float
    texture_mean: float
    perimeter_mean: float
    area_mean: float
    concavity_mean: float
    concave_points_mean: float
    radius_worst: float
    concave_points_worst: float
