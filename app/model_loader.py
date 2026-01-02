import joblib

lr_model = joblib.load("ml/models/logistic_regression_pipeline.joblib")
dt_model = joblib.load("ml/models/decision_tree_pipeline.joblib")


def get_model(model_type: str):
    if model_type == "LR":
        return lr_model
    elif model_type == "DT":
        return dt_model
    else:
        raise ValueError("Invalid model type")
