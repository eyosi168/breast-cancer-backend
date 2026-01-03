import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv("dataset/breast_cancer.csv")

# Clean column names
df.columns = (
    df.columns
    .str.strip()
    .str.replace(" ", "_")
)

# Target variable
y = df["diagnosis"].map({"M": 1, "B": 0})

# IMPORTANT FEATURES (8)
FEATURES = [
    "radius_mean",
    "texture_mean",
    "perimeter_mean",
    "area_mean",
    "concavity_mean",
    "concave_points_mean",
    "radius_worst",
    "concave_points_worst"
]

X = df[FEATURES]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Pipelines
lr_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000))
])

dt_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", DecisionTreeClassifier(random_state=42))
])

# Train models
lr_pipeline.fit(X_train, y_train)
dt_pipeline.fit(X_train, y_train)

# Evaluation
print("\n Logistic Regression Evaluation")
lr_preds = lr_pipeline.predict(X_test)
print("Accuracy:", accuracy_score(y_test, lr_preds))
print(classification_report(y_test, lr_preds))

print("\n Decision Tree Evaluation")
dt_preds = dt_pipeline.predict(X_test)
print("Accuracy:", accuracy_score(y_test, dt_preds))
print(classification_report(y_test, dt_preds))

# Save models
joblib.dump(lr_pipeline, "ml/models/logistic_regression_pipeline.joblib")
joblib.dump(dt_pipeline, "ml/models/decision_tree_pipeline.joblib")

print("\n Models trained, evaluated, and saved successfully")
