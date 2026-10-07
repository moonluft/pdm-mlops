"""FastAPI service that predicts machine failure from one sensor reading."""
import logging
import os
import time

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

FEATURES = ["air_temp_k", "process_temp_k", "rotational_speed_rpm", "torque_nm", "tool_wear_min"]

# TODO 3d: use the same THRESHOLD you chose in train.py (TODO 2a).
DEFAULT_THRESHOLD = "0.3"
THRESHOLD = float(os.getenv("THRESHOLD", DEFAULT_THRESHOLD))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("machine-failure-api")


def load_model():
    uri = os.getenv("MODEL_URI")                 # e.g. models:/machine-failure-model@production
    if uri:
        import mlflow.sklearn
        return mlflow.sklearn.load_model(uri), uri
    path = os.getenv("MODEL_PATH", "model.joblib")
    return joblib.load(path), path


model, MODEL_SOURCE = load_model()
log.info("Model loaded from %s", MODEL_SOURCE)

app = FastAPI(title="Machine Failure Prediction API", version="1.0.0",
              description="Predicts whether a milling machine will fail soon, from one sensor reading.")


# TODO 3a: set limits (ge = minimum, le = maximum) so impossible readings get a 422 error.
# We'll use Option B (slightly wider than training limits to accommodate physical reality)
class SensorReading(BaseModel):
    air_temp_k: float = Field(..., ge=290.0, le=310.0, description="Air temperature (K)", examples=[300.0])
    process_temp_k: float = Field(..., ge=300.0, le=320.0, description="Process temperature (K)", examples=[310.5])
    rotational_speed_rpm: float = Field(..., ge=1100.0, le=2900.0, description="Spindle speed", examples=[1550])
    torque_nm: float = Field(..., ge=5.0, le=80.0, description="Torque (Nm)", examples=[62.0])
    tool_wear_min: float = Field(..., ge=0.0, le=300.0, description="Tool wear (minutes)", examples=[230])


class Prediction(BaseModel):
    failure_probability: float
    failure_predicted: bool
    recommended_action: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model-info")
def model_info():
    # TODO 3c: return a dict with the keys "model_source", "threshold" and "features".
    return {
        "model_source": str(MODEL_SOURCE),
        "threshold": THRESHOLD,
        "features": FEATURES
    }


@app.post("/predict", response_model=Prediction)
def predict(reading: SensorReading):
    start = time.perf_counter()
    X = pd.DataFrame([reading.model_dump()])[FEATURES]
    probability = float(model.predict_proba(X)[0, 1])
    predicted = probability >= THRESHOLD

    # TODO 3b: turn the prediction into an action the machine operator can follow.
    # Using Option B (3-level action strategy)
    if probability >= 0.7:
        action = "Stop the machine and inspect now"
    elif predicted:
        action = "Schedule maintenance this shift"
    else:
        action = "No action needed"

    log.info("predict p=%.3f failure=%s latency_ms=%.1f", probability, predicted,
             (time.perf_counter() - start) * 1000)
    return Prediction(failure_probability=round(probability, 4),
                      failure_predicted=predicted, recommended_action=action)
