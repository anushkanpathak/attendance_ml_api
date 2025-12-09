from fastapi import FastAPI
from pydantic import BaseModel
from .preprocess import predict_risk

app = FastAPI(title="Attendance Risk Prediction API")


class AttendanceFeatures(BaseModel):
    overall_att: float
    last7: float
    last30: float
    streak: float
    trend: float


@app.get("/")
def root():
    return {"message": "Attendance ML API is running"}


@app.post("/predict-attendance")
def predict_attendance(features: AttendanceFeatures):
    label, proba = predict_risk(features.dict())
    risk_label = "At Risk" if label == 1 else "Safe"
    return {
        "risk_class": label,
        "risk_label": risk_label,
        "probability": round(proba, 3)
    }
