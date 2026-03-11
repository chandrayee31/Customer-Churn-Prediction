from fastapi import FastAPI
from app.schema import ChurnRequest
from app.predictor import predict_churn

app = FastAPI(title="Customer Churn Prediction API")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(request: ChurnRequest):
    return predict_churn(request.dict())