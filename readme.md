# Customer Churn Prediction API

![Python](https://img.shields.io/badge/Python-3.9-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-API-green)
![Docker](https://img.shields.io/badge/Docker-Container-blue)
![Scikit-Learn](https://img.shields.io/badge/ML-ScikitLearn-orange)

A Machine Learning powered REST API that predicts whether a telecom
customer is likely to churn.

## Project Overview

This project demonstrates an end-to-end machine learning pipeline
including:

-   Data preprocessing
-   Feature engineering
-   Model comparison
-   Model selection
-   FastAPI deployment
-   Docker containerization

The API accepts customer attributes and returns:

-   churn prediction
-   churn probability

## Project Structure

    churn-prediction-api
    │
    ├── app
    │   ├── main.py
    │   ├── predictor.py
    │   └── schema.py
    │
    ├── model
    │   ├── best_churn_model.joblib
    │   └── feature_names.joblib
    │
    ├── data
    ├── train_model.py
    ├── requirements.txt
    ├── Dockerfile
    └── README.md

## Running Locally

### Create virtual environment

    python3 -m venv venv
    source venv/bin/activate

### Install dependencies

    pip install -r requirements.txt

### Start server

    uvicorn app.main:app --reload

Open Swagger UI:

    http://127.0.0.1:8000/docs

## Running with Docker

### Build image

    docker build -t churn-api .

### Run container

    docker run -p 8000:8000 churn-api

Access API:

    http://localhost:8000/docs

## API Endpoints

  Endpoint   Method   Description
  ---------- -------- ---------------------------
  /health    GET      Health check
  /predict   POST     Predict churn probability

## Example Request

``` json
{
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 12,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "Yes",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "Yes",
  "StreamingMovies": "Yes",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 85.5,
  "TotalCharges": 1026.0
}
```

## Example Response

``` json
{
  "churn_prediction": "Yes",
  "churn_probability": 0.74
}
```

## Author

Chandrayee Kumar
