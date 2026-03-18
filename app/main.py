import os
import json
import shutil
from pathlib import Path
from tempfile import NamedTemporaryFile

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from app.predictor import predict_churn
from app.schema import ChurnRequest
from app.eda.eda_analysis import run_eda

app = FastAPI(title="Churn Prediction API with EDA 🚀")

# ==============================
# Paths
# ==============================
#Blocked for running docker
# EDA_OUTPUT_DIR = "eda_outputs"
# EDA_INPUT_DIR = "eda_inputs"

# Path(EDA_OUTPUT_DIR).mkdir(exist_ok=True)
# Path(EDA_INPUT_DIR).mkdir(exist_ok=True)
#Block End for running docker
#start block for running in local
BASE_DIR = Path(__file__).resolve().parent.parent

EDA_OUTPUT_DIR = BASE_DIR / "eda_outputs"
EDA_INPUT_DIR = BASE_DIR / "eda_inputs"

EDA_OUTPUT_DIR.mkdir(exist_ok=True)
EDA_INPUT_DIR.mkdir(exist_ok=True)
#end block for running in Local
# Static mount for images
# app.mount("/eda-files", StaticFiles(directory=EDA_OUTPUT_DIR), name="eda_files")
app.mount("/eda_outputs", StaticFiles(directory=str(EDA_OUTPUT_DIR)), name="eda_outputs")


# ==============================
# Health
# ==============================
@app.get("/")
def home():
    return {"message": "Churn Prediction API is running 🚀"}


# ==============================
# Prediction Endpoint
# ==============================
@app.post("/predict")
def predict_endpoint(data: ChurnRequest):
    try:
        result = predict_churn(data.dict())
        return {"prediction": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==============================
# EDA - Run from file path
# ==============================
@app.post("/eda/run")
def run_eda_from_path():
    file_path = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Dataset not found")

    try:
        summary = run_eda(file_path, EDA_OUTPUT_DIR)
        return {
            "message": "EDA completed ✅",
            "summary": summary,
            "plots": [f"/eda-files/{p}" for p in summary["plots"]],
            "gallery": "/eda/gallery"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==============================
# EDA - Upload CSV
# ==============================
@app.post("/eda/upload")
async def run_eda_upload(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Upload CSV file only")

    with NamedTemporaryFile(delete=False, suffix=".csv", dir=EDA_INPUT_DIR) as temp:
        shutil.copyfileobj(file.file, temp)
        temp_path = temp.name

    try:
        summary = run_eda(temp_path, str(EDA_OUTPUT_DIR))
        return {
            "message": "EDA completed ✅",
            "plots": [f"app/eda-outputs/{p}" for p in summary["plots"]],
            "gallery": "/eda/gallery"
        }
    finally:
        os.remove(temp_path)


# ==============================
# EDA Summary
# ==============================
@app.get("/eda/summary")
def get_summary():
    path = os.path.join(EDA_OUTPUT_DIR, "summary.json")

    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Run EDA first")

    with open(path) as f:
        return json.load(f)


# ==============================
# EDA Gallery (UI)
# ==============================
@app.get("/eda/gallery", response_class=HTMLResponse)
def gallery():
    images = [f for f in os.listdir(EDA_OUTPUT_DIR) if f.endswith(".png")]

    if not images:
        return "<h2>No plots yet. Run /eda/run</h2>"

    html = "<h1>EDA Visualization</h1>"
    for img in images:
        html += f"""
        <div style="margin-bottom:30px">
            <h3>{img}</h3>
            <img src="/eda-files/{img}" style="width:800px"/>
        </div>
        """

    return html