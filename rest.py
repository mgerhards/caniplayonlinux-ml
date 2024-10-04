from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from utils import load_model_and_encoders
from lib import predict_compatibility

app = FastAPI()

# Laden Sie das Modell und die Encoder beim Start der API
model, le_title, le_gpu_manufacturer, le_gpu_model, le_distribution, le_cpu, le_ram, le_kernel, scaler = load_model_and_encoders()

class GameCompatibilityRequest(BaseModel):
    title: str
    gpu: Optional[str] = None
    distribution: Optional[str] = None
    cpu: Optional[str] = None
    ram: Optional[str] = None
    kernel: Optional[str] = None

@app.post("/predict")
async def predict_game_compatibility(request: GameCompatibilityRequest):
    try:
        result = predict_compatibility(
            request.title,
            request.gpu,
            request.distribution,
            request.cpu,
            request.ram,
            request.kernel,
            model, le_title, le_gpu_manufacturer, le_gpu_model,
            le_distribution, le_cpu, le_ram, le_kernel, scaler
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Welcome to the Game Compatibility Prediction API"}
