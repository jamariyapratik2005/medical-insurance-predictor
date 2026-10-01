# Medical Insurance Premium Prediction - FastAPI

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import pickle
import os

# Load saved model
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model.pkl')
with open(model_path, 'rb') as f:
    model_data = pickle.load(f)

model = model_data['model']
le_sex = model_data['le_sex']
le_smoker = model_data['le_smoker']
le_region = model_data['le_region']
model_name = model_data.get('model_name', 'XGBoost Regressor')
r2_score = model_data.get('r2_score', 0.8837)

# FastAPI app
app = FastAPI(
    title="Medical Insurance Premium Prediction API",
    description="Predict health insurance premium based on age, BMI, gender, smoking habits etc. using XGBoost Regressor.",
    version="1.0.0"
)

# Serve static files
static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Request body schema
class PredictionInput(BaseModel):
    age: int
    sex: str          # "male" or "female"
    bmi: float
    children: int
    smoker: str       # "yes" or "no"
    region: str       # "northeast", "northwest", "southeast", "southwest"

    class Config:
        json_schema_extra = {
            "example": {
                "age": 30,
                "sex": "male",
                "bmi": 25.0,
                "children": 1,
                "smoker": "no",
                "region": "northeast"
            }
        }

@app.get("/", response_class=FileResponse)
def home():
    html_path = os.path.join(static_dir, 'index.html')
    return FileResponse(html_path)

@app.post("/predict")
def predict(data: PredictionInput):
    # Encode categorical values
    sex_encoded = le_sex.transform([data.sex])[0]
    smoker_encoded = le_smoker.transform([data.smoker])[0]
    region_encoded = le_region.transform([data.region])[0]

    # Predict
    features = [[data.age, sex_encoded, data.bmi, data.children, smoker_encoded, region_encoded]]
    prediction_usd = float(model.predict(features)[0])
    prediction_inr = round(prediction_usd * 83, 2)  # USD to INR conversion

    return {
        "predicted_premium": float(prediction_inr),
        "predicted_premium_usd": round(prediction_usd, 2),
        "model_used": model_name,
        "r2_score": r2_score,
        "input": data.model_dump()
    }
