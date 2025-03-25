from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Load the trained model
model = joblib.load("iris_model.pkl")

# Initialize FastAPI app
app = FastAPI()

# ✅ Define a proper input model
class ModelInput(BaseModel):
    features: list  # FastAPI now expects {"features": [...]}

@app.get("/")
def home():
    return {"message": "Iris Classification API is running!"}

@app.post("/predict/")
def predict(input_data: ModelInput):
    features = np.array(input_data.features).reshape(1, -1)
    prediction = model.predict(features)
    return {"prediction": int(prediction[0])}
