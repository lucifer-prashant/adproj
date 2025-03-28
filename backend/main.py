from fastapi import FastAPI
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import uvicorn
from pathlib import Path

from fastapi.middleware.cors import CORSMiddleware

# Import custom models
from models import (
    LinearRegressionModel,
    LogisticRegressionModel,
    KNNModel,
    NaiveBayesModel,
    SVMModel,
    DecisionTreeModel,
    RandomForestModel
)

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Allow frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# Load and preprocess diabetes dataset
data = load_diabetes()
X = data.data
y = (data.target > data.target.mean()).astype(int)  # Convert regression target to classification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define models with descriptive names
models = {
    "linear_regression": {"model": LinearRegressionModel(), "description": "Simple Linear Regression"},
    "logistic_regression": {"model": LogisticRegressionModel(), "description": "Logistic Regression"},
    "knn": {"model": KNNModel(n_neighbors=5), "description": "K-Nearest Neighbors"},
    "naive_bayes": {"model": NaiveBayesModel(), "description": "Gaussian Naive Bayes"},
    "svm": {"model": SVMModel(), "description": "Support Vector Machine (RBF Kernel)"},
    "decision_tree": {"model": DecisionTreeModel(), "description": "Decision Tree"},
    "random_forest": {"model": RandomForestModel(), "description": "Random Forest Ensemble"}
}

# Train models
for model_data in models.values():
    model_data['model'].fit(X_train, y_train)
    model_data['accuracy'] = model_data['model'].score(X_test, y_test)

# Define input schema
class InputData(BaseModel):
    model_type: str
    input_data: dict

@app.post("/predict")
async def predict(data: InputData):
    model_type = data.model_type
    input_data = np.array([list(map(float, data.input_data.values()))]).reshape(1, -1)
    input_data = scaler.transform(input_data)
    
    if model_type not in models:
        return {"error": "Invalid model type"}
    
    model_info = models[model_type]
    model = model_info['model']
    
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1] if hasattr(model, "predict_proba") else None
    
    return {
        "prediction": int(prediction),
        "probability": probability,
        "model_description": model_info.get('description', model_type),
        "model_accuracy": model_info.get('accuracy', None)
    }

@app.get("/models")
async def get_models():
    return {
        "models": [
            {
                "name": key, 
                "description": value['description'],
                "accuracy": value['accuracy']
            } 
            for key, value in models.items()
        ]
    }

@app.get("/ui", response_class=HTMLResponse)
async def ui():
    return templates.TemplateResponse("index.html", {"request": {}})

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
