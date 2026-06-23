from fastapi import FastAPI, APIRouter
from req.iris_input import IrisInput 
from req.california_housing_input import CaliforniaHousingInput
from joblib import load
import numpy as np

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello World"}

@app.get("/items/{item_id}")
def read_item(item_id: int):
    return {"item_id": item_id}

# Prediction route
prediction_router = APIRouter(
    prefix="/prediction",
    tags=["Prediction"]
)
app.include_router(prediction_router)

model = load("models/iris_model_1.joblib")
@prediction_router.post("/iris")
def predict_iris(payload: IrisInput):
    features = np.array([
        [
            payload.sepal_length,
            payload.sepal_width,
            payload.petal_length,
            payload.petal_width
        ]
    ])

    prediction = model.predict(features)[0]
    target_names = ["setosa", "versicolor", "virginica"]

    result = {
        "predicted": target_names[prediction],
    }
    return result

california_model = load("models/california_housing_model_1.joblib")
@prediction_router.post("/california-house")
def predict_california_housing(payload:CaliforniaHousingInput):
    features = np.array([[
        payload.MedInc/10000,
        payload.AveRooms,
        payload.AveBedrms,
        payload.Latitude,
        payload.Longtitude
    ]])

    prediction = california_model.predict(features)[0]
    calculated_prediction_income = prediction*10000

    result = {
        "Predicted house price": str(f"${calculated_prediction_income:,.2f}")
    }
    return result