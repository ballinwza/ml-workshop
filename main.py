from fastapi import FastAPI
from req.iris_input import IrisInput 
from joblib import load
import numpy as np

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello World"}

@app.get("/items/{item_id}")
def read_item(item_id: int):
    return {"item_id": item_id}

model = load("models/iris_model_1.joblib")
@app.post("/predict/iris")
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