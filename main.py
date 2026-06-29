from fastapi import FastAPI, APIRouter, UploadFile, File
from fastapi.responses import StreamingResponse
from req.iris_input import IrisInput 
from req.california_housing_input import CaliforniaHousingInput
from joblib import load
import numpy as np
import tensorflow as tf
from utils.preprocess_image import prepare_image_grey
import os
import io
import zipfile

app = FastAPI()

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

fashion_model = tf.keras.models.load_model("models/fashion_model_1.keras")
@prediction_router.post("/fashion")
async def predict_fashion(file: UploadFile = File(...)):
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal',
               'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    prob_index, confidence = await prepare_image_grey(fashion_model,file)
    return {
        "Predicted Category": class_names[prob_index],
        "Confidence": f"{confidence*100:.2f}%"
    }

# Download route
download_router = APIRouter(
    prefix="/download",
    tags=["Download Example"]
)
app.include_router(download_router)

@download_router.get("/fashion-example", summary="Download fashion image for testing")
async def downlod_fashion_example():
    folder_path = "data/fashion-example"

    if not os.path.exists(folder_path) or not os.listdir(folder_path):
        return {
            "message": "Not founded"
        }

    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path):
                zip_file.write(file_path, filename)

    zip_buffer.seek(0)
    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={
            "Content-Disposition":"attachment; filename=fashion-example",
            "Cache-Control":"no-cache"
        }
    )