from fastapi import UploadFile, File
from PIL import Image
import numpy as np
import io


async def prepare_image_grey(model, file: UploadFile = File(...)):
    image_bytes = await file.read()
    img = Image.open(io.BytesIO(image_bytes)).convert('L')
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=(0, -1))

    prob_tensor = model(img_array, training=False)
    prob = prob_tensor.numpy()[0]

    prob_index = np.argmax(prob)
    confidence = float(np.max(prob))
    return  prob_index, confidence