

from fastapi import FastAPI, File, UploadFile
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import numpy as np
from io import BytesIO
import tensorflow as tf

app = FastAPI(title="Leukemia CNN API")
MODEL = tf.keras.models.load_model(r"C:\Pre\Model\leukemia_cnn_model.keras")

class_names = ["ALL", "AML", "CLL", "CML", "NORMAL"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

IMAGE_SIZE = 224
@app.get("/ping")
async def ping():
    return "Hello"
def read_image(data):
    image = Image.open(BytesIO(data)).convert("RGB")
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)  # shape = (1, 224, 224, 3)
    return image

@app.post("/predict")
async def predict(
       file: UploadFile = File(...)
):
    data = await file.read()
    img = read_image(data)
    prediction = MODEL.predict(img)[0]
    idx = int(np.argmax(prediction))
    label = class_names[idx]
    confidence = float(prediction[idx]) * 100

    return {
        "prediction": label,
        "confidence": round(confidence, 2),
        "probabilities": {
            "ALL": float(prediction[0]),
            "AML": float(prediction[1]),
            "CLL": float(prediction[2]),
            "CML": float(prediction[3]),
            "NORMAL": float(prediction[4]),
        }
    }



if __name__ == "__main__":
    uvicorn.run(app, host='localhost',port=8000)