from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

import tensorflow as tf
import gdown
import os
import numpy as np
from PIL import Image
import io

# =========================
# DOWNLOAD MODEL
# =========================

MODEL_PATH = "model.keras"

if not os.path.exists(MODEL_PATH):

    url = "https://drive.google.com/uc?id=1oCwWNIij0gtoXbnmdehefi4tfO1QGU_U"

    gdown.download(url, MODEL_PATH, quiet=False)

# =========================
# LOAD MODEL
# =========================

model = tf.keras.models.load_model(MODEL_PATH)

print("Model Loaded Successfully")

# =========================
# FASTAPI
# =========================

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# HOME
# =========================

@app.get("/")
def home():
    return {
        "status": "NSFW API Running"
    }

# =========================
# PREDICT
# =========================

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    # Read uploaded image
    contents = await file.read()

    # Open image
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    # Resize image
    image = image.resize((224, 224))

    # Convert image to array
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)[0][0]

    # Result
    result = prediction > 0.5

    return {
        "success": True,
        "nsfw": bool(result),
        "confidence": float(prediction)
    }
