from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import tensorflow as tf
import gdown
import os

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
# REQUEST MODEL
# =========================

class ImageData(BaseModel):
    image_url: str

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
async def predict(data: ImageData):

    image_url = data.image_url

    # Temporary prediction
    # Real prediction later

    result = False

    return {
        "success": True,
        "nsfw": result,
        "image": image_url
    }
