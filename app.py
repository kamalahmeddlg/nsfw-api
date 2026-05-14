from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

# CORS Fix
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request Model
class ImageData(BaseModel):
    image_url: str

# Home Route
@app.get("/")
def home():
    return {"status": "NSFW API Running"}

# Predict Route
@app.post("/predict")
async def predict(data: ImageData):

    image_url = data.image_url

    # Dummy Prediction
    # Replace later with real AI model

    nsfw = False

    keywords = ["porn", "nude", "xxx", "adult"]

    for word in keywords:
        if word in image_url.lower():
            nsfw = True

    return {
        "success": True,
        "nsfw": nsfw,
        "image": image_url
    }