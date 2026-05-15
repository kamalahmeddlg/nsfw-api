import os
import io
import logging
from contextlib import asynccontextmanager
from typing import Optional

import gdown
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field

# ------------------------------------------------------------
# Configuration via environment variables
# ------------------------------------------------------------
MODEL_URL: str = os.getenv(
    "MODEL_URL",
    "https://drive.google.com/uc?id=1oCwWNIij0gtoXbnmdehefi4tfO1QGU_U",
)
MODEL_PATH: str = os.getenv("MODEL_PATH", "model.keras")
ALLOWED_ORIGINS: list[str] = os.getenv("ALLOWED_ORIGINS", "*").split(",")
MAX_IMAGE_SIZE_MB: int = int(os.getenv("MAX_IMAGE_SIZE_MB", "10"))
NSFW_THRESHOLD: float = float(os.getenv("NSFW_THRESHOLD", "0.5"))

# Supported image MIME types
ALLOWED_CONTENT_TYPES: set[str] = {
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/bmp",
    "image/tiff",
}

# ------------------------------------------------------------
# Logging setup
# ------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("nsfw-api")

# ------------------------------------------------------------
# Model Manager (loaded once at startup)
# ------------------------------------------------------------
model: Optional[tf.keras.Model] = None

def download_model() -> None:
    """Download the model if not present, with basic retry logic."""
    if os.path.exists(MODEL_PATH):
        logger.info("Model already exists at %s", MODEL_PATH)
        return

    logger.info("Downloading model from %s ...", MODEL_URL)
    try:
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
        logger.info("Model downloaded successfully.")
    except Exception as exc:
        logger.error("Failed to download model: %s", exc)
        raise RuntimeError(f"Model download failed: {exc}") from exc


def load_model() -> tf.keras.Model:
    """Load the Keras model from disk."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    logger.info("Loading TensorFlow model from %s ...", MODEL_PATH)
    return tf.keras.models.load_model(MODEL_PATH)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan handler: runs on startup and shutdown.
    Downloads the model, loads it into memory, and stores it globally.
    """
    global model
    try:
        download_model()
        model = load_model()
        logger.info("Model loaded successfully. Ready to serve.")
    except Exception as e:
        logger.critical("Fatal error during model loading: %s", e)
        raise SystemExit(1) from e

    yield  # application runs here

    # Cleanup on shutdown (if needed)
    logger.info("Shutting down, clearing model...")
    model = None


# ------------------------------------------------------------
# Application instance
# ------------------------------------------------------------
app = FastAPI(
    title="NSFW Image Detection API",
    description="A professional API to detect potentially unsafe/sexual content in images using deep learning.",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------------------------------------
# Pydantic Response Models
# ------------------------------------------------------------
class HealthResponse(BaseModel):
    status: str = Field(..., example="ok")
    model_loaded: bool = Field(..., example=True)


class PredictionResponse(BaseModel):
    success: bool
    nsfw: bool
    confidence: float = Field(..., ge=0.0, le=1.0)


# ------------------------------------------------------------
# Utility: async image preprocessing (runs in default executor)
# ------------------------------------------------------------
def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """
    Convert raw bytes to a normalized (224,224,3) numpy array.
    Raises ValueError if image cannot be opened or is invalid.
    """
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except UnidentifiedImageError:
        raise ValueError("Uploaded file is not a valid image.")

    image = image.resize((224, 224))
    img_array = np.array(image, dtype=np.float32) / 255.0
    return np.expand_dims(img_array, axis=0)


async def async_predict(image_bytes: bytes) -> tuple[bool, float]:
    """
    Run image preprocessing and model prediction in a thread pool
    to avoid blocking the async event loop.
    """
    global model
    if model is None:
        raise RuntimeError("Model not loaded. The server is not ready.")

    loop = asyncio.get_running_loop()

    # Preprocess in executor
    img_array = await loop.run_in_executor(None, preprocess_image, image_bytes)

    # Predict in executor (TensorFlow's predict is CPU/GPU bound)
    prediction = await loop.run_in_executor(None, model.predict, img_array)
    prob = float(prediction[0][0])

    return prob > NSFW_THRESHOLD, prob


# ------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------
@app.get("/", response_model=dict)
async def root():
    """Basic welcome message with API status."""
    return {"status": "NSFW API Running", "docs": "/docs"}


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Liveness/readiness probe. Checks if model is loaded."""
    return HealthResponse(
        status="ok",
        model_loaded=model is not None,
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(..., description="Image file to analyze")):
    """
    Analyze an uploaded image for NSFW content.

    - **file**: JPEG, PNG, WebP, BMP, or TIFF image (max 10 MB).
    - Returns a boolean `nsfw` flag and a confidence score.
    """
    # 1. Validate content type
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type: {file.content_type}. "
                   f"Allowed types: {', '.join(sorted(ALLOWED_CONTENT_TYPES))}",
        )

    # 2. Read file with size limit
    contents = await file.read()
    if len(contents) > MAX_IMAGE_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Image too large. Maximum size is {MAX_IMAGE_SIZE_MB} MB.",
        )

    # 3. Process and predict
    try:
        is_nsfw, confidence = await async_predict(contents)
    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(ve),
        )
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal error occurred during prediction.",
        )

    return PredictionResponse(success=True, nsfw=is_nsfw, confidence=confidence)


# ------------------------------------------------------------
# Global exception handler for unhandled errors
# ------------------------------------------------------------
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error("Unhandled exception: %s", exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected server error occurred."},
    )
