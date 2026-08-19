import os
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

load_dotenv(os.path.join(BASE_DIR, ".env"))


# ==========================================================
# STATIC FOLDER
# ==========================================================

STATIC_FOLDER = os.path.join(
    BASE_DIR,
    "static"
)


# ==========================================================
# UPLOAD / REPORT / MODEL FOLDERS
# ==========================================================

UPLOAD_FOLDER = os.path.join(
    STATIC_FOLDER,
    "uploads"
)

REPORT_FOLDER = os.path.join(
    STATIC_FOLDER,
    "reports"
)

MODEL_FOLDER = os.path.join(
    BASE_DIR,
    "models"
)


# ==========================================================
# CREATE REQUIRED DIRECTORIES
# ==========================================================

os.makedirs(
    STATIC_FOLDER,
    exist_ok=True
)

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)

os.makedirs(
    REPORT_FOLDER,
    exist_ok=True
)

os.makedirs(
    MODEL_FOLDER,
    exist_ok=True
)


# ==========================================================
# MODEL
# ==========================================================

MODEL_NAME = "skin_model.keras"

MODEL_PATH = os.path.join(
    MODEL_FOLDER,
    MODEL_NAME
)


# ==========================================================
# IMAGE
# ==========================================================

IMAGE_SIZE = (
    224,
    224
)

IMAGE_CHANNELS = 3


# ==========================================================
# UPLOAD
# ==========================================================

ALLOWED_EXTENSIONS = {
    "jpg",
    "jpeg",
    "png",
    "webp"
}

MAX_CONTENT_LENGTH = (
    10 * 1024 * 1024
)


# ==========================================================
# PREDICTION
# ==========================================================

TOP_PREDICTIONS = 3

CONFIDENCE_THRESHOLD = 0.50


# If confidence is below this value,
# the result is treated as uncertain.

NO_DISEASE_CONFIDENCE = 0.50


# ==========================================================
# GEMINI FALLBACK MODELS
# ==========================================================
GEMINI_FALLBACK_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-1.5-flash",
]


# ==========================================================
# HEALTHY CLASS THRESHOLD
# ==========================================================
#
# IMPORTANT:
#
# This value is used by app.py when the model
# predicts the Healthy class.
#
# app.py compares this with confidence expressed
# as a percentage, for example:
#
#     87.50
#
# Therefore the threshold is:
#
#     85.0
#
# ==========================================================

HEALTHY_THRESHOLD = 85.0


# ==========================================================
# IMAGE QUALITY GATE
# ==========================================================
# If the image quality score from the local validator
# falls below this threshold, prediction is blocked.
QUALITY_GATE_THRESHOLD = 40.0


# ==========================================================
# FLASK
# ==========================================================

SECRET_KEY = os.getenv(
    "SKINAI_SECRET_KEY",
    "SkinAI2026_FinalYear_Project"
)

DEBUG = True


# ==========================================================
# LOGIN
# ==========================================================

LOGIN_USERNAME = os.getenv(
    "SKINAI_USERNAME",
    "admin"
)

LOGIN_PASSWORD = os.getenv(
    "SKINAI_PASSWORD",
    "SkinAI@2026"
)


# ==========================================================
# GEMINI
# ==========================================================

GEMINI_API_KEY = os.getenv(
    "GEMINI_API_KEY",
    ""
) or os.getenv("GOOGLE_API_KEY", "")

GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.6-flash"
)


# ==========================================================
# APPLICATION
# ==========================================================

APP_NAME = (
    "Skin Disease Classification "
    "& Stage Identification"
)

VERSION = "3.0"

AUTHOR = "Riya Kesharwani"


# ==========================================================
# MODEL CHECK
# ==========================================================

if os.path.exists(MODEL_PATH):

    print(
        "\n=========================================="
    )

    print(
        "AI MODEL FOUND"
    )

    print(
        "=========================================="
    )

    print(
        "Model:",
        MODEL_PATH
    )

else:

    print(
        "\n=========================================="
    )

    print(
        "WARNING: AI MODEL NOT FOUND!"
    )

    print(
        "Expected Model Path:"
    )

    print(
        MODEL_PATH
    )

    print(
        "=========================================="
    )