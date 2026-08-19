# ==========================================================
# SKIN DISEASE AI
# HELPER FUNCTIONS
# ==========================================================

from config import ALLOWED_EXTENSIONS
from dataset_config import (
    CLASS_DISPLAY_NAMES,
    CLASS_DESCRIPTIONS,
    CLASS_RISK_LEVEL,
    get_disease_info,
    get_risk_percentage
)


# ==========================================================
# Check Allowed File Extension
# ==========================================================

def allowed_file(filename):
    """Check whether uploaded file has a valid image extension."""
    if not filename:
        return False
    if "." not in filename:
        return False
    extension = filename.rsplit(".", 1)[1].lower()
    return extension in ALLOWED_EXTENSIONS


# ==========================================================
# Format Confidence
# ==========================================================

def format_confidence(confidence):
    """Convert confidence into percentage with 2 decimal places."""
    return round(float(confidence), 2)


# ==========================================================
# Risk Level
# ==========================================================

def get_risk_level(disease):
    """
    Estimate disease risk based on predicted disease class.
    Uses centralized config from dataset_config.py.
    This is NOT a medical diagnosis.
    """
    level = CLASS_RISK_LEVEL.get(disease, "unknown")
    level_display = {
        "high": "High Risk",
        "moderate": "Moderate Risk",
        "low": "Low Risk",
        "none": "No Risk",
        "unknown": "Unknown"
    }
    return level_display.get(level, "Unknown")


# ==========================================================
# Stage Information
# ==========================================================

def get_stage(disease):
    """
    HAM10000 dataset does not contain stage labels.
    Therefore stage prediction is not possible.
    """
    return "Not Available"


# ==========================================================
# Disease Description
# ==========================================================

def get_short_description(disease):
    """Get short description from centralized config."""
    return CLASS_DESCRIPTIONS.get(
        disease,
        "No description available."
    )