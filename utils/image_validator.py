"""
SkinAI local image-quality validator.

IMPORTANT:
- This module does NOT decide human vs animal vs plant vs object.
- It does NOT diagnose disease.
- It does NOT reject a genuine human-skin image.
- Gemini performs semantic human-skin screening in app.py.

The validator only reports technical image quality so it can be shown
in the UI without breaking the prediction pipeline.
"""

from __future__ import annotations

import os
from typing import Any, Dict

import numpy as np
from PIL import Image, ImageFilter, ImageStat


def _clip(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return float(np.clip(float(value), low, high))


def _blur_score(gray: Image.Image) -> float:
    """Estimate sharpness using variance of a high-pass image."""
    try:
        arr = np.asarray(gray, dtype=np.float32)
        if arr.size < 4:
            return 0.0

        # Difference from a small Gaussian-blurred version.
        blurred = gray.filter(ImageFilter.GaussianBlur(radius=1.2))
        blurred_arr = np.asarray(blurred, dtype=np.float32)
        detail = arr - blurred_arr
        variance = float(np.var(detail))

        # This is deliberately only a quality score, not a hard gate.
        return _clip((variance / 35.0) * 100.0)
    except Exception:
        return 50.0


def validate_skin_image(image_path: str) -> Dict[str, Any]:
    """
    Return a technical quality score for an uploaded image.

    The return shape is compatible with app.py:
        {"score": float, "status": str, ...}

    A low score is informational only. The Flask prediction pipeline
    must continue to Gemini/TensorFlow for valid RGB images.
    """

    result: Dict[str, Any] = {
        "score": 100.0,
        "status": "good",
        "width": 0,
        "height": 0,
        "brightness": 0.0,
        "contrast": 0.0,
        "sharpness": 0.0,
        "message": "Image quality is suitable for analysis.",
    }

    try:
        if not image_path or not os.path.isfile(image_path):
            result.update(
                score=0.0,
                status="unavailable",
                message="Image file could not be accessed.",
            )
            return result

        with Image.open(image_path) as original:
            image = original.convert("RGB")
            width, height = image.size
            result["width"] = int(width)
            result["height"] = int(height)

            if width < 80 or height < 80:
                # Informational only; app.py has its own basic size check.
                result.update(
                    score=35.0,
                    status="low_quality",
                    message="Image resolution is low for reliable analysis.",
                )
                return result

            gray = image.convert("L")
            stat = ImageStat.Stat(gray)

            brightness = float(stat.mean[0])
            contrast = float(stat.stddev[0])
            sharpness = _blur_score(gray)

            result["brightness"] = round(brightness, 2)
            result["contrast"] = round(contrast, 2)
            result["sharpness"] = round(sharpness, 2)

            # Brightness: middle range is best; very dark/bright images
            # receive a lower technical-quality score.
            if 55 <= brightness <= 205:
                brightness_score = 100.0
            elif 35 <= brightness < 55 or 205 < brightness <= 225:
                brightness_score = 75.0
            elif 20 <= brightness < 35 or 225 < brightness <= 240:
                brightness_score = 50.0
            else:
                brightness_score = 25.0

            # Contrast: extremely flat images are harder to analyze.
            if contrast >= 35:
                contrast_score = 100.0
            elif contrast >= 25:
                contrast_score = 85.0
            elif contrast >= 15:
                contrast_score = 65.0
            else:
                contrast_score = 40.0

            # Resolution score, kept generous because faces/arms can be
            # analyzed at moderate resolution after resizing.
            pixels = width * height
            if pixels >= 800 * 800:
                resolution_score = 100.0
            elif pixels >= 500 * 500:
                resolution_score = 90.0
            elif pixels >= 300 * 300:
                resolution_score = 75.0
            else:
                resolution_score = 60.0

            score = (
                0.25 * resolution_score
                + 0.25 * brightness_score
                + 0.20 * contrast_score
                + 0.30 * sharpness
            )
            score = round(_clip(score), 2)

            if score >= 75:
                status = "good"
                message = "Image quality is suitable for analysis."
            elif score >= 55:
                status = "acceptable"
                message = "Image quality is acceptable; clearer lighting/focus may improve results."
            else:
                status = "low_quality"
                message = "Image quality is low; a clearer, well-lit image is recommended."

            result.update(
                score=score,
                status=status,
                message=message,
            )
            return result

    except Exception as exc:
        # Never make the local validator a hard failure.
        result.update(
            score=100.0,
            status="validator_error",
            message=f"Local quality check skipped: {exc}",
        )
        return result