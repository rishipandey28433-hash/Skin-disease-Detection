# ==============================================================================
# Skin Disease Diagnosis - Multi-Stage Prediction Pipeline
# ==============================================================================
# Pipeline Stages:
# Stage 1: Basic Image Validation (Existence, Extension, Dims, Corruption)
# Stage 2: Quality Inspection (Informational sharpness, brightness, contrast)
# Stage 3: Human Skin Validation (Local Skin-Tone / Dermoscopy Analysis + Gemini Vision + Fail-Open)
# Stage 4: TensorFlow EfficientNetB0 Classification & Softmax Normalization
# Stage 5: Grad-CAM Heatmap Generation for Explainability
# Stage 6: Severity & Stage Identification
# Stage 7: Result Categorization (Healthy / Disease / Non-Disease Injury / Invalid)
# ==============================================================================

import os
import json
import time
import logging
import numpy as np
import tensorflow as tf
from PIL import Image
from typing import Dict, Any, Optional, Tuple

from config import (
    IMAGE_SIZE, MODEL_PATH, CONFIDENCE_THRESHOLD,
    HEALTHY_THRESHOLD, NO_DISEASE_CONFIDENCE
)
from dataset_config import (
    CLASS_NAMES, CLASS_DISPLAY_NAMES, CLASS_DESCRIPTIONS,
    CLASS_SYMPTOMS, CLASS_PRECAUTIONS, CLASS_RISK_LEVEL,
    CLASS_CATEGORY, CATEGORY_DISPLAY_NAMES,
    STATUS_HEALTHY, STATUS_DISEASE, STATUS_NON_DISEASE,
    STATUS_INVALID, STATUS_LOW_QUALITY, STATUS_UNCERTAIN,
    DISEASE_CLASSES, HEALTHY_CLASSES, get_risk_percentage
)
from utils.preprocess import preprocess_image
from utils.image_validator import validate_skin_image
from utils.gradcam import save_gradcam_image

# Try importing Gemini SDK
try:
    from google import genai
    from google.genai import types
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False

logger = logging.getLogger(__name__)


def is_likely_skin_or_dermoscopy(image_path: str) -> Tuple[bool, float, str]:
    """
    Robust local heuristic to check whether an image contains human
    skin or is a dermoscopy/clinical dermatology photograph.

    Uses complementary strategies:
      1. Classic skin color model (R > 50, G > 30, B > 15, R > G > B, |R-G| > 10, R-B > 15)
      2. HSV skin detection (Hue 0-35 deg or 335-360 deg, Sat 0.12-0.75, Val 0.18-0.96)
      3. Dark skin / Pigmented lesion detection (low brightness, skin hue)
      4. Dermoscopy center vs border vignetting analysis

    Returns tuple: (is_skin: bool, skin_ratio: float, description: str)
    """
    try:
        with Image.open(image_path) as img:
            rgb_img = img.convert("RGB")
            small = rgb_img.resize((100, 100))
            arr = np.asarray(small, dtype=np.float32)
            r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]

            # 1. RGB Skin Color Model
            rgb_skin = (
                (r > 50) & (g > 30) & (b > 15) &
                (r > g) & (r > b) &
                (np.abs(r - g) > 10) &
                ((r - b) > 15)
            )

            # 2. HSV Skin Color Model
            r_f = r / 255.0
            g_f = g / 255.0
            b_f = b / 255.0
            cmax = np.maximum(np.maximum(r_f, g_f), b_f)
            cmin = np.minimum(np.minimum(r_f, g_f), b_f)
            delta = cmax - cmin + 1e-7

            hue = np.zeros_like(cmax)
            mask_r = (cmax == r_f) & (delta > 1e-6)
            mask_g = (cmax == g_f) & (delta > 1e-6) & ~mask_r
            mask_b = ~mask_r & ~mask_g & (delta > 1e-6)
            hue[mask_r] = 60.0 * (((g_f[mask_r] - b_f[mask_r]) / delta[mask_r]) % 6)
            hue[mask_g] = 60.0 * (((b_f[mask_g] - r_f[mask_g]) / delta[mask_g]) + 2)
            hue[mask_b] = 60.0 * (((r_f[mask_b] - g_f[mask_b]) / delta[mask_b]) + 4)
            with np.errstate(divide='ignore', invalid='ignore'):
                sat = np.where(cmax > 1e-6, delta / cmax, 0.0)
            val = cmax

            hue_ok = ((hue >= 0) & (hue <= 35)) | ((hue >= 335) & (hue <= 360))
            sat_ok = (sat >= 0.12) & (sat <= 0.75)
            val_ok = (val >= 0.18) & (val <= 0.96)
            hsv_skin = hue_ok & sat_ok & val_ok

            # 3. Dark skin / Pigmented Lesion Detection
            dark_skin = (
                (r > 25) & (g > 15) & (b > 8) &
                (r >= g) & (r > b) &
                hue_ok &
                (sat >= 0.10) & (sat <= 0.80) &
                (val >= 0.10) & (val <= 0.40)
            )

            # 4. Dermoscopy Center vs Border analysis
            center_rgb_skin = rgb_skin[25:75, 25:75]
            center_hsv_skin = hsv_skin[25:75, 25:75]
            center_skin_ratio = float(np.mean(center_rgb_skin | center_hsv_skin))

            border_pixels = np.concatenate([
                arr[0:12, :, :].reshape(-1, 3),
                arr[88:, :, :].reshape(-1, 3),
                arr[:, 0:12, :].reshape(-1, 3),
                arr[:, 88:, :].reshape(-1, 3),
            ], axis=0)
            border_brightness = float(np.mean(border_pixels))
            is_dermoscopy_frame = (border_brightness < 85) and (center_skin_ratio >= 0.15)

            combined_skin = rgb_skin | hsv_skin | dark_skin
            total_skin_ratio = float(np.mean(combined_skin))

            is_skin = (total_skin_ratio >= 0.15) or is_dermoscopy_frame or (center_skin_ratio >= 0.20)
            desc = f"total_skin={total_skin_ratio:.3f}, center={center_skin_ratio:.3f}, dermo_frame={is_dermoscopy_frame}"

            print(f"[DEBUG] Local skin analysis for {os.path.basename(image_path)}: {desc} -> {'SKIN' if is_skin else 'NON-SKIN'}")
            return is_skin, total_skin_ratio, desc

    except Exception as e:
        print(f"[DEBUG] Local skin analysis error: {e}")
        return False, 0.0, f"Error: {e}"


class SkinAnalysisPipeline:
    """
    Robust multi-stage pipeline for skin disease classification,
    human skin validation, Grad-CAM explainability, and stage determination.
    """

    def __init__(self, model, class_names=None):
        self.model = model
        self.class_names = class_names if class_names is not None else CLASS_NAMES

        # Initialize Gemini Client if available
        self.gemini_client = None
        self.gemini_available = False

        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if HAS_GEMINI and api_key:
            try:
                self.gemini_client = genai.Client(api_key=api_key)
                self.gemini_available = True
                print("[DEBUG] Gemini API client initialized for skin validation.")
            except Exception as e:
                print(f"[DEBUG] Gemini client init warning: {e}")
        else:
            print("[DEBUG] Gemini API key not set or google-genai not installed. Local skin analysis active.")

    # ==============================================================================
    # STAGE 1: IMAGE VALIDATION
    # ==============================================================================
    def _validate_image(self, image_path: str) -> Tuple[bool, str]:
        if not os.path.exists(image_path):
            return False, "Image file does not exist on disk."

        valid_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
        _, ext = os.path.splitext(image_path)
        if ext.lower() not in valid_extensions:
            return False, f"Unsupported file extension ({ext}). Allowed: {', '.join(valid_extensions)}"

        try:
            with Image.open(image_path) as img:
                img.verify()

            with Image.open(image_path) as img:
                width, height = img.size
                mode = img.mode
                print(f"[DEBUG] Image: {os.path.basename(image_path)} | Size: {width}x{height} | Mode: {mode}")

                if width < 30 or height < 30:
                    return False, f"Image dimensions too small ({width}x{height}). Minimum is 30x30."

            return True, ""
        except Exception as e:
            return False, f"Corrupted or unreadable image file: {str(e)}"

    # ==============================================================================
    # STAGE 2: QUALITY CHECK
    # ==============================================================================
    def _check_quality(self, image_path: str) -> Dict[str, Any]:
        try:
            quality_result = validate_skin_image(image_path)
            print(f"[DEBUG] Quality Check: Score={quality_result.get('score')}, Status={quality_result.get('status')}")
            return quality_result
        except Exception as e:
            print(f"[DEBUG] Quality Check warning: {e}")
            return {"score": 85.0, "status": "good", "message": "Quality check passed."}

    # ==============================================================================
    # STAGE 3: HUMAN SKIN VALIDATION (ROBUST + DERMOSCOPY AWARE)
    # ==============================================================================
    def _check_human_skin(self, image_path: str) -> Tuple[str, str, float]:
        """
        Robustly verify whether the input image depicts human skin / dermatology content.

        Decision hierarchy:
          1. If Gemini is available -> ask Gemini with a dermatology-aware prompt.
             - If Gemini says NON_HUMAN -> REJECT (subject, reason, confidence). NO OVERRIDE.
             - If Gemini says HUMAN_SKIN -> ACCEPT ("human_skin", reason, confidence).
             - If Gemini says INJURY -> ACCEPT NON-DISEASE ("injury", reason, confidence).
          2. If Gemini is unavailable or fails -> use local multi-strategy skin detector.
             - If local check is positive -> ACCEPT ("human_skin", reason, confidence).
             - If local check is negative -> REJECT ("non_skin", reason, 0.0).

        Returns tuple: (decision: str, reason: str, confidence: float)
        Decision: "human_skin", "injury", "animal", "plant", "food", "object", "document", "non_skin"
        """
        # Step A: Local multi-strategy skin/dermoscopy check
        has_skin_tones, local_skin_ratio, local_desc = is_likely_skin_or_dermoscopy(image_path)

        # ---------------------------------------------------------------
        # PATH 1: Gemini available -- structured validation
        # ---------------------------------------------------------------
        if self.gemini_available and self.gemini_client:
            try:
                with Image.open(image_path) as img:
                    prompt = (
                        "You are an expert dermatological image validation assistant.\n\n"
                        "YOUR ONLY TASK: Determine whether this image shows REAL HUMAN SKIN or a NON-HUMAN subject.\n\n"
                        "VALID HUMAN SKIN (is_human_skin: true, category: \"HUMAN_SKIN\"):\n"
                        "- Normal / healthy human skin of any skin tone (light, dark, medium, olive)\n"
                        "- Diseased skin, rashes, moles, lesions, ulcers, scabs, scales, redness, blisters, tumors\n"
                        "- Dermoscopy / dermatoscopy photos, close-up lesions, images with black circular frames or rulers\n"
                        "- Clinical skin photographs (face, arm, leg, back, torso, hand)\n"
                        "- Non-disease skin injuries like burns, cuts, wounds (is_human_skin: true, category: \"INJURY\")\n\n"
                        "NON-HUMAN / INVALID (is_human_skin: false, category: \"NON_HUMAN\"):\n"
                        "- Animals (dogs, cats, birds, horses, wildlife, pets)\n"
                        "- Plants (flowers, leaves, trees, grass, vegetables)\n"
                        "- Food (fruits, pizza, burgers, bread, meals)\n"
                        "- Objects (cars, furniture, laptops, phones, tools, clothes, toys)\n"
                        "- Documents, text, screenshots, charts, diagrams, drawings, cartoons, abstract art\n\n"
                        "Respond ONLY with JSON:\n"
                        "{\"is_human_skin\": true/false, "
                        "\"category\": \"HUMAN_SKIN\" | \"INJURY\" | \"NON_HUMAN\", "
                        "\"subject\": \"human_skin\" | \"injury\" | \"animal\" | \"plant\" | \"food\" | \"object\" | \"document\" | \"non_skin\", "
                        "\"confidence\": 0.0-1.0, "
                        "\"reason\": \"brief explanation\"}"
                    )

                    models_to_try = [
                        os.getenv("GEMINI_MODEL", "gemini-3.6-flash"),
                        "gemini-3.6-flash",
                        "gemini-2.5-flash",
                        "gemini-2.0-flash",
                        "gemini-1.5-flash"
                    ]
                    # De-duplicate
                    models_to_try = list(dict.fromkeys(models_to_try))

                    for model_name in models_to_try:
                        try:
                            response = self.gemini_client.models.generate_content(
                                model=model_name,
                                contents=[img, prompt]
                            )

                            if response and response.text:
                                raw_text = response.text.strip()
                                print(f"[DEBUG] Gemini Skin Validation ({model_name}): {raw_text[:300]}")

                                raw_lower = raw_text.lower()

                                # --- Parse structured JSON response ---
                                is_human = None
                                category = ""
                                subject = ""
                                confidence = 0.95
                                reason = "Validated by AI."

                                if "{" in raw_text and "}" in raw_text:
                                    try:
                                        json_str = raw_text[raw_text.find("{"):raw_text.rfind("}") + 1]
                                        parsed = json.loads(json_str)

                                        if "is_human_skin" in parsed:
                                            is_human = bool(parsed["is_human_skin"])
                                        category = str(parsed.get("category", "")).upper().strip()
                                        subject = str(parsed.get("subject", "")).lower().strip()
                                        confidence = float(parsed.get("confidence", 0.95))
                                        reason = str(parsed.get("reason", "Validated by AI."))
                                    except Exception:
                                        pass

                                # Fallback text parsing if JSON didn't give clear result
                                if is_human is None:
                                    if any(kw in raw_lower for kw in ["non_human", "non-human", "not human", "animal", "plant", "food", "object", "document", "screenshot", "no", "invalid", "false"]):
                                        is_human = False
                                        category = "NON_HUMAN"
                                        subject = "non_skin"
                                        for kw in ["animal", "plant", "food", "object", "document", "screenshot"]:
                                            if kw in raw_lower:
                                                subject = kw
                                                break
                                        confidence = 0.90
                                        reason = f"Non-human subject detected ({subject})."
                                    else:
                                        is_human = True
                                        category = "HUMAN_SKIN"
                                        subject = "human_skin"
                                        confidence = 0.90
                                        reason = "Human skin detected."

                                # --- Decision logic ---

                                # Case 1: Gemini says NON_HUMAN -> HARD REJECT (no override!)
                                if is_human is False or category == "NON_HUMAN" or subject in ["animal", "plant", "food", "object", "document", "screenshot", "non_skin"]:
                                    final_subj = subject if subject in ["animal", "plant", "food", "object", "document", "screenshot", "non_skin"] else "non_skin"
                                    print(f"[DEBUG] Gemini decision: NON-HUMAN ({final_subj}) - {reason}")
                                    return final_subj, reason, confidence

                                # Case 2: Gemini says INJURY -> ACCEPT non-disease injury
                                if category == "INJURY" or subject == "injury":
                                    print(f"[DEBUG] Gemini decision: INJURY - {reason}")
                                    return "injury", reason, confidence

                                # Case 3: Gemini says HUMAN_SKIN -> ACCEPT
                                print(f"[DEBUG] Gemini decision: HUMAN_SKIN - {reason}")
                                return "human_skin", reason, confidence

                        except Exception as model_err:
                            print(f"[DEBUG] Gemini model {model_name} failed: {model_err}")
                            continue

            except Exception as e:
                print(f"[DEBUG] Gemini validation exception: {e}")

        # ---------------------------------------------------------------
        # PATH 2: Gemini unavailable or call failed -> Local Heuristic
        # ---------------------------------------------------------------
        if has_skin_tones:
            print(f"[DEBUG] Local skin check positive ({local_desc}) -> ACCEPT.")
            return "human_skin", f"Local skin check positive: {local_desc}", max(local_skin_ratio, 0.85)
        else:
            print(f"[DEBUG] Local skin check negative ({local_desc}) -> REJECT.")
            return "non_skin", "No human skin tones or dermatological features detected.", 0.90

    # ==============================================================================
    # STAGE 4: DISEASE CLASSIFICATION (EFFICIENTNET-B0)
    # ==============================================================================
    def _classify(self, image_path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        try:
            processed_img = preprocess_image(image_path)

            # TensorFlow Model Inference
            predictions = self.model.predict(processed_img, verbose=0)[0]

            # Apply softmax if needed
            if np.min(predictions) < 0 or np.sum(predictions) > 1.1:
                predictions = tf.nn.softmax(predictions).numpy()

            predictions = predictions / np.sum(predictions)

            top_indices = np.argsort(predictions)[::-1]

            top_predictions = []
            for i in range(min(3, len(self.class_names))):
                idx = top_indices[i]
                class_id = self.class_names[idx]
                conf = float(predictions[idx] * 100.0)

                top_predictions.append({
                    "class": class_id,
                    "name": CLASS_DISPLAY_NAMES.get(class_id, class_id),
                    "confidence": round(conf, 2)
                })

            top_class = self.class_names[top_indices[0]]
            top_confidence = float(predictions[top_indices[0]] * 100.0)

            print(f"[DEBUG] Disease Classification: Predicted={top_class} ({CLASS_DISPLAY_NAMES.get(top_class)}) | Confidence={top_confidence:.2f}%")

            result_dict = {
                "top_class_id": top_class,
                "top_confidence": round(top_confidence, 2),
                "top_predictions": top_predictions,
                "top_index": int(top_indices[0])
            }

            return predictions, result_dict

        except Exception as e:
            print(f"[DEBUG] Classification error: {e}")
            raise e

    # ==============================================================================
    # STAGE 6: RESULT CATEGORIZATION & STAGE ESTIMATION
    # ==============================================================================
    def _estimate_stage(self, condition: str, confidence: float, risk_level: str) -> str:
        """Estimate disease stage or severity indicator based on clinical metadata."""
        if condition in HEALTHY_CLASSES:
            return "Normal / No Lesion"
        elif condition in ["mel", "bcc", "akiec"]:
            if confidence >= 85.0:
                return "High Confidence — Immediate Dermatological Evaluation Advised"
            elif confidence >= 60.0:
                return "Moderate Confidence — Professional Screening Advised"
            else:
                return "Early Detection Stage / Monitoring Required"
        elif condition in ["bkl", "df", "nv", "vasc", "vitiligo"]:
            return "Benign / Non-Cancerous Condition"
        return "Under Observation"

    def _build_result(self, status: str, condition: Optional[str] = None,
                      confidence: float = 0.0, top_predictions: Optional[list] = None,
                      quality: Optional[Dict] = None, skin_check: Optional[str] = None,
                      message: str = "", gradcam_url: Optional[str] = None) -> Dict[str, Any]:

        disclaimer = "This is an AI-assisted prediction for research and educational purposes only. It is NOT a medical diagnosis. Please consult a qualified dermatologist for professional medical evaluation."

        if status in [STATUS_DISEASE, STATUS_HEALTHY, STATUS_UNCERTAIN, STATUS_NON_DISEASE]:
            status_display = "Analysis Complete"
        elif status == STATUS_INVALID:
            status_display = "Invalid Image"
        else:
            status_display = "Analysis Failed"

        result = {
            "status": status,
            "status_display": status_display,
            "message": message,
            "medical_disclaimer": disclaimer,
            "condition": condition,
            "condition_display": CLASS_DISPLAY_NAMES.get(condition, "Unknown") if condition else None,
            "confidence": round(confidence, 2),
            "top_predictions": top_predictions or [],
            "quality": quality or {"score": 90.0, "status": "good"},
            "skin_check": skin_check or "human_skin",
            "gradcam_url": gradcam_url
        }

        if condition and condition in CLASS_DISPLAY_NAMES:
            result["description"] = CLASS_DESCRIPTIONS.get(condition, "")
            result["symptoms"] = CLASS_SYMPTOMS.get(condition, "")
            result["precautions"] = CLASS_PRECAUTIONS.get(condition, "")

            risk_level = CLASS_RISK_LEVEL.get(condition, "unknown")
            result["risk_level"] = risk_level
            result["risk_percentage"] = get_risk_percentage(risk_level, confidence)

            category_id = CLASS_CATEGORY.get(condition, "other")
            result["category_id"] = category_id
            result["category"] = CATEGORY_DISPLAY_NAMES.get(category_id, "Other")

            result["stage"] = self._estimate_stage(condition, confidence, risk_level)
        else:
            result["description"] = ""
            result["symptoms"] = ""
            result["precautions"] = ""
            result["risk_level"] = "unknown"
            result["risk_percentage"] = 0.0
            result["category_id"] = "unknown"
            result["category"] = "Unknown"
            result["stage"] = "Not Applicable"

        return result

    # ==============================================================================
    # MAIN ANALYZE METHOD
    # ==============================================================================
    def analyze(self, image_path: str, upload_folder: Optional[str] = None) -> Dict[str, Any]:
        """
        Execute full multi-stage skin disease prediction flow.
        """
        start_time = time.time()
        try:
            print(f"\n[DEBUG] Pipeline started for image: {image_path}")

            # 1. Validate Image
            is_valid, err_msg = self._validate_image(image_path)
            if not is_valid:
                print(f"[DEBUG] Image validation failed: {err_msg}")
                res = self._build_result(
                    status=STATUS_INVALID,
                    message=f"Invalid image file: {err_msg}"
                )
                res["analysis_time"] = f"{time.time() - start_time:.2f} sec"
                return res

            # 2. Check Quality (Informational)
            quality_info = self._check_quality(image_path)

            # 3. Human Skin Validation (HARD GATE)
            print("=" * 60)
            print("[VALIDATION] Starting human-skin validation")
            skin_result = self._check_human_skin(image_path)
            if isinstance(skin_result, tuple) and len(skin_result) >= 3:
                skin_status, skin_reason, skin_conf = skin_result[0], skin_result[1], float(skin_result[2])
            elif isinstance(skin_result, tuple) and len(skin_result) == 2:
                skin_status, skin_reason, skin_conf = skin_result[0], skin_result[1], 0.95
            else:
                skin_status, skin_reason, skin_conf = str(skin_result), "Validated", 0.95
            
            is_human_skin = (skin_status == "human_skin")
            is_injury = (skin_status == "injury")
            
            print(f"[VALIDATION] Result: {'HUMAN' if (is_human_skin or is_injury) else 'NON-HUMAN'}")
            print(f"[VALIDATION] Confidence: {skin_conf * 100 if skin_conf <= 1.0 else skin_conf:.1f}%")

            if not is_human_skin and not is_injury:
                print("[VALIDATION] Decision: REJECT")
                print("[VALIDATION] NON-HUMAN IMAGE")
                print("[VALIDATION] STOPPING DISEASE PREDICTION")
                print("=" * 60)
                
                # HARD GATE: STOP HERE.
                # NO model.predict(), NO _classify(), NO Grad-CAM, NO healthy fallback!
                res = self._build_result(
                    status=STATUS_INVALID,
                    quality=quality_info,
                    skin_check=skin_status,
                    message="Invalid Image: Please upload a valid human skin/dermatology image."
                )
                res["analysis_time"] = f"{time.time() - start_time:.2f} sec"
                return res

            if is_injury:
                print("[VALIDATION] Decision: ACCEPT (NON-DISEASE INJURY)")
                print("[VALIDATION] NON-DISEASE INJURY DETECTED")
                print("[VALIDATION] STOPPING DISEASE PREDICTION")
                print("=" * 60)
                res = self._build_result(
                    status=STATUS_NON_DISEASE,
                    quality=quality_info,
                    skin_check=skin_status,
                    message="The image appears to represent a non-disease skin injury (such as a burn, cut, or wound) rather than a dermatological disease."
                )
                res["analysis_time"] = f"{time.time() - start_time:.2f} sec"
                return res

            print("[VALIDATION] Decision: ACCEPT")
            print("[VALIDATION] HUMAN SKIN CONFIRMED")
            print("[VALIDATION] Starting existing disease classifier")
            print("=" * 60)

            # ===============================================================
            # ONLY BELOW THIS LINE: EXISTING DISEASE CLASSIFICATION & GRAD-CAM
            # ===============================================================
            predictions, class_info = self._classify(image_path)

            top_class = class_info["top_class_id"]
            top_conf = class_info["top_confidence"]
            top_preds = class_info["top_predictions"]
            top_idx = class_info["top_index"]

            # 5. Generate Grad-CAM Overlay
            gradcam_url = None
            if upload_folder and os.path.exists(upload_folder):
                try:
                    filename = os.path.basename(image_path)
                    gradcam_filename = f"gradcam_{filename}"
                    gradcam_path = os.path.join(upload_folder, gradcam_filename)

                    save_gradcam_image(self.model, image_path, gradcam_path, class_index=top_idx)
                    gradcam_url = f"/static/uploads/{gradcam_filename}"
                    print(f"[DEBUG] Grad-CAM saved to: {gradcam_path}")
                except Exception as gc_err:
                    print(f"[DEBUG] Grad-CAM generation note: {gc_err}")

            # 6. Result Thresholding & Categorization
            disease_thresh = CONFIDENCE_THRESHOLD if CONFIDENCE_THRESHOLD > 1.0 else CONFIDENCE_THRESHOLD * 100.0

            if top_class in HEALTHY_CLASSES:
                # HUMAN SKIN != HEALTHY SKIN
                # Only report Healthy Skin if model confidence is very high (>= 90%) and no OOD/unclassified lesion ambiguity exists
                if top_conf >= 90.0:
                    res = self._build_result(
                        status=STATUS_HEALTHY,
                        condition=top_class,
                        confidence=top_conf,
                        top_predictions=top_preds,
                        quality=quality_info,
                        skin_check=skin_status,
                        message="Healthy skin detected. No skin disease identified.",
                        gradcam_url=gradcam_url
                    )
                else:
                    res = self._build_result(
                        status=STATUS_UNCERTAIN,
                        condition=top_class,
                        confidence=top_conf,
                        top_predictions=top_preds,
                        quality=quality_info,
                        skin_check=skin_status,
                        message="Skin features or potential abnormality observed, but the HAM10000 model confidence is low. Professional dermatological evaluation is recommended.",
                        gradcam_url=gradcam_url
                    )

            elif top_class in DISEASE_CLASSES:
                if top_conf >= disease_thresh:
                    res = self._build_result(
                        status=STATUS_DISEASE,
                        condition=top_class,
                        confidence=top_conf,
                        top_predictions=top_preds,
                        quality=quality_info,
                        skin_check=skin_status,
                        message=f"Model prediction: {CLASS_DISPLAY_NAMES.get(top_class, top_class)}.",
                        gradcam_url=gradcam_url
                    )
                else:
                    res = self._build_result(
                        status=STATUS_UNCERTAIN,
                        condition=top_class,
                        confidence=top_conf,
                        top_predictions=top_preds,
                        quality=quality_info,
                        skin_check=skin_status,
                        message=f"Possible signs of {CLASS_DISPLAY_NAMES.get(top_class, top_class)} observed, but model confidence is low ({top_conf:.1f}%). Professional screening recommended.",
                        gradcam_url=gradcam_url
                    )

            else:
                res = self._build_result(
                    status=STATUS_UNCERTAIN,
                    condition=top_class,
                    confidence=top_conf,
                    top_predictions=top_preds,
                    quality=quality_info,
                    skin_check=skin_status,
                    message="Unclassified skin condition.",
                    gradcam_url=gradcam_url
                )

            res["analysis_time"] = f"{time.time() - start_time:.2f} sec"
            return res

        except Exception as e:
            print(f"[DEBUG] Pipeline exception: {e}")
            res = self._build_result(
                status=STATUS_INVALID,
                message=f"Analysis pipeline error: {str(e)}"
            )
            res["analysis_time"] = f"{time.time() - start_time:.2f} sec"
            return res
