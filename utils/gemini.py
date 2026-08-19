import os
import base64
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(BASE_DIR, ".env"))

try:
    from google import genai
except ImportError:
    genai = None


def get_api_key():
    return os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""


GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
GEMINI_FALLBACK_MODELS = [
    GEMINI_MODEL,
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-flash-lite-latest",
    "gemini-3.7-flash",
    "gemini-flash-latest"
]


def gemini_available():
    return (
        genai is not None
        and bool(get_api_key())
    )


_CACHED_WORKING_MODEL = None


def find_working_model():
    """
    Tries each model in the fallback list to find one that works.
    Returns the name of the first working model and caches it.
    """
    global _CACHED_WORKING_MODEL
    if _CACHED_WORKING_MODEL:
        return _CACHED_WORKING_MODEL

    if not gemini_available():
        return None

    try:
        client = genai.Client(api_key=get_api_key())

        seen = set()
        models_to_test = [m for m in GEMINI_FALLBACK_MODELS if m and not (m in seen or seen.add(m))]

        for model_name in models_to_test:
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents="Hello"
                )
                if response and getattr(response, "text", None):
                    _CACHED_WORKING_MODEL = model_name
                    print(f"[Gemini] Active model resolved: {_CACHED_WORKING_MODEL}")
                    return model_name
            except Exception as e:
                print(f"[Gemini notice] Model {model_name} failed: {e}")
                continue
    except Exception as init_err:
        print(f"[Gemini ERROR] Client init error: {init_err}")

    return None


def assess_skin_image_structured(image_path):
    """
    Structured Gemini visual assessment for human skin validation and appearance (normal vs abnormal).
    Returns dict with keys: is_human_skin (bool), subject (str), appearance (str), has_lesion (bool), reason (str).
    """
    default_res = {
        "is_human_skin": True,
        "subject": "human_skin",
        "appearance": "unknown",
        "has_lesion": False,
        "reason": "Default open assessment."
    }

    if not gemini_available():
        return default_res

    try:
        client = genai.Client(api_key=get_api_key())
        with open(image_path, "rb") as f:
            image_data = f.read()

        prompt = (
            "Analyze this dermatological/medical skin photograph. Return ONLY valid JSON with keys:\n"
            "1. \"is_human_skin\": true/false\n"
            "2. \"subject\": one of [\"human_skin\", \"injury\", \"animal\", \"plant\", \"food\", \"object\", \"non_skin\"]\n"
            "3. \"appearance\": \"normal\" or \"abnormal\"\n"
            "4. \"has_lesion\": true if any mole, rash, macule, papule, patch, scale, or abnormal pigmented spot is visible, else false\n"
            "5. \"reason\": short 1-sentence visual explanation\n\n"
            "Note: Moles, lesions, rashes, close-ups, dermoscopy images with black vignetting frames ARE ALL VALID HUMAN SKIN (\"human_skin\")."
        )

        try:
            from google.genai import types
            image_part = types.Part.from_bytes(data=image_data, mime_type="image/jpeg")
            contents = [prompt, image_part]
        except Exception:
            contents = [prompt, {"mime_type": "image/jpeg", "data": base64.b64encode(image_data).decode("utf-8")}]

        working_model = find_working_model() or GEMINI_MODEL
        response = client.models.generate_content(model=working_model, contents=contents)

        if response and response.text:
            raw_text = response.text.strip()
            if "{" in raw_text and "}" in raw_text:
                import json
                json_str = raw_text[raw_text.find("{"):raw_text.rfind("}") + 1]
                parsed = json.loads(json_str)
                return {
                    "is_human_skin": bool(parsed.get("is_human_skin", True)),
                    "subject": str(parsed.get("subject", "human_skin")).lower(),
                    "appearance": str(parsed.get("appearance", "unknown")).lower(),
                    "has_lesion": bool(parsed.get("has_lesion", False)),
                    "reason": str(parsed.get("reason", "Verified by AI."))
                }
    except Exception as e:
        print("[GEMINI] Structured assessment notice:", e)

    return default_res


def check_skin_image(image_path):
    """
    Sends image to Gemini to classify whether it is human_skin, animal, plant, object, food, non_skin, or injury.
    Robustly understands dermatological / skin lesion photography.
    """
    res = assess_skin_image_structured(image_path)
    if not res["is_human_skin"]:
        return res["subject"]
    return "human_skin"


def ask_gemini(message, result=None, history=None):

    if not gemini_available():
        return None

    try:
        client = genai.Client(
            api_key=get_api_key()
        )

        context = ""

        if result:
            if isinstance(result, str):
                context = f"Context: {result}\n"
            elif isinstance(result, dict):
                context = f"""
Current SkinAI project result:
Disease/Class: {result.get('name') or result.get('condition_display', 'N/A')}
Confidence: {result.get('confidence', 'N/A')}%
Risk: {result.get('risk') or result.get('risk_level', 'N/A')}
Stage/Severity Indicator: {result.get('stage', 'N/A')}

Important:
- This is an educational AI project.
- Do not claim that the prediction is a confirmed medical diagnosis.
- Do not invent a disease or stage.
"""

        previous_messages = ""

        if history:
            for item in history[-8:]:
                previous_messages += (
                    f"{item.get('role', 'user')}: "
                    f"{item.get('message', '')}\n"
                )

        prompt = f"""
You are Gemini AI Medical Assistant inside a final-year
B.Tech project called Skin Disease Classification &
Stage Identification.

{context}

Previous conversation:
{previous_messages}

User message:
{message}

Rules:
1. Give clear, simple answers.
2. You may explain skin-disease concepts and the project result.
3. Never present an AI prediction as a confirmed diagnosis.
4. Never claim a cancer stage is medically confirmed.
5. For serious symptoms such as bleeding, rapidly changing
   lesions, severe pain or non-healing wounds, recommend
   professional dermatological evaluation.
6. Do not diagnose unrelated non-skin conditions.
7. If the user asks something unrelated to skin health or
   this project, politely say you are focused on skin-health
   information.
8. Keep the answer concise and student/user friendly.
"""

        seen = set()
        models_to_try = [m for m in GEMINI_FALLBACK_MODELS if m and not (m in seen or seen.add(m))]

        last_error = None
        for model_name in models_to_try:
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt
                )

                answer = getattr(response, "text", None)
                if answer:
                    global _CACHED_WORKING_MODEL
                    _CACHED_WORKING_MODEL = model_name
                    return answer.strip()
            except Exception as model_err:
                last_error = model_err
                print(f"[Gemini notice] Model '{model_name}' failed: {model_err}")
                continue

        if last_error:
            print(f"[GEMINI ERROR] All fallback models failed. Last error: {last_error}")

        return None

    except Exception as error:
        print(f"[GEMINI ERROR] Chat error: {error}")
        return None