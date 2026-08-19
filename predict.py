# ============================================================
# SKIN DISEASE PREDICTION MODULE
# ============================================================
#
# FINAL YEAR PROJECT
#
# Classes:
#   akiec
#   bcc
#   bkl
#   df
#   mel
#   nv
#   vasc
#   healthy
#
# IMPORTANT:
# ------------------------------------------------------------
# This module assumes that app.py / validator has already
# checked whether the uploaded image is HUMAN SKIN.
#
# Therefore:
#
#   Animal / Plant / Object / Random Image
#       -> handled by validator
#
#   Human Skin
#       -> disease / healthy classification here
#
# A clear human skin image must NOT become "not_detected"
# merely because the disease classifier is uncertain.
# If disease evidence is weak, it is treated as HEALTHY.
# ============================================================


import os
import json
import numpy as np
import tensorflow as tf

from config import MODEL_PATH
from utils.preprocess import preprocess_image


# ============================================================
# BASE DIRECTORY
# ============================================================

BASE_DIR = os.path.abspath(
    os.path.dirname(__file__)
)


# ============================================================
# CLASS NAMES
# ============================================================

CLASS_NAMES_PATH = os.path.join(
    BASE_DIR,
    "models",
    "class_names.json"
)


if not os.path.exists(CLASS_NAMES_PATH):

    raise FileNotFoundError(
        f"class_names.json not found: {CLASS_NAMES_PATH}"
    )


with open(
    CLASS_NAMES_PATH,
    "r",
    encoding="utf-8"
) as file:

    class_names = json.load(file)


if not isinstance(class_names, list):

    raise ValueError(
        "class_names.json must contain a JSON list."
    )


if len(class_names) == 0:

    raise ValueError(
        "class_names.json is empty."
    )


# ============================================================
# LOAD MODEL
# ============================================================

if not os.path.exists(MODEL_PATH):

    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )


model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)


# ============================================================
# EXPECTED CLASSES
# ============================================================

EXPECTED_CLASSES = [
    "akiec",
    "bcc",
    "bkl",
    "df",
    "mel",
    "nv",
    "vasc",
    "healthy"
]


# ============================================================
# FIND HEALTHY CLASS
# ============================================================

HEALTHY_INDEX = None

for index, name in enumerate(class_names):

    if str(name).strip().lower() == "healthy":

        HEALTHY_INDEX = index
        break


if HEALTHY_INDEX is None:

    raise ValueError(
        "Healthy class was not found in class_names.json."
    )


# ============================================================
# DISEASE CLASSES
# ============================================================

DISEASE_CLASSES = [
    name
    for name in class_names
    if str(name).strip().lower() != "healthy"
]


# ============================================================
# SETTINGS
# ============================================================
#
# These values control the final decision.
#
# IMPORTANT:
# We do NOT use the old:
#
#     confidence < 55
#     margin < 12
#
# gate for healthy human skin.
#
# That old gate was the reason a valid clear-skin image could
# become "not_detected".
#
# Disease result is accepted only when there is reasonably
# strong disease evidence.
#
# Otherwise, because validator has already established that the
# image is human skin, it is classified as Healthy Skin.
# ============================================================

DISEASE_MIN_CONFIDENCE = 55.0
DISEASE_MIN_MARGIN = 12.0

# If the healthy class itself is reasonably strong, prefer it.
HEALTHY_DIRECT_THRESHOLD = 45.0

# When no disease class has strong enough evidence, the human
# skin image falls back to healthy.
HEALTHY_FALLBACK = True


# ============================================================
# HELPER: CONVERT MODEL OUTPUT TO PROBABILITIES
# ============================================================

def _prepare_probabilities(raw_prediction):

    prediction = np.asarray(
        raw_prediction,
        dtype=np.float32
    )

    # --------------------------------------------------------
    # Normalize shape
    # --------------------------------------------------------

    if prediction.ndim == 2:

        prediction = prediction[0]

    else:

        prediction = np.squeeze(
            prediction
        )


    if prediction.ndim != 1:

        raise ValueError(
            f"Unexpected prediction shape: {prediction.shape}"
        )


    # --------------------------------------------------------
    # Remove invalid values
    # --------------------------------------------------------

    prediction = np.nan_to_num(
        prediction,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )


    # --------------------------------------------------------
    # Check probability / logits
    # --------------------------------------------------------

    prediction_sum = float(
        np.sum(prediction)
    )


    if (
        np.any(prediction < 0)
        or
        not np.isclose(
            prediction_sum,
            1.0,
            atol=0.05
        )
    ):

        prediction = tf.nn.softmax(
            prediction
        ).numpy()


    # --------------------------------------------------------
    # Final normalization
    # --------------------------------------------------------

    prediction = np.nan_to_num(
        prediction,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )


    total = float(
        np.sum(prediction)
    )


    if total <= 0:

        raise ValueError(
            "Invalid model prediction."
        )


    prediction = (
        prediction / total
    )


    return prediction


# ============================================================
# HELPER: TOP PREDICTIONS
# ============================================================

def _get_top_predictions(
    prediction,
    limit=3
):

    sorted_indexes = np.argsort(
        prediction
    )[::-1]


    top_predictions = []


    for rank, index in enumerate(
        sorted_indexes[:limit],
        start=1
    ):

        index = int(index)


        top_predictions.append({

            "rank":
                rank,

            "class":
                class_names[index],

            "confidence":
                round(
                    float(
                        prediction[index] * 100
                    ),
                    2
                )
        })


    return (
        sorted_indexes,
        top_predictions
    )


# ============================================================
# PREDICT
# ============================================================

def predict(image_path):

    # ========================================================
    # FILE CHECK
    # ========================================================

    if not os.path.exists(image_path):

        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )


    # ========================================================
    # PREPROCESS
    # ========================================================

    image = preprocess_image(
        image_path
    )


    # ========================================================
    # MODEL PREDICTION
    # ========================================================

    raw_prediction = model.predict(
        image,
        verbose=0
    )


    # ========================================================
    # PROBABILITIES
    # ========================================================

    prediction = _prepare_probabilities(
        raw_prediction
    )


    # ========================================================
    # CHECK CLASS COUNT
    # ========================================================

    if len(prediction) != len(class_names):

        raise ValueError(
            "Model output count does not match "
            f"class_names.json. "
            f"Model output: {len(prediction)}, "
            f"classes: {len(class_names)}"
        )


    # ========================================================
    # TOP PREDICTION
    # ========================================================

    sorted_indexes, top_predictions = (
        _get_top_predictions(
            prediction,
            limit=3
        )
    )


    best_index = int(
        sorted_indexes[0]
    )


    second_index = (
        int(sorted_indexes[1])
        if len(sorted_indexes) > 1
        else best_index
    )


    best_class = str(
        class_names[best_index]
    ).strip().lower()


    confidence = float(
        prediction[best_index] * 100
    )


    second_confidence = float(
        prediction[second_index] * 100
    )


    margin = (
        confidence
        -
        second_confidence
    )


    # ========================================================
    # HEALTHY PROBABILITY
    # ========================================================

    healthy_confidence = float(
        prediction[HEALTHY_INDEX] * 100
    )


    # ========================================================
    # DISEASE BEST PREDICTION
    # ========================================================
    #
    # We separately calculate the strongest disease class.
    #
    # This is important because "healthy" must be treated
    # separately from disease confidence.
    # ========================================================

    disease_indexes = [
        index
        for index, name in enumerate(class_names)
        if str(name).strip().lower() != "healthy"
    ]


    disease_best_index = None
    disease_confidence = 0.0
    disease_class = None


    if disease_indexes:

        disease_best_index = max(
            disease_indexes,
            key=lambda idx: prediction[idx]
        )


        disease_confidence = float(
            prediction[disease_best_index] * 100
        )


        disease_class = str(
            class_names[disease_best_index]
        )


    # ========================================================
    # DISEASE DECISION
    # ========================================================
    #
    # CASE 1:
    # Healthy probability is clearly strong.
    #
    # Example:
    #
    # healthy = 98%
    # nv      = 0.5%
    #
    # -> HEALTHY
    # ========================================================

    if (
        healthy_confidence
        >=
        HEALTHY_DIRECT_THRESHOLD
    ):

        return {

            "class":
                "healthy",

            "confidence":
                round(
                    healthy_confidence,
                    2
                ),

            "healthy_confidence":
                round(
                    healthy_confidence,
                    2
                ),

            "best_class":
                "healthy",

            "disease_class":
                disease_class,

            "disease_confidence":
                round(
                    disease_confidence,
                    2
                ),

            "margin":
                round(
                    margin,
                    2
                ),

            "all_predictions":
                prediction.tolist(),

            "top_predictions":
                top_predictions
        }


    # ========================================================
    # CASE 2:
    # Disease model has strong evidence.
    #
    # Disease confidence must satisfy BOTH:
    #
    #   confidence >= 55%
    #   margin >= 12%
    #
    # This prevents weak/random disease predictions from
    # appearing on otherwise normal human skin.
    # ========================================================

    disease_margin = (
        disease_confidence
        -
        healthy_confidence
    )


    strong_disease = (

        disease_confidence
        >=
        DISEASE_MIN_CONFIDENCE

        and

        disease_margin
        >=
        DISEASE_MIN_MARGIN
    )


    if strong_disease:

        return {

            "class":
                disease_class,

            "confidence":
                round(
                    disease_confidence,
                    2
                ),

            "healthy_confidence":
                round(
                    healthy_confidence,
                    2
                ),

            "best_class":
                disease_class,

            "disease_class":
                disease_class,

            "disease_confidence":
                round(
                    disease_confidence,
                    2
                ),

            "margin":
                round(
                    disease_margin,
                    2
                ),

            "all_predictions":
                prediction.tolist(),

            "top_predictions":
                top_predictions
        }


    # ========================================================
    # CASE 3:
    # HUMAN SKIN BUT NO STRONG DISEASE EVIDENCE
    # ========================================================
    #
    # THIS IS THE IMPORTANT FIX.
    #
    # Validator has already established:
    #
    #       image = HUMAN SKIN
    #
    # If disease model does not have strong enough evidence for
    # a disease, we should NOT return "not_detected".
    #
    # Instead:
    #
    #       HEALTHY SKIN
    #
    # This handles clear:
    #
    #   face
    #   hand
    #   arm
    #   leg
    #   normal skin
    #   skin without redness
    #   skin without rash
    #   skin without swelling
    #   skin without visible lesion
    #
    # ========================================================

    if HEALTHY_FALLBACK:

        # ----------------------------------------------------
        # Use the model's strongest available healthy evidence.
        #
        # If healthy probability is low because the disease
        # classifier is imperfect, we still return healthy
        # rather than "not_detected", because the validator has
        # already accepted the image as human skin.
        #
        # We DO NOT invent a probability here.
        # The actual model probability is returned.
        # ----------------------------------------------------

        return {

            "class":
                "healthy",

            "confidence":
                round(
                    healthy_confidence,
                    2
                ),

            "healthy_confidence":
                round(
                    healthy_confidence,
                    2
                ),

            "best_class":
                "healthy",

            "disease_class":
                disease_class,

            "disease_confidence":
                round(
                    disease_confidence,
                    2
                ),

            "margin":
                round(
                    disease_margin,
                    2
                ),

            "all_predictions":
                prediction.tolist(),

            "top_predictions":
                top_predictions
        }


    # ========================================================
    # SAFETY FALLBACK
    # ========================================================

    return {

        "class":
            "not_detected",

        "confidence":
            round(
                confidence,
                2
            ),

        "healthy_confidence":
            round(
                healthy_confidence,
                2
            ),

        "best_class":
            best_class,

        "disease_class":
            disease_class,

        "disease_confidence":
            round(
                disease_confidence,
                2
            ),

        "margin":
            round(
                margin,
                2
            ),

        "all_predictions":
            prediction.tolist(),

        "top_predictions":
            top_predictions
    }