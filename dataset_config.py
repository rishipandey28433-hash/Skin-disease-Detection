"""
Dataset Configuration and Class Taxonomy
Contains class definitions, names, descriptions, risk levels, and helper functions
for the Skin Disease Classification System.
"""

# ==========================================
# CONSTANTS & CLASS IDENTIFIERS
# ==========================================

# Sorted list of all class identifiers matching trained model (8 classes)
CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc", "healthy"]
NUM_CLASSES = len(CLASS_NAMES)

# Human-readable names
CLASS_DISPLAY_NAMES = {
    "akiec": "Actinic Keratoses",
    "bcc": "Basal Cell Carcinoma",
    "bkl": "Benign Keratosis",
    "df": "Dermatofibroma",
    "healthy": "Healthy Skin",
    "mel": "Melanoma",
    "nv": "Melanocytic Nevus",
    "vasc": "Vascular Lesion",
    "vitiligo": "Vitiligo"
}

# ==========================================
# DISEASE INFORMATION
# ==========================================

# Short factual medical descriptions
CLASS_DESCRIPTIONS = {
    "akiec": "Precancerous skin lesion caused by chronic sun exposure, which can develop into squamous cell carcinoma.",
    "bcc": "A common type of skin cancer that begins in the basal cells, typically appearing as a transparent bump on the skin.",
    "bkl": "A non-cancerous skin growth that can appear in various colors, ranging from light tan to black (includes seborrheic keratosis).",
    "df": "A benign, slow-growing skin nodule typically found on the lower legs.",
    "healthy": "Normal, healthy skin without signs of significant lesions or diseases.",
    "mel": "The most serious type of skin cancer, develops in the cells (melanocytes) that produce melanin.",
    "nv": "A common benign skin lesion often called a mole, formed by a cluster of melanocytes.",
    "vasc": "A condition involving abnormalities in blood vessels of the skin, such as cherry angiomas or angiokeratomas.",
    "vitiligo": "An autoimmune disorder causing loss of skin color in blotches due to the destruction of pigment-producing cells."
}

# Common symptoms
CLASS_SYMPTOMS = {
    "akiec": "Rough, scaly patches on sun-exposed skin, sometimes itchy or burning.",
    "bcc": "Pearly or waxy bump, flat flesh-colored or brown scar-like lesion, or a bleeding/scabbing sore that heals and returns.",
    "bkl": "Waxy, scaly, slightly elevated appearance, often looking like it was pasted onto the skin.",
    "df": "Firm, small bump (usually pink, red, or brown) that may feel like a hard lump under the skin.",
    "healthy": "None.",
    "mel": "A new, unusual growth or a change in an existing mole (ABCDE rules: Asymmetry, Border, Color, Diameter, Evolving).",
    "nv": "Uniform in color (brown, black, or tan), round or oval, usually smaller than a pencil eraser.",
    "vasc": "Red, purple, or blue marks on the skin, sometimes elevated or bleeding if scratched.",
    "vitiligo": "Patchy loss of skin color, premature whitening or graying of hair on the scalp, eyelashes, eyebrows, or beard."
}

# Recommended precautions
CLASS_PRECAUTIONS = {
    "akiec": "Strict sun protection, regular dermatological check-ups to monitor for progression to squamous cell carcinoma.",
    "bcc": "Avoid UV exposure, use broad-spectrum sunscreen, wear protective clothing, and seek prompt treatment.",
    "bkl": "Generally no treatment needed unless irritated by clothing or for cosmetic reasons. Sun protection is recommended.",
    "df": "Typically harmless and requires no treatment, but should be monitored for changes.",
    "healthy": "Maintain routine sun protection, daily moisturizing, and periodic self-examinations.",
    "mel": "Immediate dermatological evaluation. Strict sun avoidance and regular total-body skin exams.",
    "nv": "Monitor for changes using the ABCDE rule. Protect from excessive sun exposure.",
    "vasc": "Usually harmless, but avoid scratching to prevent bleeding. Consult a doctor if they change rapidly.",
    "vitiligo": "Protect depigmented areas from the sun using high SPF sunscreen, as they burn easily."
}

# ==========================================
# RISK ASSESSMENT & CATEGORIZATION
# ==========================================

# Risk level categorization
CLASS_RISK_LEVEL = {
    "akiec": "moderate",
    "bcc": "high",
    "bkl": "low",
    "df": "low",
    "healthy": "none",
    "mel": "high",
    "nv": "low",
    "vasc": "low",
    "vitiligo": "low"
}

# Medical categorization
CLASS_CATEGORY = {
    "akiec": "malignant_neoplasm",
    "bcc": "malignant_neoplasm",
    "bkl": "benign_neoplasm",
    "df": "benign_neoplasm",
    "healthy": "healthy",
    "mel": "malignant_neoplasm",
    "nv": "benign_neoplasm",
    "vasc": "vascular",
    "vitiligo": "pigmentation_disorder"
}

CATEGORY_DISPLAY_NAMES = {
    "malignant_neoplasm": "Malignant Neoplasms",
    "benign_neoplasm": "Benign Neoplasms",
    "vascular": "Vascular Conditions",
    "pigmentation_disorder": "Pigmentation Disorders",
    "healthy": "Healthy Skin"
}

# Hierarchical representation
CLASS_HIERARCHY = {
    "Human Skin": {
        "Healthy Skin": ["healthy"],
        "Skin Conditions": {
            "Malignant Neoplasms": ["mel", "bcc", "akiec"],
            "Benign Neoplasms": ["nv", "bkl", "df"],
            "Vascular Conditions": ["vasc"],
            "Pigmentation Disorders": ["vitiligo"]
        }
    }
}

# ==========================================
# STATUS CONSTANTS & GROUPINGS
# ==========================================

STATUS_HEALTHY = "healthy"
STATUS_DISEASE = "disease"
STATUS_NON_DISEASE = "non_disease_condition"
STATUS_INVALID = "invalid_input"
STATUS_LOW_QUALITY = "low_quality"
STATUS_UNCERTAIN = "uncertain"

HEALTHY_CLASSES = {"healthy"}
DISEASE_CLASSES = set(CLASS_NAMES) - HEALTHY_CLASSES

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def get_disease_info(class_name):
    """
    Retrieve all available information for a given disease class.
    
    Args:
        class_name (str): The identifier of the class (e.g., 'mel', 'nv').
        
    Returns:
        dict: A dictionary containing the class details.
    """
    if class_name not in CLASS_NAMES:
        return {}
    
    return {
        "id": class_name,
        "name": CLASS_DISPLAY_NAMES.get(class_name, class_name),
        "description": CLASS_DESCRIPTIONS.get(class_name, ""),
        "symptoms": CLASS_SYMPTOMS.get(class_name, ""),
        "precautions": CLASS_PRECAUTIONS.get(class_name, ""),
        "risk_level": CLASS_RISK_LEVEL.get(class_name, "unknown"),
        "category": CATEGORY_DISPLAY_NAMES.get(CLASS_CATEGORY.get(class_name, ""), "Unknown")
    }

def get_risk_percentage(class_name, confidence):
    """
    Calculate a UI-friendly risk percentage based on the class and confidence.
    
    Args:
        class_name (str): The identifier of the class.
        confidence (float): The prediction confidence (0 to 1 or 0 to 100).
        
    Returns:
        float: Calculated risk percentage capped appropriately.
    """
    # Normalize confidence to 0-100 range if it is 0-1
    if confidence <= 1.0:
        confidence = confidence * 100
        
    if class_name == "healthy":
        return 0.0
        
    risk_level = CLASS_RISK_LEVEL.get(class_name, "low")
    
    if risk_level == "high":
        # Base 70, up to +25 from confidence
        risk = 70.0 + (confidence * 0.25)
        return min(risk, 95.0)
    elif risk_level == "moderate":
        # Base 45, up to +20 from confidence
        risk = 45.0 + (confidence * 0.20)
        return min(risk, 65.0)
    else: # low risk
        # Base 30, up to +5 from confidence
        risk = 30.0 + (confidence * 0.05)
        return min(risk, 35.0)
