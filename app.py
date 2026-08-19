# ============================================================
# SKIN DISEASE CLASSIFICATION & STAGE IDENTIFICATION
# FINAL YEAR PROJECT
# ============================================================

# --- Imports ---
from flask import Flask, render_template, request, jsonify, send_file, url_for, redirect, session
import tensorflow as tf
import numpy as np
import os, uuid, json, sqlite3, time
from PIL import Image
from datetime import datetime
from functools import wraps
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from dotenv import load_dotenv

from config import MODEL_PATH, UPLOAD_FOLDER, IMAGE_SIZE, ALLOWED_EXTENSIONS, CONFIDENCE_THRESHOLD, HEALTHY_THRESHOLD
from dataset_config import (
    CLASS_NAMES, CLASS_DISPLAY_NAMES, CLASS_DESCRIPTIONS,
    CLASS_SYMPTOMS, CLASS_PRECAUTIONS, CLASS_RISK_LEVEL,
    CLASS_CATEGORY, CATEGORY_DISPLAY_NAMES, NUM_CLASSES,
    STATUS_HEALTHY, STATUS_DISEASE, STATUS_INVALID,
    STATUS_LOW_QUALITY, STATUS_UNCERTAIN, STATUS_NON_DISEASE,
    get_disease_info, get_risk_percentage
)
from pipeline import SkinAnalysisPipeline
from utils.gemini import ask_gemini, gemini_available

# ============================================================
# LOAD .ENV
# ============================================================
load_dotenv()

# ============================================================
# FLASK APP SETUP
# ============================================================
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "skin-ai-local-secret-key-change-this")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATABASE_PATH = os.path.join(BASE_DIR, "skin_ai_users.db")

# ============================================================
# LOAD TF MODEL & PIPELINE
# ============================================================
print("=" * 70)
print("SKIN AI - STARTING")
print("=" * 70)

if not os.path.exists(MODEL_PATH):
    print(f"\nModel file not found at {MODEL_PATH}. Initializing architecture from model.py...")
    from model import build_model
    init_model, _ = build_model(num_classes=len(CLASS_NAMES))
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    init_model.save(MODEL_PATH)
    print("Model initialized and saved successfully.")

try:
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print("\nTensorFlow Model Loaded Successfully")
    pipeline_instance = SkinAnalysisPipeline(model, CLASS_NAMES)
except Exception as e:
    raise RuntimeError("Model loading failed: " + str(e))

# ============================================================
# DATABASE
# ============================================================
def init_database():
    connection = sqlite3.connect(DATABASE_PATH)
    cursor = connection.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    connection.commit()
    connection.close()

def create_default_user():
    connection = sqlite3.connect(DATABASE_PATH)
    cursor = connection.cursor()
    cursor.execute("SELECT id FROM users WHERE username = ?", ("admin",))
    existing = cursor.fetchone()
    if not existing:
        password_hash = generate_password_hash("admin123")
        cursor.execute(
            """
            INSERT INTO users (username, password, created_at)
            VALUES (?, ?, ?)
            """,
            ("admin", password_hash, datetime.now().isoformat())
        )
        connection.commit()
    connection.close()

init_database()
create_default_user()

# ============================================================
# LOGIN REQUIRED
# ============================================================
def login_required(function):
    @wraps(function)
    def decorated_function(*args, **kwargs):
        if not session.get("logged_in"):
            if (request.path.startswith("/predict") or 
                request.path.startswith("/chat") or 
                request.path.startswith("/download_report") or 
                request.path.startswith("/uploads")):
                return jsonify({"success": False, "error": "Please login first.", "login_required": True}), 401
            return redirect(url_for("login"))
        return function(*args, **kwargs)
    return decorated_function

# ============================================================
# FILE VALIDATION
# ============================================================
def check_file(filename):
    if not filename or "." not in filename:
        return False
    extension = filename.rsplit(".", 1)[1].lower()
    allowed = {str(ext).lower() for ext in ALLOWED_EXTENSIONS}
    allowed.update({"webp"})
    return extension in allowed

# ============================================================
# ROUTES
# ============================================================

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        
        connection = sqlite3.connect(DATABASE_PATH)
        cursor = connection.cursor()
        cursor.execute("SELECT id, password FROM users WHERE username = ?", (username,))
        user = cursor.fetchone()
        connection.close()
        
        if user and check_password_hash(user[1], password):
            session["logged_in"] = True
            session["username"] = username
            return redirect(url_for("index"))
        else:
            return render_template("login.html", error="Invalid credentials")
            
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

@app.route("/", methods=["GET"])
@app.route("/home", methods=["GET"])
@login_required
def index():
    return render_template("index.html", username=session.get("username", ""))

def home():
    return index()


@app.route("/predict", methods=["POST"])
@login_required
def predict():
    print("=" * 60)
    print("[DEBUG] REQUEST METHOD:", request.method)
    print("[DEBUG] REQUEST FILES:", request.files)
    print("[DEBUG] REQUEST FORM:", request.form)
    print("=" * 60)

    # Check both 'file' and 'image' input names from HTML
    file = request.files.get("file") or request.files.get("image")

    print("[DEBUG] UPLOADED FILE:", file)
    print("[DEBUG] FILENAME:", file.filename if file else None)

    start_time = time.time()
    report_id = f"SKIN-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}"

    if not file or file.filename == "":
        print("[DEBUG] REJECTED: No file uploaded or empty filename.")
        return render_template(
            "result.html",
            report_id=report_id,
            analysis_time="0.00 sec",
            status_display="Invalid Image",
            status=STATUS_INVALID,
            message="No file uploaded",
            result={"report_id": report_id, "analysis_time": "0.00 sec", "status_display": "Invalid Image"},
            username=session.get("username", "")
        )

    if not check_file(file.filename):
        print(f"[DEBUG] REJECTED: Invalid file type ({file.filename}).")
        return render_template(
            "result.html",
            report_id=report_id,
            analysis_time="0.00 sec",
            status_display="Invalid Image",
            status=STATUS_INVALID,
            message="Invalid file type",
            result={"report_id": report_id, "analysis_time": "0.00 sec", "status_display": "Invalid Image"},
            username=session.get("username", "")
        )

    filename = secure_filename(file.filename)
    unique_filename = f"{uuid.uuid4().hex}_{filename}"
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], unique_filename)
    file.save(file_path)

    print("[DEBUG] SAVE PATH:", file_path)
    print("[DEBUG] FILE EXISTS:", os.path.exists(file_path))

    try:
        # Call pipeline with upload_folder for Grad-CAM overlay generation
        result = pipeline_instance.analyze(file_path, upload_folder=app.config["UPLOAD_FOLDER"])
        
        end_time = time.time()
        analysis_duration = end_time - start_time
        analysis_time_str = result.get("analysis_time", f"{analysis_duration:.2f} sec")
        
        # Attach meta to result dictionary
        result["report_id"] = report_id
        result["analysis_time"] = analysis_time_str
        
        res_status = result.get("status")
        if res_status in [STATUS_DISEASE, STATUS_HEALTHY, STATUS_UNCERTAIN, STATUS_NON_DISEASE]:
            status_display = "Analysis Complete"
        elif res_status == STATUS_INVALID:
            status_display = "Invalid Image"
        else:
            status_display = "Analysis Failed"
            
        result["status_display"] = status_display

        return render_template(
            "result.html",
            result=result,
            # Top summary meta fields
            report_id=report_id,
            analysis_time=analysis_time_str,
            status_display=status_display,
            # Status
            status=result.get("status"),
            # Condition info
            condition=result.get("condition", ""),
            condition_display=result.get("condition_display", ""),
            disease=result.get("condition_display", ""),  # backward compat
            disease_name=result.get("condition_display", ""),  # backward compat
            stage=result.get("stage", "Not Available"),
            # Confidence and risk
            confidence=result.get("confidence", 0),
            risk_level=result.get("risk_level", "none"),
            risk_percentage=result.get("risk_percentage", 0),
            risk=result.get("risk_level", "none"),  # backward compat
            # Category
            category=result.get("category", ""),
            # Description info
            description=result.get("description", ""),
            symptoms=result.get("symptoms", ""),
            precautions=result.get("precautions", ""),
            # Predictions & Explainability
            top_predictions=result.get("top_predictions", []),
            gradcam_url=result.get("gradcam_url"),
            # Quality
            quality=result.get("quality", {}),
            # Messages
            message=result.get("message", ""),
            medical_disclaimer=result.get("medical_disclaimer", ""),
            # Image
            image_url=url_for('static', filename='uploads/' + unique_filename),
            filename=unique_filename,
            # User
            username=session.get("username", "")
        )
    except Exception as e:
        end_time = time.time()
        analysis_duration = end_time - start_time
        analysis_time_str = f"{analysis_duration:.2f} sec"
        return render_template(
            "result.html",
            report_id=report_id,
            analysis_time=analysis_time_str,
            status_display="Analysis Failed",
            status=STATUS_INVALID,
            message=f"Error analyzing image: {str(e)}",
            result={"report_id": report_id, "analysis_time": analysis_time_str, "status_display": "Analysis Failed"},
            username=session.get("username", "")
        )

@app.route("/chat", methods=["POST"])
@login_required
def chat():
    data = request.get_json()
    if not data or "message" not in data:
        return jsonify({"success": False, "error": "No message provided."}), 400
        
    user_message = data["message"]
    print(f"[CHAT] User message received: {user_message}")
    
    # Support both "context" and individual result fields (disease, risk, confidence, stage)
    context = data.get("context")
    if not context:
        disease_name = data.get("disease") or data.get("name") or data.get("condition_display")
        if disease_name:
            context = {
                "name": disease_name,
                "risk": data.get("risk") or data.get("risk_level", ""),
                "confidence": data.get("confidence", 0),
                "stage": data.get("stage", "")
            }
        else:
            context = ""
    
    try:
        print("[CHAT] Calling Gemini Assistant...")
        reply = ask_gemini(user_message, context)
        if reply:
            print("[CHAT] Gemini response received successfully.")
            return jsonify({"success": True, "reply": reply})
        else:
            print("[CHAT ERROR] Gemini returned empty response.")
            return jsonify({"success": False, "error": "Gemini could not generate a response at this moment. Please try again."}), 500
    except Exception as e:
        print(f"[CHAT ERROR] {type(e).__name__}: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/download_report", methods=["POST"])
@login_required
def download_report():
    try:
        data = request.json
        if not data:
            return jsonify({"success": False, "error": "No data provided"}), 400
            
        filename = f"report_{uuid.uuid4().hex}.pdf"
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        
        c = canvas.Canvas(filepath, pagesize=letter)
        rpt_id = data.get('report_id')
        if rpt_id:
            c.drawString(100, 750, f"Skin Disease Diagnosis Report — ID: {rpt_id}")
        else:
            c.drawString(100, 750, "Skin Disease Diagnosis Report")
        c.drawString(100, 730, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        condition_name = data.get('disease_name') or data.get('condition_display', 'Unknown')
        c.drawString(100, 710, f"Condition: {condition_name}")
        c.drawString(100, 690, f"Confidence: {data.get('confidence', 0):.2f}%")
        
        risk = data.get('risk_level') or data.get('risk', 'Unknown')
        c.drawString(100, 670, f"Risk Level: {risk.title()}")
        
        y = 640
        for key in ['description', 'symptoms', 'precautions']:
            if data.get(key):
                c.drawString(100, y, f"{key.capitalize()}:")
                y -= 20
                text = data.get(key)
                words = text.split()
                line = ""
                for word in words:
                    if len(line) + len(word) > 80:
                        c.drawString(120, y, line)
                        y -= 15
                        line = word + " "
                    else:
                        line += word + " "
                if line:
                    c.drawString(120, y, line)
                    y -= 30
                    
        c.save()
        
        return send_file(filepath, as_attachment=True)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/uploads/<filename>")
@login_required
def uploaded_file(filename):
    return send_file(os.path.join(app.config["UPLOAD_FOLDER"], filename))

@app.route("/health")
def health():
    return jsonify({"status": "ok", "timestamp": datetime.now().isoformat()})

@app.errorhandler(404)
def page_not_found(e):
    return render_template("404.html"), 404

@app.errorhandler(413)
def request_entity_too_large(e):
    return render_template("500.html", message="File too large (Max 10MB)"), 413

@app.errorhandler(500)
def internal_server_error(e):
    return render_template("500.html", message="Internal Server Error"), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)