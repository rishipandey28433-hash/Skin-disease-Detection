# Skin Disease Classification & Stage Identification

## Multi-Stage AI-Powered Skin Condition Analysis System

A comprehensive skin condition detection system using EfficientNetB0 deep learning architecture with multi-stage validation pipeline.

> **⚠️ Medical Disclaimer**: This is an educational and research project. It is NOT a medical diagnosis tool. Always consult a qualified dermatologist for professional evaluation.

---

## Features

### Multi-Stage Prediction Pipeline
1. **Image Validation** — Format, dimensions, corruption detection
2. **Image Quality Check** — Blur, brightness, contrast, resolution assessment
3. **Human Skin Detection** — Gemini AI-powered validation (rejects animals, plants, objects)
4. **Disease Classification** — EfficientNetB0 TensorFlow model
5. **Confidence Validation** — Uncertainty/rejection thresholding
6. **Result Categorization** — Healthy / Disease / Non-Disease Condition / Invalid / Uncertain

### Supported Classes (9)
| Class ID | Display Name | Category | Risk Level |
|---|---|---|---|
| `akiec` | Actinic Keratoses | Malignant Neoplasm | Moderate |
| `bcc` | Basal Cell Carcinoma | Malignant Neoplasm | High |
| `bkl` | Benign Keratosis | Benign Neoplasm | Low |
| `df` | Dermatofibroma | Benign Neoplasm | Low |
| `healthy` | Healthy Skin | Healthy | None |
| `mel` | Melanoma | Malignant Neoplasm | High |
| `nv` | Melanocytic Nevus | Benign Neoplasm | Low |
| `vasc` | Vascular Lesion | Vascular | Low |
| `vitiligo` | Vitiligo | Pigmentation Disorder | Low |

### Result Categories
- ✅ **Healthy Skin** — No disease detected
- ⚠️ **Disease Detected** — Condition identified with confidence score
- 🔶 **Non-Disease Condition** — Injury or non-disease skin abnormality
- ❌ **Invalid Input** — Not human skin (animal, plant, object, etc.)
- 📷 **Low Quality** — Image too blurry, dark, or low resolution
- ❓ **Uncertain** — Model cannot classify reliably

---

## Architecture

### Model
- **Base**: EfficientNetB0 (ImageNet pretrained)
- **Input**: 224×224×3 RGB
- **Classification Head**: GAP → BN → Dense(256) → Dropout(0.45) → Dense(128) → Dropout(0.30) → Dense(9, softmax)
- **Training**: 2-stage transfer learning (head training + fine-tuning top 30%)

### Tech Stack
- **Backend**: Flask (Python)
- **ML Framework**: TensorFlow / Keras
- **Skin Validation**: Google Gemini API
- **Database**: SQLite (authentication)
- **Frontend**: Custom HTML/CSS/JS with Chart.js
- **Reports**: ReportLab PDF generation

---

## Project Structure

```
SkinDiseaseDiagnosis/
├── app.py                  # Flask backend (routes, API)
├── pipeline.py             # Multi-stage prediction pipeline
├── train.py                # Model training script
├── model.py                # EfficientNetB0 model builder
├── config.py               # Application configuration
├── dataset_config.py       # Class taxonomy & disease info
├── prepare_dataset.py      # Dataset preparation & splitting
├── predict.py              # Standalone prediction module
├── requirements.txt        # Python dependencies
├── models/
│   ├── skin_model.keras    # Trained model weights
│   └── class_names.json    # Class label mapping
├── dataset/                # Training/validation/test data
│   ├── train/              #   (9 class subdirectories)
│   ├── val/
│   └── test/
├── utils/
│   ├── gemini.py           # Gemini API integration
│   ├── helper.py           # Helper functions
│   ├── image_validator.py  # Image quality assessment
│   └── preprocess.py       # Image preprocessing
├── templates/
│   ├── index.html          # Main page (upload/capture)
│   ├── result.html         # Analysis results dashboard
│   ├── login.html          # Authentication page
│   ├── 404.html            # Error page
│   └── 500.html            # Error page
├── static/
│   ├── css/style.css       # Styles
│   └── js/script.js        # Frontend logic + camera
├── tests/
│   ├── test_pipeline.py    # Pipeline tests
│   └── test_api.py         # API endpoint tests
└── reports/                # Generated training reports
```

---

## Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Environment
Create a `.env` file:
```
GEMINI_API_KEY=your_gemini_api_key_here
```

### 3. Prepare Dataset
```bash
python prepare_dataset.py
```
This creates a properly split dataset with train/val/test directories.

### 4. Train Model
```bash
python train.py
```
Generates model weights, confusion matrix, training curves, and metrics report.

### 5. Run Application
```bash
python app.py
```
Access at `http://localhost:5000`

Default login: `admin` / `admin123`

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Home page (requires login) |
| `POST` | `/login` | User authentication |
| `GET` | `/logout` | Clear session |
| `POST` | `/predict` | Upload image for analysis |
| `POST` | `/chat` | AI medical assistant chat |
| `GET` | `/health` | Server health check |
| `POST` | `/download_report` | Generate PDF report |

### Prediction Response Structure
```json
{
  "status": "disease",
  "condition": "mel",
  "condition_display": "Melanoma",
  "confidence": 93.5,
  "risk_level": "high",
  "risk_percentage": 85.0,
  "category": "Malignant Neoplasm",
  "description": "...",
  "symptoms": "...",
  "precautions": "...",
  "top_predictions": [...],
  "message": "...",
  "medical_disclaimer": "..."
}
```

---

## Training Details

### Data Augmentation
- Rotation: ±25°
- Width/Height shift: 12%
- Shear: 10%
- Zoom: 20%
- Horizontal flip: Yes
- Brightness: [0.85, 1.15]

### Training Strategy
- **Stage 1**: Head training (10 epochs, LR=3e-4)
- **Stage 2**: Fine-tuning top 30% (20 epochs, LR=1e-5)
- **Optimizer**: Adam
- **Loss**: Categorical Crossentropy
- **Class Weights**: Computed balanced weights (sklearn)

### Evaluation Metrics
- Accuracy, Precision, Recall, F1-score (per-class)
- Confusion matrix
- Training/validation curves

---

## Dataset

### Source
- **HAM10000**: 10,015 dermatoscopic images (7 disease classes)
- **Healthy Skin**: Custom healthy skin image collection
- **Vitiligo**: Roboflow dermatological images dataset

### Split Strategy
- **Train**: 70% | **Validation**: 15% | **Test**: 15%
- Patient-level splitting using `lesion_id` to prevent data leakage
- Stratified to maintain class distribution

---

## Author

**Rishi Pandey** — Final Year B.Tech Project
