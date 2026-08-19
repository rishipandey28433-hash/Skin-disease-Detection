# ============================================================
# SKIN DISEASE CLASSIFICATION & STAGE IDENTIFICATION
# IMPROVED EFFICIENTNET-B0 TRAINING
#
# 9 CLASSES:
# akiec, bcc, bkl, df, healthy, mel, nv, vasc, vitiligo
#
# OUTPUT:
# models/skin_model.keras
# models/class_names.json
# reports/training_report.txt
# reports/confusion_matrix.png
# reports/training_curves.png
# ============================================================

import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    BatchNormalization
)
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau
)

# ============================================================
# 1. CONFIGURATION
# ============================================================

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
HEAD_EPOCHS = 10
FINE_TUNE_EPOCHS = 20
SEED = 42

# ============================================================
# 2. PROJECT ROOT
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# 3. DATASET PATHS
# ============================================================

TRAIN_PATH = os.path.join(BASE_DIR, "dataset", "train")
VAL_PATH = os.path.join(BASE_DIR, "dataset", "val")
TEST_PATH = os.path.join(BASE_DIR, "dataset", "test")

# ============================================================
# 4. MODEL & REPORT PATHS
# ============================================================

MODEL_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

MODEL_PATH = os.path.join(MODEL_DIR, "skin_model.keras")
CLASS_NAMES_PATH = os.path.join(MODEL_DIR, "class_names.json")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# ============================================================
# 5. FINAL CLASS NAMES
# ============================================================

CLASS_NAMES = [
    "akiec",
    "bcc",
    "bkl",
    "df",
    "healthy",
    "mel",
    "nv",
    "vasc",
    "vitiligo"
]
NUM_CLASSES = len(CLASS_NAMES)

# ============================================================
# 6. RANDOM SEED
# ============================================================

np.random.seed(SEED)
tf.random.set_seed(SEED)

# ============================================================
# 7. HEADER
# ============================================================

print("\n")
print("=" * 70)
print("SKIN AI - EFFICIENTNET-B0 TRAINING")
print("FINAL MODEL: 9 CLASSES")
print("=" * 70)

# ============================================================
# 8. DATASET PATH CHECK
# ============================================================

print("\n")
print("=" * 70)
print("CHECKING DATASET PATHS")
print("=" * 70)

for path_name, path in [("Train", TRAIN_PATH), ("Validation", VAL_PATH), ("Test", TEST_PATH)]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"\n{path_name} folder not found:\n{path}\n")
    print(f"{path_name:12s} : {path}")

print("=" * 70)

# ============================================================
# 9. VALID IMAGE EXTENSIONS
# ============================================================

VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# ============================================================
# 10. COLLECT IMAGES FUNCTION
# ============================================================

def collect_images(folder_path, class_name):
    records = []
    if not os.path.exists(folder_path):
        return records
    for root, _, files in os.walk(folder_path):
        for file_name in files:
            if file_name.lower().endswith(VALID_EXTENSIONS):
                full_path = os.path.join(root, file_name)
                records.append({
                    "filename": full_path,
                    "class": class_name
                })
    return records

# ============================================================
# 11. COLLECT DATA (TRAIN, VAL, TEST)
# ============================================================

def build_dataframe(base_path, dataset_name):
    print("\n")
    print("=" * 70)
    print(f"COLLECTING {dataset_name.upper()} IMAGES")
    print("=" * 70)
    
    records = []
    for class_name in CLASS_NAMES:
        class_folder = os.path.join(base_path, class_name)
        class_records = collect_images(class_folder, class_name)
        records.extend(class_records)
        print(f"{class_name:10s} : {len(class_records)} images")
        
    df = pd.DataFrame(records)
    if df.empty:
        raise RuntimeError(f"{dataset_name} dataset is empty.")
    
    print("\n")
    print(f"Total {dataset_name} images:", len(df))
    print("=" * 70)
    return df

train_df = build_dataframe(TRAIN_PATH, "train")
val_df = build_dataframe(VAL_PATH, "val")
test_df = build_dataframe(TEST_PATH, "test")

# ============================================================
# 12. CHECK ALL CLASSES HAVE IMAGES
# ============================================================

print("\n")
print("=" * 70)
print("CHECKING CLASS AVAILABILITY")
print("=" * 70)

for class_name in CLASS_NAMES:
    train_count = len(train_df[train_df["class"] == class_name])
    val_count = len(val_df[val_df["class"] == class_name])
    test_count = len(test_df[test_df["class"] == class_name])
    
    if train_count == 0:
        raise RuntimeError(f"No training images found for class: {class_name}")
    if val_count == 0:
        raise RuntimeError(f"No val images found for class: {class_name}")
    if test_count == 0:
        raise RuntimeError(f"No test images found for class: {class_name}")
        
    print(f"{class_name:10s} -> train={train_count}, val={val_count}, test={test_count}")

print("=" * 70)

# ============================================================
# 13. DATA AUGMENTATION
# ============================================================

train_datagen = ImageDataGenerator(
    rotation_range=25,
    width_shift_range=0.12,
    height_shift_range=0.12,
    shear_range=0.10,
    zoom_range=0.20,
    horizontal_flip=True,
    vertical_flip=False,
    brightness_range=[0.85, 1.15],
    fill_mode="nearest"
)

test_val_datagen = ImageDataGenerator()

# ============================================================
# 14. GENERATORS
# ============================================================

print("\n")
print("=" * 70)
print("CREATING GENERATORS")
print("=" * 70)

train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col="filename",
    y_col="class",
    classes=CLASS_NAMES,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True,
    seed=SEED,
    validate_filenames=True
)

val_generator = test_val_datagen.flow_from_dataframe(
    dataframe=val_df,
    x_col="filename",
    y_col="class",
    classes=CLASS_NAMES,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False,
    validate_filenames=True
)

test_generator = test_val_datagen.flow_from_dataframe(
    dataframe=test_df,
    x_col="filename",
    y_col="class",
    classes=CLASS_NAMES,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False,
    validate_filenames=True
)

# ============================================================
# 15. CLASS WEIGHTS
# ============================================================

print("\n")
print("=" * 70)
print("COMPUTING CLASS WEIGHTS")
print("=" * 70)

train_classes = train_generator.classes

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_classes),
    y=train_classes
)

class_weight_dict = dict(enumerate(class_weights))

for idx, name in enumerate(CLASS_NAMES):
    print(f"{name:10s} : {class_weight_dict[idx]:.4f}")

print("=" * 70)

# ============================================================
# 16. MODEL ARCHITECTURE
# ============================================================

print("\n")
print("=" * 70)
print("BUILDING MODEL (EFFICIENTNET-B0)")
print("=" * 70)

base_model = EfficientNetB0(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
)

base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.45)(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.30)(x)
predictions = Dense(NUM_CLASSES, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.summary()

# ============================================================
# 17. CALLBACKS
# ============================================================

checkpoint = ModelCheckpoint(
    MODEL_PATH,
    monitor="val_loss",
    save_best_only=True,
    mode="min",
    verbose=1
)

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=8,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=3,
    min_lr=1e-6,
    verbose=1
)

callbacks = [checkpoint, early_stopping, reduce_lr]

# ============================================================
# 18. STAGE 1: TRAINING HEAD
# ============================================================

print("\n")
print("=" * 70)
print("STAGE 1: TRAINING CUSTOM HEAD")
print("=" * 70)

model.compile(
    optimizer=Adam(learning_rate=3e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

history_head = model.fit(
    train_generator,
    epochs=HEAD_EPOCHS,
    validation_data=val_generator,
    class_weight=class_weight_dict,
    callbacks=callbacks,
    verbose=1
)

# ============================================================
# 19. STAGE 2: FINE-TUNING
# ============================================================

print("\n")
print("=" * 70)
print("STAGE 2: FINE-TUNING TOP 30%")
print("=" * 70)

num_layers = len(base_model.layers)
freeze_until = int(num_layers * 0.7)

base_model.trainable = True

for layer in base_model.layers[:freeze_until]:
    layer.trainable = False

for layer in base_model.layers:
    if isinstance(layer, BatchNormalization):
        layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

history_finetune = model.fit(
    train_generator,
    epochs=FINE_TUNE_EPOCHS,
    validation_data=val_generator,
    class_weight=class_weight_dict,
    callbacks=callbacks,
    verbose=1
)

# ============================================================
# 20. SAVE CLASS NAMES
# ============================================================

print("\n")
print("=" * 70)
print("SAVING CLASS NAMES")
print("=" * 70)

with open(CLASS_NAMES_PATH, "w") as f:
    json.dump(CLASS_NAMES, f, indent=4)

print(f"Class names saved to: {CLASS_NAMES_PATH}")

# ============================================================
# 21. COMBINE HISTORY
# ============================================================

history_full = {}
for k in history_head.history.keys():
    history_full[k] = history_head.history[k] + history_finetune.history[k]

# ============================================================
# 22. PLOT TRAINING CURVES
# ============================================================

print("\n")
print("=" * 70)
print("GENERATING TRAINING CURVES")
print("=" * 70)

plt.figure(figsize=(12, 5))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(history_full["accuracy"], label="Train Accuracy")
plt.plot(history_full["val_accuracy"], label="Val Accuracy")
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

# Loss
plt.subplot(1, 2, 2)
plt.plot(history_full["loss"], label="Train Loss")
plt.plot(history_full["val_loss"], label="Val Loss")
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

curves_path = os.path.join(REPORTS_DIR, "training_curves.png")
plt.tight_layout()
plt.savefig(curves_path)
plt.close()

print(f"Training curves saved to: {curves_path}")

# ============================================================
# 23. EVALUATION ON TEST SET
# ============================================================

print("\n")
print("=" * 70)
print("EVALUATING ON TEST SET")
print("=" * 70)

test_loss, test_acc = model.evaluate(test_generator, verbose=1)
print(f"\nTest Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")

test_generator.reset()
predictions = model.predict(test_generator, verbose=1)
y_pred = np.argmax(predictions, axis=1)
y_true = test_generator.classes

# ============================================================
# 24. CONFUSION MATRIX & REPORT
# ============================================================

print("\n")
print("=" * 70)
print("GENERATING REPORTS")
print("=" * 70)

cm = confusion_matrix(y_true, y_pred)
report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, output_dict=False)

print("\nClassification Report:\n")
print(report)

# Save confusion matrix heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
cm_path = os.path.join(REPORTS_DIR, "confusion_matrix.png")
plt.savefig(cm_path)
plt.close()
print(f"Confusion matrix saved to: {cm_path}")

# Per-class accuracy
per_class_acc = cm.diagonal() / cm.sum(axis=1)

# Save text report
report_path = os.path.join(REPORTS_DIR, "training_report.txt")
with open(report_path, "w") as f:
    f.write("=" * 50 + "\n")
    f.write("SKIN AI TRAINING REPORT\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"Test Loss: {test_loss:.4f}\n")
    f.write(f"Test Accuracy: {test_acc:.4f}\n\n")
    f.write("CLASSIFICATION REPORT:\n")
    f.write(report + "\n\n")
    f.write("PER-CLASS ACCURACY:\n")
    for name, acc in zip(CLASS_NAMES, per_class_acc):
        f.write(f"{name:10s}: {acc*100:.2f}%\n")

print(f"Training report saved to: {report_path}")

print("\n")
print("=" * 70)
print("TRAINING COMPLETED SUCCESSFULLY")
print("=" * 70)
print("\n")