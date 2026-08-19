# ============================================================
# SKIN DISEASE CLASSIFICATION & STAGE IDENTIFICATION
# CONTINUE TRAINING - STAGE 2 ONLY
#
# 7 HAM10000 DISEASES + HEALTHY SKIN
#
# IMPORTANT:
# Stage 1 is NOT repeated.
# Existing skin_model.keras is loaded.
# Only Stage 2 fine-tuning is performed.
# ============================================================

import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau
)

from sklearn.utils.class_weight import compute_class_weight


# ============================================================
# 1. CONFIGURATION
# ============================================================

IMG_SIZE = (224, 224)

BATCH_SIZE = 32

FINE_TUNE_EPOCHS = 20

SEED = 42


# ============================================================
# 2. BASE DIRECTORY
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


# ============================================================
# 3. DATASET PATHS
# ============================================================

TRAIN_PATH = os.path.join(
    BASE_DIR,
    "dataset",
    "train"
)

TEST_PATH = os.path.join(
    BASE_DIR,
    "dataset",
    "test"
)


# ============================================================
# 4. HEALTHY DATASET PATHS
# ============================================================

HEALTHY_TRAIN_PATH = os.path.join(
    BASE_DIR,
    "healthy_dataset",
    "train",
    "healthy"
)

HEALTHY_TEST_PATH = os.path.join(
    BASE_DIR,
    "healthy_dataset",
    "test",
    "healthy"
)


# ============================================================
# 5. MODEL PATHS
# ============================================================

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "skin_model.keras"
)

CLASS_NAMES_PATH = os.path.join(
    MODEL_DIR,
    "class_names.json"
)

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


# ============================================================
# 6. CLASS NAMES
# ============================================================

CLASS_NAMES = [
    "akiec",
    "bcc",
    "bkl",
    "df",
    "mel",
    "nv",
    "vasc",
    "healthy"
]

NUM_CLASSES = len(CLASS_NAMES)


# ============================================================
# 7. RANDOM SEED
# ============================================================

np.random.seed(SEED)

tf.random.set_seed(SEED)


# ============================================================
# 8. HEADER
# ============================================================

print("\n")
print("=" * 70)
print("SKIN AI - CONTINUE TRAINING")
print("=" * 70)

print(
    "EXISTING STAGE 1 MODEL WILL BE USED"
)

print(
    "STAGE 1 WILL NOT RUN AGAIN"
)

print(
    "STARTING DIRECTLY FROM STAGE 2"
)

print("=" * 70)


# ============================================================
# 9. CHECK EXISTING MODEL
# ============================================================

print("\n")
print("=" * 70)
print("CHECKING EXISTING STAGE 1 MODEL")
print("=" * 70)


if not os.path.exists(MODEL_PATH):

    raise FileNotFoundError(
        "\nStage 1 model not found:\n"
        f"{MODEL_PATH}\n\n"
        "Please run Stage 1 training first."
    )


print(
    "Existing model found:"
)

print(
    MODEL_PATH
)


# ============================================================
# 10. CHECK DATASET PATHS
# ============================================================

print("\n")
print("=" * 70)
print("CHECKING DATASET PATHS")
print("=" * 70)


required_paths = {

    "HAM10000 train":
        TRAIN_PATH,

    "HAM10000 test":
        TEST_PATH,

    "Healthy train":
        HEALTHY_TRAIN_PATH,

    "Healthy test":
        HEALTHY_TEST_PATH

}


for name, path in required_paths.items():

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"\n{name} folder not found:\n"
            f"{path}"
        )

    print(
        f"{name:20s}: {path}"
    )


print("\nDataset paths verified.")


# ============================================================
# 11. VALID IMAGE EXTENSIONS
# ============================================================

VALID_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
)


# ============================================================
# 12. COLLECT IMAGES FUNCTION
# ============================================================

def collect_images(
    folder_path,
    class_name
):

    records = []

    if not os.path.exists(folder_path):

        raise FileNotFoundError(
            f"Folder not found:\n{folder_path}"
        )

    for root, _, files in os.walk(
        folder_path
    ):

        for file_name in files:

            if file_name.lower().endswith(
                VALID_EXTENSIONS
            ):

                full_path = os.path.join(
                    root,
                    file_name
                )

                records.append(
                    {
                        "filename": full_path,
                        "class": class_name
                    }
                )

    return records


# ============================================================
# 13. COLLECT TRAINING DATA
# ============================================================

print("\n")
print("=" * 70)
print("COLLECTING TRAINING DATA")
print("=" * 70)


train_records = []


# ------------------------------------------------------------
# HAM10000 CLASSES
# ------------------------------------------------------------

for class_name in CLASS_NAMES[:-1]:

    folder = os.path.join(
        TRAIN_PATH,
        class_name
    )

    records = collect_images(
        folder,
        class_name
    )

    train_records.extend(
        records
    )

    print(
        f"{class_name:10s} : "
        f"{len(records)} images"
    )


# ------------------------------------------------------------
# HEALTHY
# ------------------------------------------------------------

healthy_records = collect_images(
    HEALTHY_TRAIN_PATH,
    "healthy"
)

train_records.extend(
    healthy_records
)

print(
    f"{'healthy':10s} : "
    f"{len(healthy_records)} images"
)


# ============================================================
# 14. TRAIN DATAFRAME
# ============================================================

train_df = pd.DataFrame(
    train_records
)

if train_df.empty:

    raise RuntimeError(
        "Training dataset is empty."
    )


print()
print(
    "Total training images:",
    len(train_df)
)


# ============================================================
# 15. COLLECT TEST DATA
# ============================================================

print("\n")
print("=" * 70)
print("COLLECTING TEST DATA")
print("=" * 70)


test_records = []


# ------------------------------------------------------------
# HAM10000 CLASSES
# ------------------------------------------------------------

for class_name in CLASS_NAMES[:-1]:

    folder = os.path.join(
        TEST_PATH,
        class_name
    )

    records = collect_images(
        folder,
        class_name
    )

    test_records.extend(
        records
    )

    print(
        f"{class_name:10s} : "
        f"{len(records)} images"
    )


# ------------------------------------------------------------
# HEALTHY
# ------------------------------------------------------------

healthy_test_records = collect_images(
    HEALTHY_TEST_PATH,
    "healthy"
)

test_records.extend(
    healthy_test_records
)

print(
    f"{'healthy':10s} : "
    f"{len(healthy_test_records)} images"
)


# ============================================================
# 16. TEST DATAFRAME
# ============================================================

test_df = pd.DataFrame(
    test_records
)

if test_df.empty:

    raise RuntimeError(
        "Test dataset is empty."
    )


print()
print(
    "Total test images:",
    len(test_df)
)


# ============================================================
# 17. DATA AUGMENTATION
# ============================================================

train_datagen = ImageDataGenerator(

    rotation_range=25,

    width_shift_range=0.12,

    height_shift_range=0.12,

    shear_range=0.10,

    zoom_range=0.20,

    horizontal_flip=True,

    vertical_flip=False,

    brightness_range=[
        0.85,
        1.15
    ],

    fill_mode="nearest"

)


# ============================================================
# 18. TEST PREPROCESSING
# ============================================================

test_datagen = ImageDataGenerator()


# ============================================================
# 19. TRAIN GENERATOR
# ============================================================

print("\n")
print("=" * 70)
print("CREATING TRAIN GENERATOR")
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


# ============================================================
# 20. TEST GENERATOR
# ============================================================

print("\n")
print("=" * 70)
print("CREATING TEST GENERATOR")
print("=" * 70)


test_generator = test_datagen.flow_from_dataframe(

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
# 21. VERIFY 8 CLASSES
# ============================================================

print("\n")
print("=" * 70)
print("VERIFYING 8 CLASSES")
print("=" * 70)


print(
    "Class indices:"
)

print(
    train_generator.class_indices
)


if len(
    train_generator.class_indices
) != NUM_CLASSES:

    raise RuntimeError(
        "Training generator does not contain "
        "exactly 8 classes."
    )


if len(
    test_generator.class_indices
) != NUM_CLASSES:

    raise RuntimeError(
        "Test generator does not contain "
        "exactly 8 classes."
    )


if "healthy" not in train_generator.class_indices:

    raise RuntimeError(
        "Healthy class missing from training data."
    )


if "healthy" not in test_generator.class_indices:

    raise RuntimeError(
        "Healthy class missing from test data."
    )


print()
print(
    "8-class dataset verified successfully."
)

print(
    "Healthy class index:",
    train_generator.class_indices[
        "healthy"
    ]
)


# ============================================================
# 22. SAVE CLASS MAPPING
# ============================================================

with open(
    CLASS_NAMES_PATH,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        CLASS_NAMES,
        file,
        indent=4
    )


print()
print(
    "Class mapping saved:"
)

print(
    CLASS_NAMES_PATH
)


# ============================================================
# 23. CLASS WEIGHTS
# ============================================================

print("\n")
print("=" * 70)
print("CALCULATING CLASS WEIGHTS")
print("=" * 70)


labels = train_generator.classes

classes = np.unique(
    labels
)


class_weights_array = compute_class_weight(

    class_weight="balanced",

    classes=classes,

    y=labels

)


class_weights = {

    int(class_id): float(weight)

    for class_id, weight
    in zip(
        classes,
        class_weights_array
    )

}


for class_name, class_id in (
    train_generator.class_indices.items()
):

    print(
        f"{class_name:10s} : "
        f"{class_weights[class_id]:.4f}"
    )


# ============================================================
# 24. LOAD EXISTING MODEL
# ============================================================

print("\n")
print("=" * 70)
print("LOADING EXISTING STAGE 1 MODEL")
print("=" * 70)


model = tf.keras.models.load_model(

    MODEL_PATH,

    compile=False

)


print(
    "Existing model loaded successfully."
)

print(
    "Model input shape:",
    model.input_shape
)

print(
    "Model output shape:",
    model.output_shape
)


# ============================================================
# 25. VERIFY MODEL OUTPUT
# ============================================================

if model.output_shape[-1] != NUM_CLASSES:

    raise RuntimeError(

        "\nExisting model is not an 8-class model.\n"

        f"Expected classes: {NUM_CLASSES}\n"

        f"Found classes: {model.output_shape[-1]}\n"

    )


print()
print(
    "Existing model confirmed as 8-class model."
)


# ============================================================
# 26. SHOW MODEL LAYERS
# ============================================================

print("\n")
print("=" * 70)
print("CHECKING SAVED MODEL ARCHITECTURE")
print("=" * 70)


print(
    "Total model layers:",
    len(model.layers)
)


# ============================================================
# 27. FIND EFFICIENTNET LAYERS
# ============================================================

print("\n")
print("=" * 70)
print("FINDING EFFICIENTNET-B0 LAYERS")
print("=" * 70)


# ------------------------------------------------------------
# IMPORTANT:
#
# In your saved model EfficientNet is NOT a nested model.
#
# Layers such as:
#
# block1a...
# block2a...
# block6a...
# block7a...
# top_conv
# top_bn
# top_activation
#
# are directly inside model.layers.
#
# Therefore we identify EfficientNet layers by their names.
# ------------------------------------------------------------


efficientnet_layers = []


for layer in model.layers:

    layer_name = layer.name.lower()

    is_efficientnet_layer = (

        layer_name.startswith("stem_")

        or layer_name.startswith("block1")

        or layer_name.startswith("block2")

        or layer_name.startswith("block3")

        or layer_name.startswith("block4")

        or layer_name.startswith("block5")

        or layer_name.startswith("block6")

        or layer_name.startswith("block7")

        or layer_name.startswith("top_")

    )

    if is_efficientnet_layer:

        efficientnet_layers.append(
            layer
        )


# ============================================================
# 28. VERIFY EFFICIENTNET FOUND
# ============================================================

if len(efficientnet_layers) == 0:

    raise RuntimeError(

        "\nCould not find EfficientNet-B0 layers "
        "inside the saved model.\n\n"

        "The saved model architecture is "
        "different from the expected EfficientNet-B0 "
        "architecture."

    )


print(
    "EfficientNet-B0 layers found:",
    len(efficientnet_layers)
)


print()
print(
    "First EfficientNet layer:",
    efficientnet_layers[0].name
)


print(
    "Last EfficientNet layer:",
    efficientnet_layers[-1].name
)


# ============================================================
# 29. STAGE 2 FINE-TUNING
# ============================================================

print("\n")
print("=" * 70)
print("STAGE 2 - FINE-TUNING")
print("=" * 70)

print(
    "STAGE 1 WILL NOT RUN AGAIN"
)


# ============================================================
# 30. FREEZE ALL EFFICIENTNET LAYERS FIRST
# ============================================================

for layer in efficientnet_layers:

    layer.trainable = False


# ============================================================
# 31. CALCULATE FINE-TUNE START
# ============================================================

total_efficientnet_layers = len(
    efficientnet_layers
)


fine_tune_from = int(
    total_efficientnet_layers * 0.70
)


# ============================================================
# 32. UNFREEZE LAST 30%
# ============================================================

for layer in efficientnet_layers[
    fine_tune_from:
]:

    # BatchNormalization layers remain frozen
    # for stable fine-tuning.

    if isinstance(
        layer,
        BatchNormalization
    ):

        layer.trainable = False

    else:

        layer.trainable = True


# ============================================================
# 33. KEEP CLASSIFICATION HEAD TRAINABLE
# ============================================================

# All non-EfficientNet layers are the classification head.
#
# Keep them trainable.

for layer in model.layers:

    if layer not in efficientnet_layers:

        layer.trainable = True


# ============================================================
# 34. COUNT TRAINABLE / FROZEN LAYERS
# ============================================================

trainable_efficientnet = sum(

    1

    for layer in efficientnet_layers

    if layer.trainable

)


frozen_efficientnet = sum(

    1

    for layer in efficientnet_layers

    if not layer.trainable

)


trainable_total = sum(

    1

    for layer in model.layers

    if layer.trainable

)


frozen_total = sum(

    1

    for layer in model.layers

    if not layer.trainable

)


print()
print(
    "Total EfficientNet layers :",
    total_efficientnet_layers
)

print(
    "Frozen EfficientNet layers:",
    frozen_efficientnet
)

print(
    "Trainable EfficientNet layers:",
    trainable_efficientnet
)

print(
    "Total trainable model layers:",
    trainable_total
)

print(
    "Total frozen model layers:",
    frozen_total
)


# ============================================================
# 35. COMPILE STAGE 2
# ============================================================

print("\n")
print(
    "Compiling model for Stage 2..."
)


model.compile(

    optimizer=Adam(

        learning_rate=1e-5

    ),

    loss="categorical_crossentropy",

    metrics=[
        "accuracy"
    ]

)


# ============================================================
# 36. STAGE 2 CHECKPOINT
# ============================================================

checkpoint = ModelCheckpoint(

    MODEL_PATH,

    monitor="val_accuracy",

    save_best_only=True,

    mode="max",

    verbose=1

)


# ============================================================
# 37. EARLY STOPPING
# ============================================================

early_stop = EarlyStopping(

    monitor="val_accuracy",

    patience=6,

    mode="max",

    restore_best_weights=True,

    verbose=1

)


# ============================================================
# 38. LEARNING RATE REDUCTION
# ============================================================

reduce_lr = ReduceLROnPlateau(

    monitor="val_loss",

    factor=0.3,

    patience=2,

    min_lr=1e-7,

    verbose=1

)


# ============================================================
# 39. START STAGE 2
# ============================================================

print("\n")
print("=" * 70)
print("STARTING STAGE 2 TRAINING")
print("=" * 70)

print(
    "Stage 1: SKIPPED"
)

print(
    "Stage 2: STARTING NOW"
)

print(
    "Healthy class: INCLUDED"
)

print(
    "HAM10000 classes: INCLUDED"
)

print("=" * 70)


history = model.fit(

    train_generator,

    validation_data=test_generator,

    epochs=FINE_TUNE_EPOCHS,

    class_weight=class_weights,

    callbacks=[

        checkpoint,

        early_stop,

        reduce_lr

    ],

    verbose=1

)


# ============================================================
# 40. LOAD BEST STAGE 2 MODEL
# ============================================================

print("\n")
print("=" * 70)
print("LOADING BEST STAGE 2 MODEL")
print("=" * 70)


best_model = tf.keras.models.load_model(

    MODEL_PATH,

    compile=False

)


print(
    "Best Stage 2 model loaded successfully."
)


# ============================================================
# 41. FINAL EVALUATION
# ============================================================

print("\n")
print("=" * 70)
print("FINAL MODEL EVALUATION")
print("=" * 70)


best_model.compile(

    optimizer=Adam(

        learning_rate=1e-5

    ),

    loss="categorical_crossentropy",

    metrics=[
        "accuracy"
    ]

)


test_generator.reset()


loss, accuracy = best_model.evaluate(

    test_generator,

    verbose=1

)


print("\n")

print(
    f"Final Test Loss     : {loss:.4f}"
)

print(
    f"Final Test Accuracy : "
    f"{accuracy * 100:.2f}%"
)


# ============================================================
# 42. PREDICTION DISTRIBUTION
# ============================================================

print("\n")
print("=" * 70)
print("CHECKING PREDICTION DISTRIBUTION")
print("=" * 70)


test_generator.reset()


predictions = best_model.predict(

    test_generator,

    verbose=1

)


predicted_classes = np.argmax(

    predictions,

    axis=1

)


for class_name, class_id in (
    train_generator.class_indices.items()
):

    count = np.sum(

        predicted_classes
        == class_id

    )

    print(

        f"{class_name:10s} : "
        f"{count}"

    )


# ============================================================
# 43. HEALTHY CLASS CHECK
# ============================================================

print("\n")
print("=" * 70)
print("HEALTHY CLASS PREDICTION CHECK")
print("=" * 70)


healthy_id = train_generator.class_indices[
    "healthy"
]


healthy_total = np.sum(

    test_generator.classes
    == healthy_id

)


healthy_correct = np.sum(

    (

        test_generator.classes
        == healthy_id

    )

    &

    (

        predicted_classes
        == healthy_id

    )

)


print(
    "Healthy test images        :",
    healthy_total
)


print(
    "Correctly predicted healthy:",
    healthy_correct
)


if healthy_total > 0:

    healthy_accuracy = (

        healthy_correct
        /
        healthy_total
        *
        100

    )

    print(

        f"Healthy class accuracy    : "
        f"{healthy_accuracy:.2f}%"

    )


# ============================================================
# 44. SAVE FINAL MODEL
# ============================================================

print("\n")
print("=" * 70)
print("SAVING FINAL STAGE 2 MODEL")
print("=" * 70)


best_model.save(
    MODEL_PATH
)


# ============================================================
# 45. SAVE CLASS MAPPING
# ============================================================

with open(
    CLASS_NAMES_PATH,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        CLASS_NAMES,
        file,
        indent=4
    )


print(
    "Model saved:"
)

print(
    MODEL_PATH
)

print()

print(
    "Class mapping saved:"
)

print(
    CLASS_NAMES_PATH
)


# ============================================================
# 46. FINAL VERIFICATION
# ============================================================

print("\n")
print("=" * 70)
print("FINAL MODEL VERIFICATION")
print("=" * 70)


verification_model = (
    tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )
)


print(
    "Model path:",
    MODEL_PATH
)


print(
    "Input shape:",
    verification_model.input_shape
)


print(
    "Output shape:",
    verification_model.output_shape
)


print(
    "Output classes:",
    verification_model.output_shape[-1]
)


if verification_model.output_shape[-1] != NUM_CLASSES:

    raise RuntimeError(
        "Final model is NOT an 8-class model."
    )


# ============================================================
# 47. FINAL SUCCESS
# ============================================================

print("\n")
print("=" * 70)
print("STAGE 2 TRAINING COMPLETED SUCCESSFULLY")
print("=" * 70)


print(
    "Stage 1 : ALREADY COMPLETED"
)

print(
    "Stage 1 : NOT REPEATED"
)

print(
    "Stage 2 : COMPLETED"
)

print(
    "HAM10000 : INCLUDED"
)

print(
    "Healthy : INCLUDED"
)

print(
    "Classes : 8"
)

print(
    f"Final Accuracy : "
    f"{accuracy * 100:.2f}%"
)

print()

print(
    "Updated model:"
)

print(
    MODEL_PATH
)

print()

print(
    "Updated class mapping:"
)

print(
    CLASS_NAMES_PATH
)

print("=" * 70)

print()
print(
    "Continue training completed!"
)

print()
print(
    "IMPORTANT:"
)

print(
    "Restart Flask so app.py loads the updated model."
)

print("=" * 70)