# ============================================================
# SKIN DISEASE CLASSIFICATION & STAGE IDENTIFICATION
# IMPROVED EFFICIENTNET-B0 TRAINING
# ============================================================

import os
import json
import numpy as np
import tensorflow as tf

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

from sklearn.utils.class_weight import compute_class_weight


# ============================================================
# 1. CONFIGURATION
# ============================================================

IMG_SIZE = (224, 224)

BATCH_SIZE = 32

HEAD_EPOCHS = 10

FINE_TUNE_EPOCHS = 20

TRAIN_PATH = "dataset/train"

TEST_PATH = "dataset/test"

MODEL_DIR = "models"

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
# 2. RANDOM SEED
# ============================================================

SEED = 42

np.random.seed(SEED)

tf.random.set_seed(SEED)


# ============================================================
# 3. CHECK DATASET PATHS
# ============================================================

print("\n")
print("=" * 65)
print("CHECKING DATASET")
print("=" * 65)

if not os.path.exists(TRAIN_PATH):

    raise FileNotFoundError(
        f"Training folder not found: {TRAIN_PATH}"
    )

if not os.path.exists(TEST_PATH):

    raise FileNotFoundError(
        f"Test folder not found: {TEST_PATH}"
    )

print("Training folder :", TRAIN_PATH)

print("Test folder     :", TEST_PATH)

print("=" * 65)


# ============================================================
# 4. DATA AUGMENTATION
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
# 5. TEST / VALIDATION PREPROCESSING
# ============================================================

test_datagen = ImageDataGenerator()


# ============================================================
# 6. TRAINING GENERATOR
# ============================================================

train_generator = train_datagen.flow_from_directory(

    TRAIN_PATH,

    target_size=IMG_SIZE,

    batch_size=BATCH_SIZE,

    class_mode="categorical",

    shuffle=True,

    seed=SEED
)


# ============================================================
# 7. TEST GENERATOR
# ============================================================

test_generator = test_datagen.flow_from_directory(

    TEST_PATH,

    target_size=IMG_SIZE,

    batch_size=BATCH_SIZE,

    class_mode="categorical",

    shuffle=False
)


# ============================================================
# 8. CLASS INFORMATION
# ============================================================

print("\n")
print("=" * 65)
print("CLASS INFORMATION")
print("=" * 65)

print(
    "Class Indices:"
)

print(
    train_generator.class_indices
)

print()

print(
    "Number of Classes:",
    train_generator.num_classes
)

print()

class_names = list(
    train_generator.class_indices.keys()
)

print(
    "Class Names:"
)

for index, name in enumerate(class_names):

    print(
        f"{index} -> {name}"
    )

print("=" * 65)


# ============================================================
# 9. SAVE CLASS NAMES
# ============================================================

with open(
    CLASS_NAMES_PATH,
    "w"
) as file:

    json.dump(
        class_names,
        file,
        indent=4
    )

print(
    "\nClass mapping saved:"
)

print(
    CLASS_NAMES_PATH
)


# ============================================================
# 10. DATASET DISTRIBUTION
# ============================================================

print("\n")
print("=" * 65)
print("TRAINING DATASET DISTRIBUTION")
print("=" * 65)

for class_name in class_names:

    folder = os.path.join(
        TRAIN_PATH,
        class_name
    )

    if os.path.exists(folder):

        image_files = [

            file

            for file in os.listdir(folder)

            if file.lower().endswith(
                (
                    ".jpg",
                    ".jpeg",
                    ".png",
                    ".bmp",
                    ".webp"
                )
            )

        ]

        print(
            f"{class_name:10s} : "
            f"{len(image_files)} images"
        )

print("=" * 65)


# ============================================================
# 11. CLASS WEIGHTS
# ============================================================

labels = train_generator.classes

classes = np.unique(labels)

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


print("\n")
print("=" * 65)
print("CLASS WEIGHTS")
print("=" * 65)

for class_name, class_id in (
    train_generator.class_indices.items()
):

    print(

        f"{class_name:10s} : "
        f"{class_weights[class_id]:.4f}"

    )

print("=" * 65)


# ============================================================
# 12. LOAD EFFICIENTNET-B0
# ============================================================

print("\n")
print("Loading EfficientNetB0...")

base_model = EfficientNetB0(

    weights="imagenet",

    include_top=False,

    input_shape=(
        224,
        224,
        3
    )

)


# ============================================================
# 13. STAGE 1
# FREEZE EFFICIENTNET
# ============================================================

base_model.trainable = False


# ============================================================
# 14. CLASSIFICATION HEAD
# ============================================================

x = base_model.output


x = GlobalAveragePooling2D()(x)


x = BatchNormalization()(x)


x = Dense(

    256,

    activation="relu"

)(x)


x = Dropout(

    0.45

)(x)


x = Dense(

    128,

    activation="relu"

)(x)


x = Dropout(

    0.30

)(x)


output = Dense(

    train_generator.num_classes,

    activation="softmax",

    name="disease_prediction"

)(x)


model = Model(

    inputs=base_model.input,

    outputs=output

)


# ============================================================
# 15. STAGE 1 COMPILE
# ============================================================

model.compile(

    optimizer=Adam(

        learning_rate=0.0003

    ),

    loss="categorical_crossentropy",

    metrics=[
        "accuracy"
    ]

)


# ============================================================
# 16. STAGE 1 CALLBACKS
# ============================================================

head_checkpoint = ModelCheckpoint(

    MODEL_PATH,

    monitor="val_accuracy",

    save_best_only=True,

    mode="max",

    verbose=1

)


head_early_stop = EarlyStopping(

    monitor="val_accuracy",

    patience=4,

    mode="max",

    restore_best_weights=True,

    verbose=1

)


head_reduce_lr = ReduceLROnPlateau(

    monitor="val_loss",

    factor=0.3,

    patience=2,

    min_lr=1e-6,

    verbose=1

)


# ============================================================
# 17. STAGE 1 TRAINING
# ============================================================

print("\n")
print("=" * 65)
print("STAGE 1")
print("TRAINING CLASSIFICATION HEAD")
print("=" * 65)


history_head = model.fit(

    train_generator,

    validation_data=test_generator,

    epochs=HEAD_EPOCHS,

    class_weight=class_weights,

    callbacks=[

        head_checkpoint,

        head_early_stop,

        head_reduce_lr

    ],

    verbose=1

)


# ============================================================
# 18. LOAD BEST STAGE 1 MODEL
# ============================================================

print("\n")

print(
    "Loading best Stage 1 model..."
)

model = tf.keras.models.load_model(

    MODEL_PATH,

    compile=False

)


# ============================================================
# 19. GET BASE MODEL
# ============================================================

base_model = None

for layer in model.layers:

    if isinstance(
        layer,
        tf.keras.Model
    ):

        base_model = layer

        break


if base_model is None:

    raise RuntimeError(
        "EfficientNet base model could not be found."
    )


# ============================================================
# 20. STAGE 2 - FINE TUNING
# ============================================================

print("\n")
print("=" * 65)
print("STAGE 2")
print("FINE-TUNING EFFICIENTNET-B0")
print("=" * 65)


base_model.trainable = True


total_layers = len(
    base_model.layers
)


# Keep first 70% frozen
fine_tune_from = int(
    total_layers * 0.70
)


for layer in base_model.layers[
    :fine_tune_from
]:

    layer.trainable = False


for layer in base_model.layers[
    fine_tune_from:
]:

    layer.trainable = True


print(
    "Total EfficientNet layers :",
    total_layers
)

print(
    "Frozen layers             :",
    fine_tune_from
)

print(
    "Trainable layers          :",
    total_layers - fine_tune_from
)


# ============================================================
# 21. RECOMPILE FOR FINE TUNING
# ============================================================

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
# 22. FINE-TUNING CALLBACKS
# ============================================================

fine_checkpoint = ModelCheckpoint(

    MODEL_PATH,

    monitor="val_accuracy",

    save_best_only=True,

    mode="max",

    verbose=1

)


fine_early_stop = EarlyStopping(

    monitor="val_accuracy",

    patience=6,

    mode="max",

    restore_best_weights=True,

    verbose=1

)


fine_reduce_lr = ReduceLROnPlateau(

    monitor="val_loss",

    factor=0.3,

    patience=2,

    min_lr=1e-7,

    verbose=1

)


# ============================================================
# 23. STAGE 2 TRAINING
# ============================================================

print("\n")

history_fine = model.fit(

    train_generator,

    validation_data=test_generator,

    epochs=FINE_TUNE_EPOCHS,

    class_weight=class_weights,

    callbacks=[

        fine_checkpoint,

        fine_early_stop,

        fine_reduce_lr

    ],

    verbose=1

)


# ============================================================
# 24. LOAD FINAL BEST MODEL
# ============================================================

print("\n")

print(
    "Loading final best model..."
)

model = tf.keras.models.load_model(

    MODEL_PATH,

    compile=False

)


# ============================================================
# 25. FINAL EVALUATION
# ============================================================

print("\n")
print("=" * 65)
print("FINAL MODEL EVALUATION")
print("=" * 65)


model.compile(

    optimizer=Adam(
        learning_rate=1e-5
    ),

    loss="categorical_crossentropy",

    metrics=[
        "accuracy"
    ]

)


loss, accuracy = model.evaluate(

    test_generator,

    verbose=1

)


print("\n")

print(
    f"Test Loss     : {loss:.4f}"
)

print(
    f"Test Accuracy : {accuracy * 100:.2f}%"
)

print("=" * 65)


# ============================================================
# 26. CHECK PREDICTION DISTRIBUTION
# ============================================================

print("\n")
print("=" * 65)
print("CHECKING PREDICTION DISTRIBUTION")
print("=" * 65)


test_generator.reset()


predictions = model.predict(

    test_generator,

    verbose=1

)


predicted_classes = np.argmax(

    predictions,

    axis=1

)


print("\n")

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


print("=" * 65)


# ============================================================
# 27. SAVE FINAL MODEL
# ============================================================

model.save(

    MODEL_PATH

)


# ============================================================
# 28. FINAL INFORMATION
# ============================================================

print("\n")
print("=" * 65)
print("MODEL TRAINING COMPLETED SUCCESSFULLY")
print("=" * 65)

print(
    "Model:"
)

print(
    MODEL_PATH
)

print()

print(
    "Class Mapping:"
)

print(
    CLASS_NAMES_PATH
)

print()

print(
    "Classes:"
)

print(
    class_names
)

print()

print(
    f"Final Test Accuracy: "
    f"{accuracy * 100:.2f}%"
)

print("=" * 65)

print(
    "\nTraining completed!"
)
