# ============================================================
# SKIN DISEASE AI
# MODEL ARCHITECTURE
# ============================================================
#
# Architecture: EfficientNetB0
# Pretrained:   ImageNet
# Input:        (224, 224, 3)
# Output:       Softmax over NUM_CLASSES
#
# Classification Head:
#   GlobalAveragePooling2D
#   BatchNormalization
#   Dense(256, relu)
#   Dropout(0.45)
#   Dense(128, relu)
#   Dropout(0.30)
#   Dense(NUM_CLASSES, softmax)
#
# ============================================================

import tensorflow as tf

from tensorflow.keras.applications import EfficientNetB0

from tensorflow.keras.layers import (
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    BatchNormalization
)

from tensorflow.keras.models import Model


# ============================================================
# BUILD EFFICIENTNET-B0 MODEL
# ============================================================

def build_model(
    num_classes,
    input_shape=(224, 224, 3),
    dropout_1=0.45,
    dropout_2=0.30,
    dense_1=256,
    dense_2=128
):
    """
    Build an EfficientNetB0-based classification model.

    Architecture:
        EfficientNetB0 (ImageNet, frozen)
        -> GlobalAveragePooling2D
        -> BatchNormalization
        -> Dense(dense_1, relu)
        -> Dropout(dropout_1)
        -> Dense(dense_2, relu)
        -> Dropout(dropout_2)
        -> Dense(num_classes, softmax)

    Parameters:
        num_classes:  Number of output classes
        input_shape:  Input image shape (H, W, C)
        dropout_1:    Dropout rate after first dense layer
        dropout_2:    Dropout rate after second dense layer
        dense_1:      Units in first dense layer
        dense_2:      Units in second dense layer

    Returns:
        Compiled Keras Model
    """

    # --------------------------------------------------------
    # BASE MODEL
    # --------------------------------------------------------

    base_model = EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape
    )

    base_model.trainable = False

    # --------------------------------------------------------
    # CLASSIFICATION HEAD
    # --------------------------------------------------------

    x = base_model.output

    x = GlobalAveragePooling2D(
        name="head_global_pool"
    )(x)

    x = BatchNormalization(
        name="head_batch_norm"
    )(x)

    x = Dense(
        dense_1,
        activation="relu",
        name="head_dense_1"
    )(x)

    x = Dropout(
        dropout_1,
        name="head_dropout_1"
    )(x)

    x = Dense(
        dense_2,
        activation="relu",
        name="head_dense_2"
    )(x)

    x = Dropout(
        dropout_2,
        name="head_dropout_2"
    )(x)

    output = Dense(
        num_classes,
        activation="softmax",
        name="head_output"
    )(x)

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model = Model(
        inputs=base_model.input,
        outputs=output,
        name="skin_efficientnet_b0"
    )

    return model, base_model


# ============================================================
# CONFIGURE FINE-TUNING
# ============================================================

def configure_fine_tuning(
    model,
    base_model,
    freeze_ratio=0.70
):
    """
    Unfreeze the top portion of EfficientNetB0 for fine-tuning.

    Keeps BatchNormalization layers frozen to maintain
    stable statistics during fine-tuning.

    Parameters:
        model:         The full model
        base_model:    The EfficientNetB0 base model
        freeze_ratio:  Fraction of base layers to keep frozen
    """

    base_model.trainable = True

    total_layers = len(base_model.layers)

    freeze_count = int(
        total_layers * freeze_ratio
    )

    for layer in base_model.layers[:freeze_count]:
        layer.trainable = False

    # Keep BatchNormalization layers frozen
    # for stable fine-tuning
    for layer in base_model.layers[freeze_count:]:
        if isinstance(
            layer,
            tf.keras.layers.BatchNormalization
        ):
            layer.trainable = False

    trainable = sum(
        1 for layer in base_model.layers
        if layer.trainable
    )

    print(
        f"Fine-tuning: {trainable}/{total_layers} "
        f"base layers trainable"
    )

    return model


# ============================================================
# DEBUG / TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("MODEL ARCHITECTURE TEST")
    print("=" * 60)

    test_model, test_base = build_model(
        num_classes=9
    )

    test_model.summary()

    print("\nModel built successfully.")
    print(f"Input shape: {test_model.input_shape}")
    print(f"Output shape: {test_model.output_shape}")