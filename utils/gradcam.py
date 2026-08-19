# ============================================================
# SKIN DISEASE AI — GRAD-CAM EXPLAINABILITY MODULE
# ============================================================
#
# Generates a Grad-CAM heatmap overlay for the EfficientNetB0
# disease classification model to provide visual AI explainability.
#
# ============================================================

import os
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image


def find_last_conv_layer(model):
    """
    Locate the last convolutional layer in the model or nested base_model.
    """
    # Look directly in model.layers
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
        # If it's a nested base model (like EfficientNetB0)
        if hasattr(layer, "layers"):
            for nested_layer in reversed(layer.layers):
                if isinstance(nested_layer, tf.keras.layers.Conv2D):
                    return nested_layer.name

    # Default fallback for EfficientNetB0
    return "top_conv"


def generate_gradcam(model, image_array, class_index=None, overlay_alpha=0.4):
    """
    Generate a Grad-CAM heatmap for a preprocessed image array.

    Args:
        model: Loaded Keras EfficientNetB0 model
        image_array: Preprocessed image tensor with shape (1, 224, 224, 3)
        class_index: Target class index (defaults to top predicted class)
        overlay_alpha: Opacity of heatmap overlay (0.0 to 1.0)

    Returns:
        heatmap_overlay: RGB uint8 NumPy array of the image with heatmap overlay (224, 224, 3)
    """
    try:
        # Determine last conv layer
        last_conv_layer_name = find_last_conv_layer(model)

        # Check if base_model is nested inside model
        base_layer = None
        for layer in model.layers:
            if hasattr(layer, "layers") and len(layer.layers) > 10:
                base_layer = layer
                break

        if base_layer is not None:
            # Model with nested base_model
            grad_model = tf.keras.models.Model(
                inputs=model.inputs,
                outputs=[base_layer.get_layer(last_conv_layer_name).output, model.output]
            )
        else:
            # Flattened model
            grad_model = tf.keras.models.Model(
                inputs=model.inputs,
                outputs=[model.get_layer(last_conv_layer_name).output, model.output]
            )

        # Compute gradient of top class output w.r.t. conv layer output
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(image_array)
            if class_index is None:
                class_index = tf.argmax(predictions[0])
            loss = predictions[:, class_index]

        # Extract gradients
        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # Weight feature maps by gradients
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        # Apply ReLU activation to keep positive influences only
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
        heatmap = heatmap.numpy()

        # Resize heatmap to match image size (224, 224)
        heatmap = cv2.resize(heatmap, (224, 224))
        heatmap_uint8 = np.uint8(255 * heatmap)

        # Apply JET colormap
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

        # Original image in 0-255 uint8 format
        orig_img = np.uint8(np.squeeze(image_array))
        orig_img_bgr = cv2.cvtColor(orig_img, cv2.COLOR_RGB2BGR)

        # Combine original image with colored heatmap
        overlay_bgr = cv2.addWeighted(orig_img_bgr, 1 - overlay_alpha, heatmap_colored, overlay_alpha, 0)
        overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)

        return overlay_rgb

    except Exception as e:
        print(f"[GRAD-CAM] Heatmap generation warning: {str(e)}")
        # Return original image if Grad-CAM fails
        try:
            return np.uint8(np.squeeze(image_array))
        except Exception:
            return np.zeros((224, 224, 3), dtype=np.uint8)


def save_gradcam_image(model, image_path, output_path, class_index=None):
    """
    Generate and save a Grad-CAM image to output_path.
    """
    try:
        from utils.preprocess import preprocess_image
        image_array = preprocess_image(image_path)
        overlay_rgb = generate_gradcam(model, image_array, class_index=class_index)

        # Save to file
        pil_img = Image.fromarray(overlay_rgb)
        pil_img.save(output_path)
        return output_path

    except Exception as e:
        print(f"[GRAD-CAM] Failed to save Grad-CAM image: {str(e)}")
        return None
