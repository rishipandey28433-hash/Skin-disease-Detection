# ============================================================
# SKIN DISEASE AI
# IMAGE PREPROCESSING
# ============================================================
#
# Compatible with:
# EfficientNetB0
#
# IMPORTANT:
#
# Training pipeline:
#     ImageDataGenerator
#     target_size = (224, 224)
#     No manual /255 normalization
#
# Therefore prediction pipeline also keeps
# pixel values in the 0-255 range.
#
# ============================================================

from PIL import Image
import numpy as np

from config import IMAGE_SIZE


# ============================================================
# PREPROCESS IMAGE FROM FILE PATH
# ============================================================

def preprocess_image(image_path):
    """
    Prepare an uploaded image for the TensorFlow model.

    Steps:
        1. Open image
        2. Convert to RGB
        3. Resize to IMAGE_SIZE
        4. Convert to float32 NumPy array
        5. Validate shape
        6. Add batch dimension

    Output:
        NumPy array with shape:

        (1, IMAGE_SIZE[0], IMAGE_SIZE[1], 3)

    IMPORTANT:
        No manual /255 normalization is performed here.

        This should match the training pipeline.
    """

    try:

        # ----------------------------------------------------
        # CHECK IMAGE PATH
        # ----------------------------------------------------

        if not image_path:
            raise ValueError(
                "Image path is empty."
            )

        # ----------------------------------------------------
        # CHECK FILE EXISTS
        # ----------------------------------------------------

        import os

        if not os.path.exists(image_path):
            raise FileNotFoundError(
                f"Image file not found: {image_path}"
            )

        # ----------------------------------------------------
        # OPEN IMAGE
        # ----------------------------------------------------

        image = Image.open(
            image_path
        )

        # ----------------------------------------------------
        # FORCE IMAGE LOAD
        # ----------------------------------------------------

        image.load()

        # ----------------------------------------------------
        # CONVERT TO RGB
        # ----------------------------------------------------

        image = image.convert(
            "RGB"
        )

        # ----------------------------------------------------
        # RESIZE
        # ----------------------------------------------------

        image = image.resize(
            IMAGE_SIZE,
            Image.Resampling.LANCZOS
        )

        # ----------------------------------------------------
        # CONVERT TO NUMPY
        # ----------------------------------------------------

        image_array = np.asarray(
            image,
            dtype=np.float32
        )

        # ----------------------------------------------------
        # BASIC DIMENSION CHECK
        # ----------------------------------------------------

        if image_array.ndim != 3:

            raise ValueError(
                "Image must have 3 dimensions."
            )

        # ----------------------------------------------------
        # RGB CHANNEL CHECK
        # ----------------------------------------------------

        if image_array.shape[2] != 3:

            raise ValueError(
                "Image must have exactly "
                "3 RGB channels."
            )

        # ----------------------------------------------------
        # CHECK FOR INVALID NUMBERS
        # ----------------------------------------------------

        if not np.isfinite(
            image_array
        ).all():

            raise ValueError(
                "Image contains invalid pixel values."
            )

        # ----------------------------------------------------
        # IMPORTANT:
        #
        # DO NOT DO:
        #
        # image_array = image_array / 255.0
        #
        # because the current project preprocessing
        # is designed to match the training pipeline.
        # ----------------------------------------------------

        # ----------------------------------------------------
        # ADD BATCH DIMENSION
        # ----------------------------------------------------

        image_array = np.expand_dims(
            image_array,
            axis=0
        )

        # ----------------------------------------------------
        # FINAL EXPECTED SHAPE
        # ----------------------------------------------------

        expected_shape = (

            1,

            IMAGE_SIZE[0],

            IMAGE_SIZE[1],

            3

        )

        # ----------------------------------------------------
        # FINAL SHAPE CHECK
        # ----------------------------------------------------

        if image_array.shape != expected_shape:

            raise ValueError(

                "Invalid processed image shape: "

                f"{image_array.shape}. "

                f"Expected: {expected_shape}"

            )

        # ----------------------------------------------------
        # RETURN
        # ----------------------------------------------------

        return image_array

    except Exception as e:

        raise RuntimeError(

            "Image preprocessing failed: "

            + str(e)

        )


# ============================================================
# PREPROCESS EXISTING PIL IMAGE
# ============================================================

def preprocess_array(image):
    """
    Prepare an already-loaded PIL image.

    Output:

        NumPy array with shape:

        (1, IMAGE_SIZE[0], IMAGE_SIZE[1], 3)

    IMPORTANT:

        No manual /255 normalization.
    """

    try:

        # ----------------------------------------------------
        # CHECK INPUT
        # ----------------------------------------------------

        if image is None:

            raise ValueError(
                "No image was provided."
            )

        # ----------------------------------------------------
        # CONVERT TO RGB
        # ----------------------------------------------------

        image = image.convert(
            "RGB"
        )

        # ----------------------------------------------------
        # RESIZE
        # ----------------------------------------------------

        image = image.resize(
            IMAGE_SIZE,
            Image.Resampling.LANCZOS
        )

        # ----------------------------------------------------
        # NUMPY ARRAY
        # ----------------------------------------------------

        image_array = np.asarray(
            image,
            dtype=np.float32
        )

        # ----------------------------------------------------
        # DIMENSION CHECK
        # ----------------------------------------------------

        if image_array.ndim != 3:

            raise ValueError(
                "Image must have 3 dimensions."
            )

        # ----------------------------------------------------
        # RGB CHECK
        # ----------------------------------------------------

        if image_array.shape[2] != 3:

            raise ValueError(
                "Image must have exactly "
                "3 RGB channels."
            )

        # ----------------------------------------------------
        # INVALID VALUE CHECK
        # ----------------------------------------------------

        if not np.isfinite(
            image_array
        ).all():

            raise ValueError(
                "Image contains invalid pixel values."
            )

        # ----------------------------------------------------
        # NO /255 NORMALIZATION
        # ----------------------------------------------------

        # Keep 0-255 range.

        # ----------------------------------------------------
        # ADD BATCH DIMENSION
        # ----------------------------------------------------

        image_array = np.expand_dims(
            image_array,
            axis=0
        )

        # ----------------------------------------------------
        # EXPECTED SHAPE
        # ----------------------------------------------------

        expected_shape = (

            1,

            IMAGE_SIZE[0],

            IMAGE_SIZE[1],

            3

        )

        # ----------------------------------------------------
        # FINAL SHAPE CHECK
        # ----------------------------------------------------

        if image_array.shape != expected_shape:

            raise ValueError(

                "Invalid image shape: "

                f"{image_array.shape}. "

                f"Expected: {expected_shape}"

            )

        return image_array

    except Exception as e:

        raise RuntimeError(

            "Image array preprocessing failed: "

            + str(e)

        )


# ============================================================
# VALIDATE PROCESSED IMAGE ARRAY
# ============================================================

def validate_image_array(image_array):
    """
    Validate a processed image before TensorFlow prediction.

    Expected:

        NumPy array

        Shape:
        (1, IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
    """

    # --------------------------------------------------------
    # TYPE CHECK
    # --------------------------------------------------------

    if not isinstance(
        image_array,
        np.ndarray
    ):

        raise TypeError(
            "Image must be a NumPy array."
        )

    # --------------------------------------------------------
    # DIMENSION CHECK
    # --------------------------------------------------------

    if image_array.ndim != 4:

        raise ValueError(

            "Expected 4D input, "

            f"got {image_array.ndim}D."

        )

    # --------------------------------------------------------
    # BATCH SIZE CHECK
    # --------------------------------------------------------

    if image_array.shape[0] != 1:

        raise ValueError(
            "Batch size must be exactly 1."
        )

    # --------------------------------------------------------
    # IMAGE SHAPE CHECK
    # --------------------------------------------------------

    expected_shape = (

        1,

        IMAGE_SIZE[0],

        IMAGE_SIZE[1],

        3

    )

    if image_array.shape != expected_shape:

        raise ValueError(

            "Invalid image shape: "

            f"{image_array.shape}. "

            f"Expected: {expected_shape}"

        )

    # --------------------------------------------------------
    # NUMERIC VALUE CHECK
    # --------------------------------------------------------

    if not np.isfinite(
        image_array
    ).all():

        raise ValueError(
            "Image array contains invalid values."
        )

    # --------------------------------------------------------
    # PIXEL RANGE CHECK
    # --------------------------------------------------------

    if (
        np.min(image_array) < 0
        or
        np.max(image_array) > 255
    ):

        raise ValueError(

            "Image pixel values are outside "
            "the expected 0-255 range."

        )

    return True


# ============================================================
# DEBUG / TEST
# ============================================================

if __name__ == "__main__":

    print(
        "=============================================="
    )

    print(
        "SKIN AI IMAGE PREPROCESSOR"
    )

    print(
        "=============================================="
    )

    print(
        "IMAGE_SIZE:",
        IMAGE_SIZE
    )

    print(
        "Expected input shape:",
        (
            1,
            IMAGE_SIZE[0],
            IMAGE_SIZE[1],
            3
        )
    )

    print(
        "Manual /255 normalization: DISABLED"
    )

    print(
        "Preprocessing module loaded successfully."
    )