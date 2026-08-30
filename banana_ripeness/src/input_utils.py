from __future__ import annotations

from io import BytesIO

import numpy as np
from PIL import Image, ImageOps


def normalize_image_input(image_input):
    """Normalize image inputs into RGB NumPy arrays with three channels."""
    if image_input is None:
        return None

    try:
        if isinstance(image_input, Image.Image):
            pil_image = ImageOps.exif_transpose(image_input)
            pil_image.load()
            image = np.array(pil_image.convert("RGB"))
        elif hasattr(image_input, "getvalue"):
            image_bytes = image_input.getvalue()
            with Image.open(BytesIO(image_bytes)) as opened_image:
                opened_image.verify()
            with Image.open(BytesIO(image_bytes)) as opened_image:
                image = np.array(ImageOps.exif_transpose(opened_image).convert("RGB"))
        else:
            image = np.array(image_input)
    except (OSError, ValueError) as exc:
        raise ValueError("Invalid or corrupted image. Please upload another image.") from exc

    if image.size == 0:
        raise ValueError("The selected image is empty.")

    if image.ndim == 2:
        image = np.stack((image,) * 3, axis=-1)
    elif image.ndim != 3:
        raise ValueError(f"Unsupported image shape {image.shape}; expected (H, W) or (H, W, C).")
    elif image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)
    elif image.shape[2] == 4:
        image = image[:, :, :3]
    elif image.shape[2] != 3:
        raise ValueError(f"Unsupported image shape {image.shape}; expected 1, 3, or 4 channels.")

    return np.ascontiguousarray(image)
