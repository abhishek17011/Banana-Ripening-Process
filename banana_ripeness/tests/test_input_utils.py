import numpy as np
import pytest
from PIL import Image

from src.input_utils import normalize_image_input


def test_rgb_array_is_preserved():
    rgb = np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255]]], dtype=np.uint8)

    normalized = normalize_image_input(rgb)

    assert normalized.shape == rgb.shape
    assert np.array_equal(normalized, rgb)


@pytest.mark.parametrize("image", [
    np.zeros((4, 5), dtype=np.uint8),
    np.zeros((4, 5, 1), dtype=np.uint8),
    np.zeros((4, 5, 3), dtype=np.uint8),
    np.zeros((4, 5, 4), dtype=np.uint8),
])
def test_supported_image_shapes_normalize_to_rgb(image):
    normalized = normalize_image_input(image)

    assert normalized.shape == (4, 5, 3)


def test_pil_image_is_converted_to_rgb():
    image = Image.new("RGBA", (5, 4), (10, 20, 30, 40))

    normalized = normalize_image_input(image)

    assert normalized.shape == (4, 5, 3)
    assert np.array_equal(normalized[0, 0], [10, 20, 30])


def test_none_is_safe():
    assert normalize_image_input(None) is None


def test_unsupported_image_shape_raises_value_error():
    with pytest.raises(ValueError, match="Unsupported image shape"):
        normalize_image_input(np.zeros((2, 3, 2), dtype=np.uint8))
