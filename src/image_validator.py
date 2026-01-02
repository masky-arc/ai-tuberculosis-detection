import cv2
import numpy as np

def is_valid_chest_xray(image_path: str) -> bool:
    """
    Validates whether the uploaded image resembles a chest X-ray.
    Returns True if valid, False otherwise.
    """

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        return False

    # Check resolution (X-rays are usually large & rectangular)
    height, width = image.shape
    if height < 200 or width < 200:
        return False

    # Check grayscale intensity distribution
    mean_intensity = np.mean(image)
    if mean_intensity < 40 or mean_intensity > 220:
        return False

    # Edge density check (lungs create strong edges)
    edges = cv2.Canny(image, 50, 150)
    edge_ratio = np.sum(edges > 0) / (height * width)

    return edge_ratio > 0.02
