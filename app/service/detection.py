"""
Baseline OpenCV & Tesseract License Plate Detection.
===================================================

This module provides a classic, non-deep-learning computer vision baseline
for detecting and reading license plates using:
1. Grayscale luminance transformation.
2. Gaussian blurring for high-frequency noise suppression.
3. Canny edge detection.
4. Morphological contour approximation (Ramer-Douglas-Peucker algorithm).
5. Aspect-ratio heuristics (2:1 to 4:1) for rectangular plate filtering.
6. Tesseract OCR character extraction.
"""

import cv2
import numpy as np
import pytesseract


def detect_number_plate(image_path: str) -> np.ndarray:
    """
    Detects potential license plate regions using classical contour geometry.

    Pipeline Steps:
    1. Read the input image from disk.
    2. Convert to single-channel grayscale to simplify processing.
    3. Apply Gaussian blur (5x5 kernel) to suppress noise and fine textures.
    4. Canny edge detector computes spatial intensity gradients with double thresholding (50, 150).
    5. Find external and hierarchical contours.
    6. For each contour, compute perimeter and approximate polygon (cv2.approxPolyDP).
    7. If the polygon has 4 vertices (quadrilateral) and an aspect ratio (width/height)
       between 2.0 and 4.0, mark it with a bounding rectangle.

    Args:
        image_path: Path to the image file.

    Returns:
        Image with drawn bounding box candidates.
    """
    # Load the image from disk
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Unable to read image at {image_path}")

    # Convert BGR color image to single-channel grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Blur (5x5 kernel) to smooth noise before edge detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Use Canny Edge Detection with lower threshold 50 and upper threshold 150
    edges = cv2.Canny(blurred, 50, 150)

    # Extract geometric contours from the binary edge map
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        # Approximate contour polygon using Ramer-Douglas-Peucker algorithm (epsilon = 2% of arc length)
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Check if the approximated contour has 4 vertices (quadrilateral candidate)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = float(w) / h if h > 0 else 0

            # Filter by typical license plate aspect ratio (typically 2:1 to 4:1)
            if 2 < aspect_ratio < 4:
                # Draw a green bounding rectangle around the candidate plate
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return image


def extract_text(image: np.ndarray) -> str:
    """
    Extracts text from a candidate image using Tesseract OCR.

    Args:
        image: Candidate image array.

    Returns:
        Recognized text string.
    """
    # Convert image to grayscale for OCR engine
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

    # Run Tesseract OCR on the grayscale image
    text = pytesseract.image_to_string(gray)

    return text.strip()


# Example demonstration (only executed when run directly as a script)
if __name__ == "__main__":
    image_path = 'path_to_image.jpg'
    try:
        detected_image = detect_number_plate(image_path)
        cv2.imshow('Number Plate Detection', detected_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        extracted_text = extract_text(detected_image)
        print("Extracted Text:", extracted_text)
    except Exception as e:
        print(f"Demo run note: {e}")
