import cv2
import numpy as np
import pytesseract

def detect_number_plate(image_path):
    # Load the image
    image = cv2.imread(image_path)
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian Blur for noise reduction
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Use Canny Edge Detection
    edges = cv2.Canny(blurred, 50, 150)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        # Approximate the contour
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        if len(approx) == 4:  # Check if it's a quadrilateral
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = float(w) / h
            
            # Check for typical number plate aspect ratio (e.g., 3:1)
            if 2 < aspect_ratio < 4:
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    return image

def extract_text(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Use Tesseract for OCR
    text = pytesseract.image_to_string(gray)

    return text

# Example usage
image_path = 'path_to_image.jpg'
detected_image = detect_number_plate(image_path)
cv2.imshow('Number Plate Detection', detected_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

extracted_text = extract_text(detected_image)
print("Extracted Text:", extracted_text)
