"""
Image Service for License Plate Recognition.
============================================

Provides single-image license plate detection and text recognition
with dual-stage plate localization:
1. Primary Stage: YOLOv8 deep learning plate detector.
2. Fallback Stage: OpenCV morphological contour & aspect ratio analysis.
3. Preprocessing: CLAHE-enhanced contrast normalization.
4. Recognition: PaddleOCR text recognition and heuristic Indian plate matching.
"""

import cv2
import numpy as np
import re
from pathlib import Path
from fastapi import UploadFile
from paddleocr import PaddleOCR
from ultralytics import YOLO

# Recognized Indian State & Union Territory prefix codes
STATE_CODES = {
    "AP", "AR", "AS", "BR", "CG", "DL", "DN", "GA", "GJ", "HP",
    "HR", "JH", "JK", "KA", "KL", "LD", "MH", "ML", "MN", "MP",
    "MZ", "NL", "OD", "PB", "PY", "RJ", "SK", "TN", "TR", "TS",
    "UK", "UP", "WB"
}

# ==============================================================================
# Model Singletons (Lazy Loading)
# ==============================================================================
_detector = None
_ocr_engine = None


def get_detector():
    """
    Lazy-loads and returns the singleton YOLOv8 model instance.
    Prevents redundant model reloads across requests.
    """
    global _detector
    if _detector is None:
        model_path = Path(__file__).resolve().parent.parent / "yolo_data" / "plate_detector" / "weights" / "best.pt"
        _detector = YOLO(str(model_path))
    return _detector


def get_ocr():
    """
    Lazy-loads and returns the singleton PaddleOCR engine instance
    with tuned detection thresholds for license plates.
    """
    global _ocr_engine
    if _ocr_engine is None:
        _ocr_engine = PaddleOCR(
            use_textline_orientation=True,
            lang="en",
            text_det_thresh=0.15,
            text_det_box_thresh=0.2
        )
    return _ocr_engine


# ==============================================================================
# Plate Crop Preprocessing
# ==============================================================================
def preprocess(img, target_h=128):
    """
    Normalizes cropped plate image for OCR.
    
    Steps:
    1. Proportional resize to fixed target height (128px) via bilinear interpolation.
    2. Convert to grayscale luminance.
    3. CLAHE (Contrast Limited Adaptive Histogram Equalization) with (4x4) tile grid
       to reveal faint letters without blowing out highlights.
    """
    h, w = img.shape[:2]
    if h == 0:
        return img
    scale = target_h / h
    new_w = max(int(w * scale), target_h)
    resized = cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_LINEAR)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY) if len(resized.shape) == 3 else resized
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    return clahe.apply(gray)


# ==============================================================================
# Indian Plate Pattern Scoring
# ==============================================================================
def best_plate_from_tokens(tokens):
    """
    Selects the candidate token that best matches Indian plate registration syntax.
    
    Evaluates:
    - Length score (penalizes deviation from standard 10-char plate).
    - State prefix bonus (+5 for matching recognized RTO codes).
    - Full syntax regex bonus (+15 for matching State + District + Series + Number).
    """
    scored = []
    for t in tokens:
        t_clean = re.sub(r"[^A-Z0-9]", "", t.upper())
        if not t_clean:
            continue
        score = 0
        if 7 <= len(t_clean) <= 13:
            score += 10 - abs(len(t_clean) - 10)
            if t_clean[:2] in STATE_CODES:
                score += 5
            if re.match(r"^[A-Z]{2}[0-9]{1,2}[A-Z]*[0-9]*$", t_clean):
                score += 15
        scored.append((t_clean, score))
    if not scored:
        return ""
    return max(scored, key=lambda x: x[1])[0]


# ==============================================================================
# Async Image Processing Endpoint Handler
# ==============================================================================
async def process_image(file: UploadFile) -> dict:
    """
    Processes an uploaded image file from FastAPI request.
    
    Workflow:
    1. Decode file bytes to OpenCV image matrix.
    2. Primary Detection: Run YOLOv8 to locate license plate box with highest confidence.
    3. Fallback Detection: If YOLO yields no box, run Canny edge detection and contour
       analysis looking for quadrilaterals with plate-like aspect ratios (2.0 to 3.5).
    4. Preprocess cropped plate with CLAHE.
    5. Run PaddleOCR text recognition and heuristic format matching.
    """
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # 1. Primary: YOLO plate detection
    detector = get_detector()
    det = detector.predict(image, conf=0.25, verbose=False)
    boxes = det[0].boxes if det and det[0] is not None and len(det[0].boxes) > 0 else None

    plate_crop = None
    if boxes is not None and len(boxes) > 0:
        # Pick the bounding box with the highest confidence score
        best_idx = boxes.conf.argmax().item()
        x1, y1, x2, y2 = boxes.xyxy[best_idx].cpu().numpy().astype(int)
        # Add safety margin around plate boundary
        margin = 3
        x1, y1 = max(0, x1 - margin), max(0, y1 - margin)
        x2, y2 = min(image.shape[1], x2 + margin), min(image.shape[0], y2 + margin)
        plate_crop = image[y1:y2, x1:x2]

    # 2. Fallback: Classical OpenCV contour analysis if YOLO misses
    if plate_crop is None or plate_crop.size == 0:
        h, w = image.shape[:2]
        min_area = (h * w) * 0.005
        max_area = (h * w) * 0.4
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 75, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area or area > max_area:
                continue
            # Approximate polygonal curve
            epsilon = 0.03 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            # Check for quadrilateral geometry (4 vertices)
            if len(approx) == 4:
                x, y, cw, ch = cv2.boundingRect(approx)
                ar = float(cw) / ch if ch > 0 else 0
                # License plates typically have aspect ratios between 2:1 and 3.5:1
                if 2.0 <= ar <= 3.5:
                    candidates.append((area, (x, y, cw, ch)))
        if candidates:
            # Sort by area descending to find the most prominent rectangular plate candidate
            candidates.sort(key=lambda c: c[0], reverse=True)
            x, y, cw, ch = candidates[0][1]
            plate_crop = gray[y:y+ch, x:x+cw]

    if plate_crop is None or plate_crop.size == 0:
        return {"text": "", "success": False, "error": "No plate detected"}

    # 3. Preprocess and enhance cropped plate
    normalized = preprocess(plate_crop, 128)
    bgr = cv2.cvtColor(normalized, cv2.COLOR_GRAY2BGR)

    # 4. PaddleOCR inference on enhanced plate
    try:
        ocr = get_ocr()
        result = ocr.predict(bgr)
        tokens = []
        for res in result:
            data = res.json["res"]
            tokens.extend(data.get("rec_texts", []))
        text = best_plate_from_tokens(tokens)
    except Exception as e:
        return {"text": "", "success": False, "error": str(e)}

    return {"text": text.strip(), "success": True}
