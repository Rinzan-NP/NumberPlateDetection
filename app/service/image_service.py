import cv2
import numpy as np
import re
from pathlib import Path
from fastapi import UploadFile
from paddleocr import PaddleOCR
from ultralytics import YOLO

STATE_CODES = {
    "AP", "AR", "AS", "BR", "CG", "DL", "DN", "GA", "GJ", "HP",
    "HR", "JH", "JK", "KA", "KL", "LD", "MH", "ML", "MN", "MP",
    "MZ", "NL", "OD", "PB", "PY", "RJ", "SK", "TN", "TR", "TS",
    "UK", "UP", "WB"
}

# Lazy-load engines
_detector = None
_ocr_engine = None


def get_detector():
    global _detector
    if _detector is None:
        model_path = Path(__file__).resolve().parent.parent / "yolo_data" / "plate_detector" / "weights" / "best.pt"
        _detector = YOLO(str(model_path))
    return _detector


def get_ocr():
    global _ocr_engine
    if _ocr_engine is None:
        _ocr_engine = PaddleOCR(use_textline_orientation=True, lang="en",
                                 text_det_thresh=0.15, text_det_box_thresh=0.2)
    return _ocr_engine


def preprocess(img, target_h=128):
    """Normalize plate crop for OCR."""
    h, w = img.shape[:2]
    if h == 0:
        return img
    scale = target_h / h
    new_w = max(int(w * scale), target_h)
    resized = cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_LINEAR)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY) if len(resized.shape) == 3 else resized
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    return clahe.apply(gray)


def best_plate_from_tokens(tokens):
    """Select token best matching Indian plate pattern."""
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


async def process_image(file: UploadFile) -> dict:
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # 1. YOLO plate detection
    detector = get_detector()
    det = detector.predict(image, conf=0.25, verbose=False)
    boxes = det[0].boxes if det and det[0] is not None and len(det[0].boxes) > 0 else None

    plate_crop = None
    if boxes is not None and len(boxes) > 0:
        best_idx = boxes.conf.argmax().item()
        x1, y1, x2, y2 = boxes.xyxy[best_idx].cpu().numpy().astype(int)
        margin = 3
        x1, y1 = max(0, x1 - margin), max(0, y1 - margin)
        x2, y2 = min(image.shape[1], x2 + margin), min(image.shape[0], y2 + margin)
        plate_crop = image[y1:y2, x1:x2]

    # Fallback: contour detection if YOLO misses
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
            epsilon = 0.03 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            if len(approx) == 4:
                x, y, cw, ch = cv2.boundingRect(approx)
                ar = float(cw) / ch if ch > 0 else 0
                if 2.0 <= ar <= 3.5:
                    candidates.append((area, (x, y, cw, ch)))
        if candidates:
            candidates.sort(key=lambda c: c[0], reverse=True)
            x, y, cw, ch = candidates[0][1]
            plate_crop = gray[y:y+ch, x:x+cw]

    if plate_crop is None or plate_crop.size == 0:
        return {"text": "", "success": False, "error": "No plate detected"}

    # 2. OCR on plate crop
    normalized = preprocess(plate_crop, 128)
    bgr = cv2.cvtColor(normalized, cv2.COLOR_GRAY2BGR)

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
