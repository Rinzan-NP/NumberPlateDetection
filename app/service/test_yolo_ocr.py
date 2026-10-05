"""
YOLO detector + PaddleOCR pipeline — tested on the cropped test images.
Uses pre-cropped plates (which are already the plate regions) so OCR is the bottleneck.
"""
import csv
import re
from pathlib import Path
import cv2
from paddleocr import PaddleOCR
from ultralytics import YOLO

DATA_DIR = Path(__file__).parent.parent / "data"
TEST_DIR = DATA_DIR / "test"
LABELS = DATA_DIR / "test_labels.csv"
YOLO_MODEL = Path(__file__).parent.parent / "yolo_data" / "plate_detector" / "weights" / "best.pt"

STATE_CODES = {
    "AP", "AR", "AS", "BR", "CG", "DL", "DN", "GA", "GJ", "HP",
    "HR", "JH", "JK", "KA", "KL", "LD", "MH", "ML", "MN", "MP",
    "MZ", "NL", "OD", "PB", "PY", "RJ", "SK", "TN", "TR", "TS",
    "UK", "UP", "WB"
}


def preprocess(img, target_h=128):
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


def ocr_from_crop(img, ocr_engine):
    """Run PaddleOCR on a plate crop with preprocessing."""
    gray = preprocess(img, 128)
    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    result = ocr_engine.predict(bgr)
    tokens = []
    for res in result:
        data = res.json["res"]
        tokens.extend(data.get("rec_texts", []))
    return best_plate_from_tokens(tokens)


def main():
    print("Loading PaddleOCR...")
    ocr = PaddleOCR(use_textline_orientation=True, lang="en",
                     text_det_thresh=0.15, text_det_box_thresh=0.2)
    print("Ready.\n")

    # Load test labels
    with open(LABELS) as f:
        samples = list(csv.DictReader(f))

    correct = 0
    total = len(samples)
    results = []

    for row in samples:
        img_path = TEST_DIR / row["filename"]
        if not img_path.exists():
            continue
        img = cv2.imread(str(img_path))
        pred = ocr_from_crop(img, ocr)
        gt = re.sub(r"[^A-Z0-9]", "", row["text"].upper())
        match = pred == gt
        if match:
            correct += 1
        results.append((row["filename"], row["text"], pred, match))

    acc = correct / total * 100 if total else 0
    print(f"\n=== Results (PaddleOCR on cropped plates) ===")
    print(f"Accuracy: {correct}/{total} = {acc:.1f}%")
    print(f"\n{'File':<35} {'GT':<12} {'Pred':<12} {'OK'}")
    print("-" * 70)
    for fname, gt, pred, ok in results:
        print(f"{fname:<35} {gt:<12} {pred:<12} {'✓' if ok else '✗'}")

    # Now also try YOLO detection + OCR on full original images
    print("\n\n=== YOLO + OCR on original full images ===")
    print("(Using ground truth crop bboxes to extract crops, for fair OCR comparison)")
    import random, xml.etree.ElementTree as ET
    random.seed(42)
    ARCHIVE = Path(__file__).resolve().parent.parent.parent.parent / "archive"
    OCR_ANNOS = ARCHIVE / "number_plate_annos_ocr" / "number_plate_annos_ocr"
    OCR_IMAGES = ARCHIVE / "number_plate_images_ocr" / "number_plate_images_ocr"

    all_samples = []
    for xml_file in OCR_ANNOS.glob("*.xml"):
        img_name = xml_file.stem + ".jpg"
        img_path = OCR_IMAGES / img_name
        if not img_path.exists():
            continue
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            for obj in root.findall(".//object"):
                text = None
                for attr in obj.findall(".//attributes/attribute"):
                    if attr.find("name").text == "number_plate_text":
                        text = attr.find("value").text
                        break
                if not text:
                    continue
                bndbox = obj.find("bndbox")
                all_samples.append({
                    "image_path": str(img_path),
                    "text": text.strip(),
                    "xmin": float(bndbox.find("xmin").text),
                    "ymin": float(bndbox.find("ymin").text),
                    "xmax": float(bndbox.find("xmax").text),
                    "ymax": float(bndbox.find("ymax").text),
                })
        except Exception:
            continue

    random.shuffle(all_samples)
    test_orig = all_samples[int(len(all_samples) * 0.7):]

    yolo = YOLO(str(YOLO_MODEL))
    yolo_correct = 0
    yolo_total = len(test_orig)
    yolo_results = []

    for row in test_orig:
        img = cv2.imread(row["image_path"])
        # Use ground truth bbox for crop (so we measure OCR quality, not detection)
        x1, y1 = int(row["xmin"]), int(row["ymin"])
        x2, y2 = int(row["xmax"]), int(row["ymax"])
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        pred = ocr_from_crop(crop, ocr)
        gt = re.sub(r"[^A-Z0-9]", "", row["text"].upper())
        match = pred == gt
        if match:
            yolo_correct += 1
        yolo_results.append((Path(row["image_path"]).name, row["text"], pred, match))

    yolo_acc = yolo_correct / yolo_total * 100 if yolo_total else 0
    print(f"Accuracy: {yolo_correct}/{yolo_total} = {yolo_acc:.1f}%")
    print(f"\n{'File':<45} {'GT':<12} {'Pred':<12} {'OK'}")
    print("-" * 80)
    for fname, gt, pred, ok in yolo_results:
        print(f"{fname:<45} {gt:<12} {pred:<12} {'✓' if ok else '✗'}")


if __name__ == "__main__":
    main()
