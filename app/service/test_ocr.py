"""
PaddleOCR-only test with tuned preprocessing + Indian plate pattern matching.
"""
import csv
import re
from pathlib import Path
import cv2
from paddleocr import PaddleOCR

DATA_DIR = Path(__file__).parent.parent / "data"
TEST_DIR = DATA_DIR / "test"
LABELS = DATA_DIR / "test_labels.csv"

STATE_CODES = {
    "AP", "AR", "AS", "BR", "CG", "DL", "DN", "GA", "GJ", "HP",
    "HR", "JH", "JK", "KA", "KL", "LD", "MH", "ML", "MN", "MP",
    "MZ", "NL", "OD", "PB", "PY", "RJ", "SK", "TN", "TR", "TS",
    "UK", "UP", "WB"
}


def normalize(img, target_h=128):
    h, w = img.shape[:2]
    if h == 0:
        return img
    scale = target_h / h
    new_w = max(int(w * scale), target_h)
    resized = cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_LINEAR)
    if len(resized.shape) == 3:
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    else:
        gray = resized
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    return clahe.apply(gray)


def best_plate_from_tokens(tokens):
    """Select token best matching Indian plate: 2-letter state code + digits + letters + digits."""
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


def main():
    print("Loading PaddleOCR...")
    ocr = PaddleOCR(use_textline_orientation=True, lang="en",
                     text_det_thresh=0.15, text_det_box_thresh=0.2)
    print("Ready.\n")

    with open(LABELS) as f:
        reader_csv = csv.DictReader(f)
        samples = list(reader_csv)

    correct = 0
    total = len(samples)
    results = []

    for row in samples:
        img_path = TEST_DIR / row["filename"]
        if not img_path.exists():
            continue

        img = cv2.imread(str(img_path))
        gray = normalize(img, 128)
        bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        ocr_result = ocr.predict(bgr)
        tokens = []
        for res in ocr_result:
            data = res.json["res"]
            tokens.extend(data.get("rec_texts", []))

        pred = best_plate_from_tokens(tokens)
        gt = re.sub(r"[^A-Z0-9]", "", row["text"].upper())
        match = pred == gt
        if match:
            correct += 1
        results.append((row["filename"], row["text"], tokens, pred, match))

    acc = correct / total * 100 if total else 0
    print(f"\n=== Results ===")
    print(f"Accuracy: {correct}/{total} = {acc:.1f}%")
    print(f"\n{'File':<35} {'GT':<12} {'Tokens':<35} {'Pred':<12} {'OK'}")
    print("-" * 100)
    for fname, gt, tokens, pred, ok in results:
        print(f"{fname:<35} {gt:<12} {str(tokens):<35} {pred:<12} {'✓' if ok else '✗'}")


if __name__ == "__main__":
    main()
