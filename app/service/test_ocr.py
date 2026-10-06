"""
PaddleOCR Standalone Benchmark & Evaluation Script.
===================================================

This script benchmarks the character recognition performance of PaddleOCR
on cropped license plate images using:
1. CLAHE-based image normalization.
2. Rule-based heuristic scoring for Indian license plate formats.
3. Ground-truth evaluation against test_labels.csv.
"""

import csv
import re
from pathlib import Path
import cv2
from paddleocr import PaddleOCR

# ==============================================================================
# Paths Configuration
# ==============================================================================
DATA_DIR = Path(__file__).parent.parent / "data"
TEST_DIR = DATA_DIR / "test"
LABELS = DATA_DIR / "test_labels.csv"

# Recognized 2-letter state & union territory codes across India
STATE_CODES = {
    "AP", "AR", "AS", "BR", "CG", "DL", "DN", "GA", "GJ", "HP",
    "HR", "JH", "JK", "KA", "KL", "LD", "MH", "ML", "MN", "MP",
    "MZ", "NL", "OD", "PB", "PY", "RJ", "SK", "TN", "TR", "TS",
    "UK", "UP", "WB"
}


# ==============================================================================
# Preprocessing / Normalization
# ==============================================================================
def normalize(img, target_h=128):
    """
    Normalizes a cropped license plate image for character recognition.

    Steps:
    1. Aspect-ratio preserving resize to a fixed target height (128px).
    2. Convert to single-channel grayscale to eliminate chromatic noise.
    3. Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
       with tileGridSize=(4,4) to enhance character stroke definition.
    """
    h, w = img.shape[:2]
    if h == 0:
        return img

    # Scale width proportionally to maintain aspect ratio
    scale = target_h / h
    new_w = max(int(w * scale), target_h)
    resized = cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_LINEAR)

    # Convert to grayscale luminance
    if len(resized.shape) == 3:
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    else:
        gray = resized

    # Apply CLAHE to balance uneven illumination
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    return clahe.apply(gray)


# ==============================================================================
# Token Scoring & Plate Selection
# ==============================================================================
def best_plate_from_tokens(tokens):
    """
    Selects the candidate token that best matches standard Indian license plate structure.

    Scoring Heuristics:
    - Length between 7 and 13 characters: +Score (penalty proportional to distance from 10 chars).
    - Starts with valid 2-letter Indian state code: +5 points.
    - Matches standard regex pattern (e.g. MH12DE1433): +15 points.

    Args:
        tokens: List of raw text tokens from OCR.

    Returns:
        Highest-scoring plate string candidate.
    """
    scored = []
    for t in tokens:
        # Strip all punctuation and whitespace, keep uppercase alphanumeric
        t_clean = re.sub(r"[^A-Z0-9]", "", t.upper())
        if not t_clean:
            continue

        score = 0
        # Typical Indian plate length is 10 characters (e.g. DL01AB1234)
        if 7 <= len(t_clean) <= 13:
            score += 10 - abs(len(t_clean) - 10)
            # Bonus for valid state code prefix
            if t_clean[:2] in STATE_CODES:
                score += 5
            # Bonus for complete structural regex match
            if re.match(r"^[A-Z]{2}[0-9]{1,2}[A-Z]*[0-9]*$", t_clean):
                score += 15

        scored.append((t_clean, score))

    if not scored:
        return ""

    # Return the token with highest heuristic score
    return max(scored, key=lambda x: x[1])[0]


# ==============================================================================
# Benchmark Evaluation Runner
# ==============================================================================
def main():
    """Runs OCR evaluation against the labeled test dataset and prints an accuracy report."""
    print("Loading PaddleOCR...")
    # Initialize PaddleOCR with lower detection thresholds to catch faint letters
    ocr = PaddleOCR(
        use_textline_orientation=True,
        lang="en",
        text_det_thresh=0.15,
        text_det_box_thresh=0.2
    )
    print("Ready.\n")

    # Load test dataset annotations
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

        # Load and preprocess image
        img = cv2.imread(str(img_path))
        gray = normalize(img, 128)
        bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        # Run OCR inference
        ocr_result = ocr.predict(bgr)
        tokens = []
        for res in ocr_result:
            data = res.json["res"]
            tokens.extend(data.get("rec_texts", []))

        # Select best candidate and compare with ground truth
        pred = best_plate_from_tokens(tokens)
        gt = re.sub(r"[^A-Z0-9]", "", row["text"].upper())
        match = pred == gt
        if match:
            correct += 1

        results.append((row["filename"], row["text"], tokens, pred, match))

    # Print summary performance metrics
    acc = correct / total * 100 if total else 0
    print(f"\n=== Results ===")
    print(f"Accuracy: {correct}/{total} = {acc:.1f}%")
    print(f"\n{'File':<35} {'GT':<12} {'Tokens':<35} {'Pred':<12} {'OK'}")
    print("-" * 100)
    for fname, gt, tokens, pred, ok in results:
        print(f"{fname:<35} {gt:<12} {str(tokens):<35} {pred:<12} {'✓' if ok else '✗'}")


if __name__ == "__main__":
    main()
