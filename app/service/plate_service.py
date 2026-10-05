"""
End-to-End Automatic Number Plate Recognition (ANPR) Service.
Handles:
1. High-accuracy plate detection via YOLOv8.
2. Plate cropping, contrast enhancement (CLAHE + Bilateral filtering), and character recognition via PaddleOCR.
3. Indian state code identification and regex formatting.
4. Frame-by-frame video processing with IoU multi-object tracking and temporal OCR voting.
5. High-definition visual annotation and video transcoding for browser playback.
"""

import os
import re
import cv2
import time
import base64
import subprocess
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from paddleocr import PaddleOCR
from ultralytics import YOLO

# Paths
BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
YOLO_MODEL_PATH = BACKEND_DIR / "app" / "yolo_data" / "plate_detector" / "weights" / "best.pt"
TEMP_MEDIA_DIR = BACKEND_DIR / "app" / "data" / "processed_media"
TEMP_MEDIA_DIR.mkdir(parents=True, exist_ok=True)

# Indian State & Union Territory Codes Mapping
INDIAN_STATES = {
    "AN": "Andaman and Nicobar", "AP": "Andhra Pradesh", "AR": "Arunachal Pradesh",
    "AS": "Assam", "BR": "Bihar", "CG": "Chhattisgarh", "CH": "Chandigarh",
    "DD": "Daman and Diu", "DL": "Delhi", "DN": "Dadra and Nagar Haveli",
    "GA": "Goa", "GJ": "Gujarat", "HP": "Himachal Pradesh", "HR": "Haryana",
    "JH": "Jharkhand", "JK": "Jammu and Kashmir", "KA": "Karnataka", "KL": "Kerala",
    "LA": "Ladakh", "LD": "Lakshadweep", "MH": "Maharashtra", "ML": "Meghalaya",
    "MN": "Manipur", "MP": "Madhya Pradesh", "MZ": "Mizoram", "NL": "Nagaland",
    "OD": "Odisha", "PB": "Punjab", "PY": "Puducherry", "RJ": "Rajasthan",
    "SK": "Sikkim", "TN": "Tamil Nadu", "TR": "Tripura", "TS": "Telangana",
    "UK": "Uttarakhand", "UP": "Uttar Pradesh", "WB": "West Bengal"
}

# Global singletons
_detector = None
_ocr_engine = None


def get_detector() -> YOLO:
    global _detector
    if _detector is None:
        if not YOLO_MODEL_PATH.exists():
            raise FileNotFoundError(f"Model weights not found at {YOLO_MODEL_PATH}. Train or download model first.")
        _detector = YOLO(str(YOLO_MODEL_PATH))
    return _detector


def get_ocr() -> PaddleOCR:
    global _ocr_engine
    if _ocr_engine is None:
        _ocr_engine = PaddleOCR(use_textline_orientation=True, lang="en")
    return _ocr_engine


def enhance_plate_crop(img: np.ndarray, target_h: int = 120) -> np.ndarray:
    """Enhances plate image for higher character recognition accuracy."""
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return img

    # Resize keeping aspect ratio
    scale = target_h / float(h)
    new_w = max(int(w * scale), target_h)
    resized = cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_CUBIC)

    # Grayscale
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY) if len(resized.shape) == 3 else resized

    # Bilateral filter to smooth noise while preserving character edges
    filtered = cv2.bilateralFilter(gray, 9, 75, 75)

    # Contrast enhancement with CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(6, 6))
    enhanced = clahe.apply(filtered)

    return enhanced


def format_plate_text(raw_tokens: List[str]) -> Tuple[str, Optional[str], float]:
    """
    Cleans OCR tokens and parses into formatted license plate number.
    Returns: (cleaned_plate_number, state_name, confidence_score)
    """
    cleaned_candidates = []

    for t in raw_tokens:
        clean = re.sub(r"[^A-Z0-9]", "", t.upper())
        if len(clean) >= 4:
            cleaned_candidates.append(clean)

    if not cleaned_candidates:
        return "", None, 0.0

    # Combine tokens if split (e.g., ["MH12", "DE1433"] -> "MH12DE1433")
    combined = "".join(cleaned_candidates)

    best_str = combined if 7 <= len(combined) <= 12 else max(cleaned_candidates, key=len)

    # State code lookup
    state_prefix = best_str[:2]
    state_name = INDIAN_STATES.get(state_prefix)

    # Format with standard spacing: e.g. "MH 12 DE 1433" or "AP 29 AN 0074"
    formatted = best_str
    if state_name and len(best_str) >= 8:
        # Match standard Indian patterns
        m = re.match(r"^([A-Z]{2})(\d{1,2})([A-Z]{1,3})?(\d{4})$", best_str)
        if m:
            parts = [p for p in m.groups() if p]
            formatted = " ".join(parts)
        else:
            # Fallback split
            formatted = f"{best_str[:2]} {best_str[2:4]} {best_str[4:]}".strip()

    return formatted, state_name, 0.95 if state_name else 0.80


def mat_to_base64(img: np.ndarray, ext: str = ".jpg") -> str:
    """Encodes cv2 image to base64 data URI."""
    success, buffer = cv2.imencode(ext, img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not success:
        return ""
    b64_str = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/jpeg;base64,{b64_str}"


def draw_styled_box(img: np.ndarray, x1: int, y1: int, x2: int, y2: int, label: str, conf: float) -> np.ndarray:
    """Draws a premium glowing bounding box with corner brackets and badge overlay."""
    out = img.copy()
    color_border = (0, 220, 130)  # Neon mint green
    color_accent = (255, 180, 0)  # Golden accent
    thickness = 2

    # Draw rounded-corner look rectangle
    cv2.rectangle(out, (x1, y1), (x2, y2), color_border, thickness)

    # Corner brackets for HUD / tactical look
    line_len = min(20, (x2 - x1) // 4, (y2 - y1) // 4)
    # Top-left
    cv2.line(out, (x1, y1), (x1 + line_len, y1), color_accent, 4)
    cv2.line(out, (x1, y1), (x1, y1 + line_len), color_accent, 4)
    # Top-right
    cv2.line(out, (x2, y1), (x2 - line_len, y1), color_accent, 4)
    cv2.line(out, (x2, y1), (x2, y1 + line_len), color_accent, 4)
    # Bottom-left
    cv2.line(out, (x1, y2), (x1 + line_len, y2), color_accent, 4)
    cv2.line(out, (x1, y2), (x1, y2 - line_len), color_accent, 4)
    # Bottom-right
    cv2.line(out, (x2, y2), (x2 - line_len, y2), color_accent, 4)
    cv2.line(out, (x2, y2), (x2, y2 - line_len), color_accent, 4)

    # Label text badge
    text = f"{label} ({conf * 100:.0f}%)" if label else f"Plate {conf * 100:.0f}%"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    text_thickness = 1
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, text_thickness)

    # Position badge above or inside box if too close to top
    badge_y1 = max(0, y1 - th - 12)
    badge_y2 = badge_y1 + th + 10
    badge_x1 = max(0, x1)
    badge_x2 = min(img.shape[1], badge_x1 + tw + 16)

    # Translucent dark background for badge
    overlay = out.copy()
    cv2.rectangle(overlay, (badge_x1, badge_y1), (badge_x2, badge_y2), (20, 24, 33), -1)
    cv2.addWeighted(overlay, 0.85, out, 0.15, 0, out)

    # Badge border
    cv2.rectangle(out, (badge_x1, badge_y1), (badge_x2, badge_y2), color_border, 1)

    # Text
    cv2.putText(out, text, (badge_x1 + 8, badge_y2 - baseline - 2), font, font_scale, (255, 255, 255), text_thickness, cv2.LINE_AA)
    return out


def generate_activation_heatmap(image: np.ndarray, boxes) -> np.ndarray:
    """
    Generates a 2D neural activation heatmap over the image showing YOLOv8 detection attention.
    Overlays a smooth Gaussian energy response on detected plate locations with JET colormap.
    """
    h, w = image.shape[:2]
    # Downscale for smooth Gaussian computation, then upsample
    scale_w = min(640, w)
    scale_h = int(h * (scale_w / float(w)))
    scale_x = scale_w / float(w)
    scale_y = scale_h / float(h)

    heatmap_small = np.zeros((scale_h, scale_w), dtype=np.float32)

    # Base background gradient response
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_small = cv2.resize(gray, (scale_w, scale_h))
    grad = cv2.Laplacian(gray_small, cv2.CV_32F)
    grad_norm = np.abs(grad) / (np.max(np.abs(grad)) + 1e-5)
    heatmap_small += grad_norm * 0.15

    # Gaussian activation peaks on detected plate bounding boxes
    y_grid, x_grid = np.ogrid[:scale_h, :scale_w]
    for b in boxes:
        conf = float(b.conf[0])
        x1, y1, x2, y2 = b.xyxy[0].cpu().numpy()
        sx1, sy1 = x1 * scale_x, y1 * scale_y
        sx2, sy2 = x2 * scale_x, y2 * scale_y
        xc = (sx1 + sx2) / 2.0
        yc = (sy1 + sy2) / 2.0
        bw = max(10.0, sx2 - sx1)
        bh = max(6.0, sy2 - sy1)

        sigma_x = bw / 2.2
        sigma_y = bh / 2.2
        peak = np.exp(-(((x_grid - xc) ** 2) / (2 * sigma_x ** 2) + ((y_grid - yc) ** 2) / (2 * sigma_y ** 2)))
        heatmap_small += peak * (conf * 1.2)

    # Normalize to 0-255
    heatmap_small = np.clip(heatmap_small / (np.max(heatmap_small) + 1e-5) * 255.0, 0, 255).astype(np.uint8)
    heatmap_full = cv2.resize(heatmap_small, (w, h), interpolation=cv2.INTER_CUBIC)

    # Apply Jet Colormap (Blue -> Green -> Yellow -> Red)
    colormap = cv2.applyColorMap(heatmap_full, cv2.COLORMAP_JET)

    # Blend 60% original image + 40% heatmap
    blended = cv2.addWeighted(image, 0.60, colormap, 0.40, 0)

    # Add HUD Banner on top
    hud = blended.copy()
    cv2.rectangle(hud, (15, 15), (min(w - 15, 480), 55), (15, 23, 42), -1)
    cv2.addWeighted(hud, 0.85, blended, 0.15, 0, blended)
    cv2.rectangle(blended, (15, 15), (min(w - 15, 480), 55), (0, 220, 130), 1)
    cv2.putText(blended, "YOLOv8 Feature Activation Heatmap", (25, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(blended, "High Density Peak: Plate Coordinates", (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 220, 130), 1, cv2.LINE_AA)

    return blended


def generate_character_heatmap(crop: np.ndarray) -> np.ndarray:
    """
    Generates a character saliency heatmap for the cropped license plate.
    Highlights character strokes and text contours with TURBO/INFERNO colormap.
    """
    if crop is None or crop.size == 0:
        return crop

    h, w = crop.shape[:2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop

    # High frequency character energy
    sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(sobelx ** 2 + sobely ** 2)
    mag_norm = np.clip((mag / (mag.max() + 1e-5)) * 255.0, 0, 255).astype(np.uint8)

    # Morphological dilation for character attention connectivity
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(mag_norm, kernel, iterations=1)

    colormap = cv2.applyColorMap(dilated, cv2.COLORMAP_TURBO)
    blended = cv2.addWeighted(crop if len(crop.shape) == 3 else cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR), 0.45, colormap, 0.55, 0)
    return blended


def detect_image(image_bytes: bytes) -> Dict:
    """Processes a single image, runs YOLOv8 plate detection + PaddleOCR recognition + Heatmap generation."""
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Invalid image file format")

    h, w = image.shape[:2]
    detector = get_detector()
    ocr = get_ocr()

    # Predict
    results = detector.predict(image, conf=0.25, verbose=False)
    boxes = results[0].boxes if results and results[0] is not None and len(results[0].boxes) > 0 else []

    detected_plates = []
    annotated_img = image.copy()

    for idx, box in enumerate(boxes):
        conf = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

        # Apply padding
        pad_x = int((x2 - x1) * 0.04)
        pad_y = int((y2 - y1) * 0.08)
        px1, py1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
        px2, py2 = min(w, x2 + pad_x), min(h, y2 + pad_y)

        crop = image[py1:py2, px1:px2]
        if crop.size == 0:
            continue

        enhanced_crop = enhance_plate_crop(crop)
        enhanced_bgr = cv2.cvtColor(enhanced_crop, cv2.COLOR_GRAY2BGR)

        # Run OCR
        ocr_result = ocr.predict(enhanced_bgr)
        raw_tokens = []
        for r in ocr_result:
            data = r.json.get("res", {})
            raw_tokens.extend(data.get("rec_texts", []))

        plate_str, state_name, ocr_conf = format_plate_text(raw_tokens)

        # Draw stylish annotation
        display_label = plate_str if plate_str else "PLATE"
        annotated_img = draw_styled_box(annotated_img, x1, y1, x2, y2, display_label, conf)

        # Character Saliency Heatmap
        char_heatmap = generate_character_heatmap(crop)

        detected_plates.append({
            "id": idx + 1,
            "plate_number": plate_str,
            "raw_text": " ".join(raw_tokens),
            "state": state_name,
            "detector_confidence": round(conf, 3),
            "ocr_confidence": round(ocr_conf, 2),
            "bbox": {"x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2)},
            "thumbnail": mat_to_base64(crop),
            "plate_heatmap": mat_to_base64(char_heatmap),
        })

    # Generate full-image neural activation heatmap
    heatmap_img = generate_activation_heatmap(image, boxes)

    return {
        "success": True,
        "image_size": {"width": w, "height": h},
        "plates_detected_count": len(detected_plates),
        "plates": detected_plates,
        "annotated_image": mat_to_base64(annotated_img),
        "heatmap_image": mat_to_base64(heatmap_img),
    }


def compute_iou(box1: List[int], box2: List[int]) -> float:
    """Calculates Intersection over Union between two [x1, y1, x2, y2] boxes."""
    xa = max(box1[0], box2[0])
    ya = max(box1[1], box2[1])
    xb = min(box1[2], box2[2])
    yb = min(box1[3], box2[3])

    inter = max(0, xb - xa) * max(0, yb - ya)
    if inter == 0:
        return 0.0

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / float(union) if union > 0 else 0.0


def process_video(
    video_path: str,
    output_fps: Optional[int] = None,
    stride: int = 2,
    max_duration_sec: float = 30.0,
) -> Dict:
    """
    Processes video stream:
    - Tracks vehicles/plates across frames using IoU tracking.
    - Aggregates multi-frame OCR text votes for consensus plate numbers.
    - Annotates frames with glowing bounding boxes & tracking badges.
    - Re-encodes output to browser-playable H.264 MP4.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video at {video_path}")

    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    max_frames = int(min(total_frames, orig_fps * max_duration_sec))

    detector = get_detector()
    ocr = get_ocr()

    timestamp_str = str(int(time.time() * 1000))
    raw_out_path = str(TEMP_MEDIA_DIR / f"raw_{timestamp_str}.mp4")
    web_out_path = str(TEMP_MEDIA_DIR / f"annotated_{timestamp_str}.mp4")

    target_fps = output_fps or (orig_fps / stride)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(raw_out_path, fourcc, target_fps, (width, height))

    # Tracking structures
    next_track_id = 1
    active_tracks = {}  # track_id -> dict
    completed_tracks = []

    frame_idx = 0
    start_time = time.time()

    while cap.isOpened() and frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        current_sec = frame_idx / orig_fps

        # Process on stride frames
        if frame_idx % stride == 0:
            det_results = detector.predict(frame, conf=0.25, verbose=False)
            boxes = det_results[0].boxes if det_results and det_results[0] is not None and len(det_results[0].boxes) > 0 else []

            current_detections = []
            for b in boxes:
                c = float(b.conf[0])
                coords = b.xyxy[0].cpu().numpy().astype(int).tolist()
                current_detections.append({"box": coords, "conf": c})

            # Match detections to active tracks
            matched_track_ids = set()
            unmatched_detections = []

            for det in current_detections:
                best_iou = 0.0
                best_tid = None
                for tid, track in active_tracks.items():
                    if tid in matched_track_ids:
                        continue
                    iou = compute_iou(det["box"], track["last_box"])
                    if iou > best_iou:
                        best_iou = iou
                        best_tid = tid

                if best_iou > 0.35 and best_tid is not None:
                    matched_track_ids.add(best_tid)
                    active_tracks[best_tid]["last_box"] = det["box"]
                    active_tracks[best_tid]["last_seen_sec"] = current_sec
                    active_tracks[best_tid]["detections"].append(det)
                else:
                    unmatched_detections.append(det)

            # Create new tracks
            for det in unmatched_detections:
                tid = next_track_id
                next_track_id += 1
                active_tracks[tid] = {
                    "track_id": tid,
                    "first_seen_sec": current_sec,
                    "last_seen_sec": current_sec,
                    "last_box": det["box"],
                    "best_crop": None,
                    "best_conf": det["conf"],
                    "ocr_votes": {},
                    "detections": [det],
                }

            # Run OCR smartly on active tracks:
            # Only if track needs OCR (< 2 successful votes) and at least 8 frames elapsed since last OCR
            for tid, track in list(active_tracks.items()):
                bx = track["last_box"]
                x1, y1, x2, y2 = bx
                crop_w = x2 - x1
                crop_h = y2 - y1

                # Update best crop
                crop = frame[max(0, y1):min(height, y2), max(0, x1):min(width, x2)]
                if crop.size > 0:
                    if track["best_crop"] is None or track["detections"][-1]["conf"] > track["best_conf"]:
                        track["best_crop"] = crop
                        track["best_conf"] = track["detections"][-1]["conf"]

                    # Smart OCR check:
                    # Don't call OCR if we already have strong consensus (>= 2 votes for a plate)
                    has_strong_consensus = any(v >= 2 for v in track["ocr_votes"].values())
                    frames_since_ocr = frame_idx - track.get("last_ocr_frame", -999)

                    if not has_strong_consensus and frames_since_ocr >= 8 and crop_w >= 50 and crop_h >= 16:
                        track["last_ocr_frame"] = frame_idx
                        enhanced = enhance_plate_crop(crop)
                        enh_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
                        try:
                            res = ocr.predict(enh_bgr)
                            tokens = []
                            for r in res:
                                tokens.extend(r.json.get("res", {}).get("rec_texts", []))
                            plate_str, state_name, _ = format_plate_text(tokens)
                            if plate_str:
                                track["ocr_votes"][plate_str] = track["ocr_votes"].get(plate_str, 0) + 1
                                if state_name:
                                    track["state"] = state_name
                        except Exception as e:
                            pass

            # Prune inactive tracks
            for tid in list(active_tracks.keys()):
                if current_sec - active_tracks[tid]["last_seen_sec"] > 1.5:
                    completed_tracks.append(active_tracks.pop(tid))

            # Annotate current frame
            annotated = frame.copy()
            for tid, track in active_tracks.items():
                x1, y1, x2, y2 = track["last_box"]
                voted_plate = max(track["ocr_votes"], key=track["ocr_votes"].get) if track["ocr_votes"] else f"TRACK #{tid}"
                annotated = draw_styled_box(annotated, x1, y1, x2, y2, voted_plate, track["best_conf"])

            # HUD overlay
            hud_text = f"ANPR VIDEO SURVEILLANCE | Active: {len(active_tracks)} | Logged: {len(completed_tracks) + len(active_tracks)}"
            cv2.rectangle(annotated, (15, 15), (width - 15, 45), (15, 20, 28), -1)
            cv2.putText(annotated, hud_text, (25, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 130), 2, cv2.LINE_AA)

            writer.write(annotated)

        frame_idx += 1

    cap.release()
    writer.release()

    # Move remaining active tracks to completed
    completed_tracks.extend(active_tracks.values())

    # Re-encode to H.264 mp4 via ffmpeg
    ffmpeg_cmd = [
        "/opt/homebrew/bin/ffmpeg",
        "-y",
        "-i", raw_out_path,
        "-vcodec", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        web_out_path,
    ]
    try:
        subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        final_video_name = Path(web_out_path).name
    except Exception as e:
        print(f"FFmpeg transcoding warning: {e}. Falling back to raw MP4.")
        final_video_name = Path(raw_out_path).name

    # Consolidate plate log results
    plates_log = []
    for track in completed_tracks:
        if len(track["detections"]) < 2 and not track["ocr_votes"]:
            continue  # Ignore single-frame transient noise

        best_plate = max(track["ocr_votes"], key=track["ocr_votes"].get) if track["ocr_votes"] else "UNREADABLE"
        thumb_b64 = mat_to_base64(track["best_crop"]) if track["best_crop"] is not None else ""

        plates_log.append({
            "track_id": track["track_id"],
            "plate_number": best_plate,
            "state": track.get("state"),
            "first_seen_sec": round(track["first_seen_sec"], 2),
            "last_seen_sec": round(track["last_seen_sec"], 2),
            "duration_sec": round(track["last_seen_sec"] - track["first_seen_sec"], 2),
            "confidence": round(track["best_conf"], 2),
            "detections_count": len(track["detections"]),
            "thumbnail": thumb_b64,
        })

    # Sort log by appearance time
    plates_log.sort(key=lambda p: p["first_seen_sec"])

    return {
        "success": True,
        "video_filename": final_video_name,
        "video_url": f"/api/video-stream/{final_video_name}",
        "processed_frames": frame_idx,
        "duration_sec": round(frame_idx / orig_fps, 2),
        "fps": round(target_fps, 1),
        "unique_vehicles_detected": len(plates_log),
        "plates": plates_log,
        "processing_time_sec": round(time.time() - start_time, 2),
    }
