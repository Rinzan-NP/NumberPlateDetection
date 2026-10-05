"""
Dataset Manager for License Plate Detection.
Handles:
1. Parsing local Indian vehicle dataset (VOC XML format).
2. Downloading and extracting extended benchmark dataset from Hugging Face (COCO format).
3. Merging, normalizing, and splitting into Ultralytics YOLO format.
4. Generating dataset.yaml and summary statistics.
"""

import os
import shutil
import random
import json
import zipfile
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Base directories
BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
WORKSPACE_DIR = BACKEND_DIR.parent
ARCHIVE_DIR = WORKSPACE_DIR / "archive"
YOLO_DATA_DIR = BACKEND_DIR / "app" / "yolo_data"
HF_CACHE_DIR = ARCHIVE_DIR / "hf_extended"

HF_DATASET_URLS = {
    "test": "https://huggingface.co/datasets/keremberke/license-plate-object-detection/resolve/main/data/test.zip",
    "valid": "https://huggingface.co/datasets/keremberke/license-plate-object-detection/resolve/main/data/valid.zip",
}

RANDOM_SEED = 42


def parse_voc_xml(xml_path: Path) -> List[Tuple[float, float, float, float, float, float, Optional[str]]]:
    """
    Parses Pascal VOC XML annotation file.
    Returns: list of (xmin, ymin, xmax, ymax, img_w, img_h, plate_text)
    """
    results = []
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        size = root.find("size")
        if size is None:
            return results
        img_w = float(size.find("width").text)
        img_h = float(size.find("height").text)
        if img_w <= 0 or img_h <= 0:
            return results

        for obj in root.findall("object"):
            bndbox = obj.find("bndbox")
            if bndbox is None:
                continue
            xmin = float(bndbox.find("xmin").text)
            ymin = float(bndbox.find("ymin").text)
            xmax = float(bndbox.find("xmax").text)
            ymax = float(bndbox.find("ymax").text)

            # Clamp coordinates
            xmin = max(0.0, min(xmin, img_w))
            ymin = max(0.0, min(ymin, img_h))
            xmax = max(0.0, min(xmax, img_w))
            ymax = max(0.0, min(ymax, img_h))

            if xmax - xmin < 4 or ymax - ymin < 4:
                continue

            plate_text = None
            for attr in obj.findall(".//attribute"):
                name_elem = attr.find("name")
                val_elem = attr.find("value")
                if name_elem is not None and name_elem.text == "number_plate_text":
                    plate_text = val_elem.text if val_elem is not None else None
                    break

            results.append((xmin, ymin, xmax, ymax, img_w, img_h, plate_text))
    except Exception as e:
        print(f"Error parsing VOC {xml_path}: {e}")
    return results


def collect_local_samples() -> List[Dict]:
    """Collects samples from local archive directory."""
    sources = [
        {
            "images": ARCHIVE_DIR / "Indian_Number_Plates" / "Sample_Images",
            "annotations": ARCHIVE_DIR / "Annotations" / "Annotations",
            "origin": "indian_local",
        },
        {
            "images": ARCHIVE_DIR / "number_plate_images_ocr" / "number_plate_images_ocr",
            "annotations": ARCHIVE_DIR / "number_plate_annos_ocr" / "number_plate_annos_ocr",
            "origin": "ocr_local",
        },
    ]

    samples = []
    for src in sources:
        img_dir = src["images"]
        anno_dir = src["annotations"]
        if not anno_dir.exists():
            continue

        for xml_file in anno_dir.glob("*.xml"):
            img_name = xml_file.stem + ".jpg"
            img_path = img_dir / img_name
            if not img_path.exists():
                alt_name = img_name.replace(" ", "_")
                img_path = img_dir / alt_name
            if not img_path.exists():
                continue

            parsed = parse_voc_xml(xml_file)
            if not parsed:
                continue

            boxes = []
            for xmin, ymin, xmax, ymax, w, h, text in parsed:
                xc = (xmin + xmax) / 2.0 / w
                yc = (ymin + ymax) / 2.0 / h
                bw = (xmax - xmin) / w
                bh = (ymax - ymin) / h
                boxes.append({
                    "class_id": 0,
                    "x_center": xc,
                    "y_center": yc,
                    "width": bw,
                    "height": bh,
                    "text": text,
                })

            samples.append({
                "image_path": str(img_path),
                "boxes": boxes,
                "origin": src["origin"],
            })

    return samples


def download_hf_partition(partition: str = "test", max_samples: Optional[int] = None) -> List[Dict]:
    """
    Downloads and extracts benchmark license plate dataset from Hugging Face.
    Converts COCO annotations into standard format.
    """
    if partition not in HF_DATASET_URLS:
        raise ValueError(f"Unknown partition {partition}")

    HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = HF_CACHE_DIR / f"{partition}.zip"
    extract_dir = HF_CACHE_DIR / partition

    if not extract_dir.exists() or not (extract_dir / "_annotations.coco.json").exists():
        if not zip_path.exists() or zip_path.stat().st_size < 1000:
            print(f"Downloading {partition} dataset from Hugging Face ({HF_DATASET_URLS[partition]})...")
            urllib.request.urlretrieve(HF_DATASET_URLS[partition], zip_path)
            print(f"Downloaded {zip_path.stat().st_size / (1024*1024):.1f} MB.")

        print(f"Extracting {zip_path.name}...")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(extract_dir)

    coco_path = extract_dir / "_annotations.coco.json"
    with open(coco_path, "r") as f:
        coco = json.load(f)

    img_map = {img["id"]: img for img in coco.get("images", [])}
    annos_by_img = {}
    for ann in coco.get("annotations", []):
        img_id = ann["image_id"]
        annos_by_img.setdefault(img_id, []).append(ann)

    samples = []
    for img_id, img_info in img_map.items():
        img_name = img_info["file_name"]
        img_file = extract_dir / img_name
        if not img_file.exists():
            continue

        w = float(img_info["width"])
        h = float(img_info["height"])
        if w <= 0 or h <= 0:
            continue

        boxes = []
        for ann in annos_by_img.get(img_id, []):
            bx, by, bw, bh = ann["bbox"]
            xc = (bx + bw / 2.0) / w
            yc = (by + bh / 2.0) / h
            norm_w = bw / w
            norm_h = bh / h
            # Clip bounds
            xc = max(0.0, min(1.0, xc))
            yc = max(0.0, min(1.0, yc))
            norm_w = max(0.0, min(1.0, norm_w))
            norm_h = max(0.0, min(1.0, norm_h))
            boxes.append({
                "class_id": 0,
                "x_center": xc,
                "y_center": yc,
                "width": norm_w,
                "height": norm_h,
                "text": None,
            })

        if boxes:
            samples.append({
                "image_path": str(img_file),
                "boxes": boxes,
                "origin": f"hf_{partition}",
            })

        if max_samples and len(samples) >= max_samples:
            break

    return samples


def prepare_dataset(
    include_hf_benchmark: bool = True,
    hf_partitions: List[str] = ["test"],
    val_ratio: float = 0.2,
    seed: int = RANDOM_SEED,
) -> Dict:
    """
    Builds the unified YOLO dataset in Backend/app/yolo_data.
    Combines local Indian plates with extended Hugging Face benchmark data.
    """
    random.seed(seed)
    local_samples = collect_local_samples()
    print(f"Loaded {len(local_samples)} local Indian vehicle samples.")

    hf_samples = []
    if include_hf_benchmark:
        for part in hf_partitions:
            part_samples = download_hf_partition(part)
            print(f"Loaded {len(part_samples)} samples from Hugging Face '{part}' partition.")
            hf_samples.extend(part_samples)

    all_samples = local_samples + hf_samples
    random.shuffle(all_samples)

    if not all_samples:
        raise RuntimeError("No samples found to build dataset!")

    split_idx = int(len(all_samples) * (1.0 - val_ratio))
    train_samples = all_samples[:split_idx]
    val_samples = all_samples[split_idx:]

    # Clean and create directory structure
    for split_name in ["train", "val"]:
        img_dir = YOLO_DATA_DIR / "images" / split_name
        lbl_dir = YOLO_DATA_DIR / "labels" / split_name
        if img_dir.exists():
            shutil.rmtree(img_dir)
        if lbl_dir.exists():
            shutil.rmtree(lbl_dir)
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

    def write_split(split, split_name):
        img_out = YOLO_DATA_DIR / "images" / split_name
        lbl_out = YOLO_DATA_DIR / "labels" / split_name
        box_count = 0

        for idx, s in enumerate(split):
            src_img = Path(s["image_path"])
            dest_name = f"{s['origin']}_{idx:05d}_{src_img.name}"
            dest_img = img_out / dest_name
            shutil.copy2(src_img, dest_img)

            lbl_file = lbl_out / (dest_img.stem + ".txt")
            with open(lbl_file, "w") as f:
                for b in s["boxes"]:
                    f.write(f"{b['class_id']} {b['x_center']:.6f} {b['y_center']:.6f} {b['width']:.6f} {b['height']:.6f}\n")
                    box_count += 1
        return box_count

    train_boxes = write_split(train_samples, "train")
    val_boxes = write_split(val_samples, "val")

    # Generate dataset.yaml
    yaml_content = f"""path: {YOLO_DATA_DIR.resolve()}
train: images/train
val: images/val
names:
  0: license_plate
"""
    yaml_path = YOLO_DATA_DIR / "dataset.yaml"
    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    summary = {
        "status": "success",
        "total_images": len(all_samples),
        "train_images": len(train_samples),
        "val_images": len(val_samples),
        "train_boxes": train_boxes,
        "val_boxes": val_boxes,
        "local_samples": len(local_samples),
        "hf_samples": len(hf_samples),
        "yaml_path": str(yaml_path),
        "yolo_data_dir": str(YOLO_DATA_DIR),
    }

    summary_file = YOLO_DATA_DIR / "dataset_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print("\nDataset preparation complete:")
    print(f"  Total Images: {summary['total_images']}")
    print(f"  Train: {summary['train_images']} images ({train_boxes} plates)")
    print(f"  Val:   {summary['val_images']} images ({val_boxes} plates)")
    print(f"  YAML Config: {yaml_path}")
    return summary


def get_dataset_stats() -> Dict:
    """Returns current dataset statistics."""
    summary_file = YOLO_DATA_DIR / "dataset_summary.json"
    if summary_file.exists():
        try:
            with open(summary_file, "r") as f:
                return json.load(f)
        except Exception:
            pass

    # Fallback compute on disk
    train_imgs = list((YOLO_DATA_DIR / "images" / "train").glob("*.*")) if (YOLO_DATA_DIR / "images" / "train").exists() else []
    val_imgs = list((YOLO_DATA_DIR / "images" / "val").glob("*.*")) if (YOLO_DATA_DIR / "images" / "val").exists() else []
    return {
        "status": "ready" if train_imgs else "uninitialized",
        "total_images": len(train_imgs) + len(val_imgs),
        "train_images": len(train_imgs),
        "val_images": len(val_imgs),
        "yaml_path": str(YOLO_DATA_DIR / "dataset.yaml"),
    }


if __name__ == "__main__":
    prepare_dataset(include_hf_benchmark=True, hf_partitions=["test"])
