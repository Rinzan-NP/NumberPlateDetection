"""
Prepares license plate dataset from archive XML annotations.
Crops plate regions, splits 70% train / 30% test.
"""
import os
import shutil
import random
import xml.etree.ElementTree as ET
from pathlib import Path
from PIL import Image

ARCHIVE = Path(__file__).parent.parent.parent.parent / "archive"
OUTPUT = Path(__file__).parent.parent / "data"
TRAIN_DIR = OUTPUT / "train"
TEST_DIR = OUTPUT / "test"

# Dataset 1: Indian_Number_Plates
INDIAN_IMAGES = ARCHIVE / "Indian_Number_Plates" / "Sample_Images"
INDIAN_ANNOS = ARCHIVE / "Annotations" / "Annotations"

# Dataset 2: number_plate_images_ocr
OCR_IMAGES = ARCHIVE / "number_plate_images_ocr" / "number_plate_images_ocr"
OCR_ANNOS = ARCHIVE / "number_plate_annos_ocr" / "number_plate_annos_ocr"

SEED = 42
random.seed(SEED)


def parse_voc_xml(xml_path):
    """Returns list of dicts with keys: xmin, ymin, xmax, ymax, text."""
    results = []
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for obj in root.findall("object"):
            text = None
            for attr in obj.findall("attributes/attribute"):
                if attr.find("name").text == "number_plate_text":
                    text = attr.find("value").text
                    break
            if text is None:
                continue
            bndbox = obj.find("bndbox")
            results.append({
                "text": text.strip(),
                "xmin": float(bndbox.find("xmin").text),
                "ymin": float(bndbox.find("ymin").text),
                "xmax": float(bndbox.find("xmax").text),
                "ymax": float(bndbox.find("ymax").text),
            })
    except Exception as e:
        print(f"Error parsing {xml_path}: {e}")
    return results


def collect_samples():
    samples = []

    # Indian plates
    for xml_file in INDIAN_ANNOS.iterdir():
        if not xml_file.suffix == ".xml":
            continue
        img_name = xml_file.stem + ".jpg"  # e.g. Datacluster_number_plates (1)
        img_path = INDIAN_IMAGES / img_name
        if not img_path.exists():
            # Try without spaces
            alt_name = img_name.replace(" ", "_")
            img_path = INDIAN_IMAGES / alt_name
        if not img_path.exists():
            print(f"Image not found for {xml_file.name}")
            continue
        for anno in parse_voc_xml(xml_file):
            samples.append({
                "image_path": str(img_path),
                "text": anno["text"],
                "xmin": anno["xmin"], "ymin": anno["ymin"],
                "xmax": anno["xmax"], "ymax": anno["ymax"],
            })

    # OCR plates
    for xml_file in OCR_ANNOS.iterdir():
        if not xml_file.suffix == ".xml":
            continue
        img_name = xml_file.stem + ".jpg"
        img_path = OCR_IMAGES / img_name
        if not img_path.exists():
            print(f"Image not found for {xml_file.name}")
            continue
        for anno in parse_voc_xml(xml_file):
            samples.append({
                "image_path": str(img_path),
                "text": anno["text"],
                "xmin": anno["xmin"], "ymin": anno["ymin"],
                "xmax": anno["xmax"], "ymax": anno["ymax"],
            })

    return samples


def crop_and_save(sample, out_dir):
    img = Image.open(sample["image_path"])
    xmin = int(sample["xmin"])
    ymin = int(sample["ymin"])
    xmax = int(sample["xmax"])
    ymax = int(sample["ymax"])
    # Clamp to image bounds
    xmin = max(0, xmin)
    ymin = max(0, ymin)
    xmax = min(img.width, xmax)
    ymax = min(img.height, ymax)
    crop = img.crop((xmin, ymin, xmax, ymax))
    # Save with text label as filename prefix
    safe_text = sample["text"].replace("/", "_").replace(" ", "_")
    idx = random.randint(1000, 9999)
    filename = f"{safe_text}_{idx}.jpg"
    crop_path = out_dir / filename
    crop.save(crop_path, "JPEG", quality=95)
    return filename, sample["text"]


def main():
    print("Collecting samples...")
    samples = collect_samples()
    print(f"Total samples: {len(samples)}")

    # 70 / 30 split
    random.shuffle(samples)
    split_idx = int(len(samples) * 0.7)
    train_samples = samples[:split_idx]
    test_samples = samples[split_idx:]
    print(f"Train: {len(train_samples)}, Test: {len(test_samples)}")

    # Clean output dirs
    for d in [TRAIN_DIR, TEST_DIR]:
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)

    # Save splits as CSV
    import csv
    for split_name, split_samples in [("train", train_samples), ("test", test_samples)]:
        out_dir = TRAIN_DIR if split_name == "train" else TEST_DIR
        csv_path = OUTPUT / f"{split_name}_labels.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "text"])
            for s in split_samples:
                filename, text = crop_and_save(s, out_dir)
                writer.writerow([filename, text])
        print(f"Saved {len(split_samples)} crops to {out_dir} and labels to {csv_path}")


if __name__ == "__main__":
    main()
