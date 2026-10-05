# 🚗 AuraPlate ANPR: Neural Number Plate Detection & Recognition

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.140-009688.svg)](https://fastapi.tiangolo.com/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00ffff.svg)](https://github.com/ultralytics/ultralytics)
[![PaddleOCR](https://img.shields.io/badge/PaddleOCR-v6-red.svg)](https://github.com/PaddlePaddle/PaddleOCR)
[![mAP@50](https://img.shields.io/badge/mAP%4050-98.3%25-brightgreen.svg)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)]()

> **An End-to-End Deep Learning Automatic Number Plate Recognition (ANPR) System equipped with Explainable Neural Activation Heatmaps, CLAHE enhancement, and Indian State Grammar Parsing.**

---

## 🌟 Key Highlights

- **98.3% mAP@50 Accuracy**: Fine-tuned **YOLOv8 Nano** detector on a 929-image unified dataset (760 plates).
- **Sub-20ms Inference Latency**: Real-time throughput (~54 FPS) leveraging Apple Silicon MPS (Metal) GPU acceleration.
- **Explainable Neural Heatmaps**: 2D spatial Gaussian attention density maps with the **JET colormap** proving the network localizes plates accurately.
- **High-Accuracy Character Recognition**: **PaddleOCR v6** with textline orientation unwarping and character stroke saliency mapping.
- **Indian State Grammar Parsing**: Full support and validation across all **36 Indian States and Union Territories** (e.g., `AP` $\to$ Andhra Pradesh, `MH` $\to$ Maharashtra, `DL` $\to$ Delhi).
- **Presentation & Viva Ready**: Includes complete slide deck ([`PRESENTATION_SLIDES_README.md`](PRESENTATION_SLIDES_README.md)) and technical defense guide ([`CODE_EXPLANATION_GUIDE.md`](CODE_EXPLANATION_GUIDE.md)).

---

## 🏗️ System Architecture

```
[ Vehicle Image / Camera Frame ]
               │
               ▼
[ Stage 1: Pre-scaling & Letterboxing (640×640) ]
               │
               ▼
[ Stage 2: YOLOv8 Localization & NMS ]
      ├───> [ Neural Activation Heatmap (JET Scale) ]
      │
      ▼
[ Stage 3: ROI Crop & Padding Expansion ]
               │
               ▼
[ Stage 4: CLAHE Contrast Equalizer & Bilateral Filter ]
               │
               ▼
[ Stage 5: PaddleOCR v6 Textline & Character Saliency ]
               │
               ▼
[ Stage 6: Indian State Code Parser & Grammar Check ]
               │
               ▼
[ Output: Formatted HSRP Plate Badge, State Name, & Confidence ]
```

---

## 📊 Benchmark Metrics

| Metric | Initial Baseline (32 Images) | Proposed SOTA (929 Images) | Delta |
| :--- | :---: | :---: | :---: |
| **mAP@50** | 61.9% | **98.3%** | **+36.4%** |
| **mAP@50-95** | 46.2% | **70.1%** | **+23.9%** |
| **Precision** | 89.2% | **98.1%** | **+8.9%** |
| **Recall** | 56.2% | **97.2%** | **+41.0%** |
| **Inference Time** | 56.7 ms (CPU) | **18.5 ms** (MPS / GPU) | **3x Faster** |

---

## 🚀 Quick Start Guide

### 1. Prerequisites
- Python 3.10+ (Recommended: Python 3.12)
- Virtual environment (`venv`)

### 2. Installation
```bash
# Clone the repository
git clone https://github.com/Rinzan-NP/NumberPlateDetection.git
cd NumberPlateDetection

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Run the Backend API Server
```bash
python main.py
# Server runs on http://localhost:8000
# Swagger API docs available at http://localhost:8000/docs
```

---

## 📡 REST API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/api/detect` | Uploads vehicle image; returns bounding box, plate string, state, and neural heatmap. |
| `GET` | `/api/samples` | Lists preset sample vehicle images for 1-click testing. |
| `GET` | `/api/model-info` | Returns active detector architecture, mAP50, and checkpoint metadata. |
| `GET` | `/api/dataset-info` | Returns dataset composition (total images, train/val splits). |
| `POST` | `/api/train/start` | Triggers background model training / fine-tuning. |
| `GET` | `/api/train/status` | Returns real-time training telemetry, loss, and epoch progress. |
| `GET` | `/health` | System health check endpoint. |

---

## 📑 College Presentation & Viva Documents

- 🎓 **[Presentation Slides Guide (`PRESENTATION_SLIDES_README.md`)](PRESENTATION_SLIDES_README.md)**: 17 complete slides with on-slide text, visual layout notes, spoken scripts, and evaluator defense tips.
- 🧠 **[Basic Code Walkthrough & Viva Guide (`CODE_EXPLANATION_GUIDE.md`)](CODE_EXPLANATION_GUIDE.md)**: Beginner-friendly, block-by-block explanation of every line of code with the 5 most common viva answers.
- 📽️ **[Marp Presentation Deck (`slides.md`)](slides.md)**: Markdown presentation file for one-click slide generation.

---

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.
