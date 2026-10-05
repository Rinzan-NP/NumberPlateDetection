"""
YOLOv8 Number Plate Detection Training Pipeline.
Supports:
- Apple Silicon MPS GPU acceleration.
- Custom epoch, batch size, and learning rate configuration.
- Real-time training progress tracking via train_status.json for API/UI.
- Post-training validation and metric reporting.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
import torch
from ultralytics import YOLO

# Paths
BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
APP_DIR = BACKEND_DIR / "app"
YOLO_DATA_DIR = APP_DIR / "yolo_data"
DATASET_YAML = YOLO_DATA_DIR / "dataset.yaml"
OUTPUT_PROJECT = YOLO_DATA_DIR / "runs"
WEIGHTS_DIR = YOLO_DATA_DIR / "plate_detector" / "weights"
STATUS_FILE = YOLO_DATA_DIR / "train_status.json"


def update_status(status_dict: dict):
    try:
        STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(STATUS_FILE, "w") as f:
            json.dump(status_dict, f, indent=2)
    except Exception as e:
        print(f"Error updating status: {e}")


class TrainingProgressCallback:
    """Ultralytics callback to record training metrics per epoch."""

    def __init__(self, total_epochs: int):
        self.total_epochs = total_epochs
        self.history = []

    def on_train_epoch_end(self, trainer):
        epoch = trainer.epoch + 1
        metrics = trainer.metrics or {}
        losses = trainer.loss_items.tolist() if hasattr(trainer, "loss_items") and trainer.loss_items is not None else []
        
        box_loss = losses[0] if len(losses) > 0 else 0.0
        cls_loss = losses[1] if len(losses) > 1 else 0.0
        dfl_loss = losses[2] if len(losses) > 2 else 0.0

        map50 = float(metrics.get("metrics/mAP50(B)", 0.0))
        map50_95 = float(metrics.get("metrics/mAP50-95(B)", 0.0))
        precision = float(metrics.get("metrics/precision(B)", 0.0))
        recall = float(metrics.get("metrics/recall(B)", 0.0))

        record = {
            "epoch": epoch,
            "total_epochs": self.total_epochs,
            "train_loss": round(box_loss + cls_loss + dfl_loss, 4),
            "box_loss": round(box_loss, 4),
            "cls_loss": round(cls_loss, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "map50": round(map50, 4),
            "map50_95": round(map50_95, 4),
            "timestamp": time.time(),
        }
        self.history.append(record)

        update_status({
            "status": "training",
            "progress_percent": round((epoch / self.total_epochs) * 100, 1),
            "current_epoch": epoch,
            "total_epochs": self.total_epochs,
            "latest_metrics": record,
            "history": self.history[-30:],  # keep last 30 for visualization
            "message": f"Training epoch {epoch}/{self.total_epochs} - mAP50: {map50:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}",
        })


def get_default_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"


def train_model(
    epochs: int = 25,
    batch: int = 16,
    imgsz: int = 640,
    device: str = None,
    base_model: str = "yolov8n.pt",
    name: str = "plate_run",
) -> dict:
    if not DATASET_YAML.exists():
        raise FileNotFoundError(f"dataset.yaml not found at {DATASET_YAML}. Run dataset_manager.py first.")

    chosen_device = device or get_default_device()
    print(f"Starting YOLOv8 training on device: {chosen_device} for {epochs} epochs...")

    update_status({
        "status": "starting",
        "progress_percent": 0.0,
        "current_epoch": 0,
        "total_epochs": epochs,
        "device": chosen_device,
        "base_model": base_model,
        "message": f"Initializing YOLOv8 model ({base_model}) on {chosen_device}...",
        "history": [],
    })

    start_time = time.time()
    try:
        model = YOLO(base_model)
        callback_handler = TrainingProgressCallback(total_epochs=epochs)
        model.add_callback("on_train_epoch_end", callback_handler.on_train_epoch_end)

        train_results = model.train(
            data=str(DATASET_YAML),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=chosen_device,
            name=name,
            project=str(OUTPUT_PROJECT),
            verbose=True,
            plots=True,
            val=True,
        )

        # Retrieve best weights
        run_dir = Path(train_results.save_dir) if hasattr(train_results, "save_dir") else OUTPUT_PROJECT / name
        best_pt = run_dir / "weights" / "best.pt"

        WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
        if best_pt.exists():
            import shutil
            shutil.copy2(best_pt, WEIGHTS_DIR / "best.pt")
            print(f"Copied newly trained best weights to {WEIGHTS_DIR / 'best.pt'}")

        # Run final validation
        val_model = YOLO(str(WEIGHTS_DIR / "best.pt"))
        val_res = val_model.val(data=str(DATASET_YAML), device=chosen_device, verbose=False)

        final_summary = {
            "status": "completed",
            "progress_percent": 100.0,
            "current_epoch": epochs,
            "total_epochs": epochs,
            "total_time_seconds": round(time.time() - start_time, 2),
            "weights_path": str(WEIGHTS_DIR / "best.pt"),
            "mAP50": round(float(val_res.box.map50), 4),
            "mAP50_95": round(float(val_res.box.map), 4),
            "precision": round(float(val_res.box.mp), 4),
            "recall": round(float(val_res.box.mr), 4),
            "device": chosen_device,
            "message": f"Training complete! Final mAP50: {val_res.box.map50:.4f}, Precision: {val_res.box.mp:.4f}, Recall: {val_res.box.mr:.4f}",
            "history": callback_handler.history,
        }

        update_status(final_summary)
        print("\n=== Training Completed Successfully ===")
        print(f"mAP@50:    {final_summary['mAP50']:.4f}")
        print(f"mAP@50-95: {final_summary['mAP50_95']:.4f}")
        print(f"Precision: {final_summary['precision']:.4f}")
        print(f"Recall:    {final_summary['recall']:.4f}")
        print(f"Saved to:  {WEIGHTS_DIR / 'best.pt'}")
        return final_summary

    except Exception as e:
        err_msg = f"Training failed with error: {str(e)}"
        print(err_msg, file=sys.stderr)
        update_status({
            "status": "failed",
            "progress_percent": 0.0,
            "error": str(e),
            "message": err_msg,
        })
        raise


def get_training_status() -> dict:
    if STATUS_FILE.exists():
        try:
            with open(STATUS_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "status": "idle",
        "progress_percent": 0.0,
        "message": "Model ready for inference or fine-tuning.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv8 plate detector")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--device", type=str, default=None, help="Device (mps, cpu, cuda)")
    parser.add_argument("--base-model", type=str, default="yolov8n.pt", help="Base weights")
    args = parser.parse_args()

    train_model(
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        base_model=args.base_model,
    )
