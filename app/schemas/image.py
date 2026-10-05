from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class BoundingBox(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int

class DetectedPlate(BaseModel):
    id: int
    plate_number: str
    raw_text: Optional[str] = ""
    state: Optional[str] = None
    detector_confidence: float
    ocr_confidence: float
    bbox: BoundingBox
    thumbnail: str
    plate_heatmap: Optional[str] = ""

class ImageDetectionResponse(BaseModel):
    success: bool = True
    text: str = ""
    plates_detected_count: int = 0
    plates: List[DetectedPlate] = []
    annotated_image: Optional[str] = ""
    heatmap_image: Optional[str] = ""
    error: Optional[str] = None

class VideoPlateTrack(BaseModel):
    track_id: int
    plate_number: str
    state: Optional[str] = None
    first_seen_sec: float
    last_seen_sec: float
    duration_sec: float
    confidence: float
    detections_count: int
    thumbnail: str

class VideoDetectionResponse(BaseModel):
    success: bool = True
    video_filename: str
    video_url: str
    processed_frames: int
    duration_sec: float
    fps: float
    unique_vehicles_detected: int
    plates: List[VideoPlateTrack] = []
    processing_time_sec: float
    error: Optional[str] = None

class TrainingRequest(BaseModel):
    epochs: int = Field(default=20, ge=1, le=100)
    batch: int = Field(default=16, ge=1, le=64)
    device: Optional[str] = None
    base_model: Optional[str] = "yolov8n.pt"

class TrainingStatusResponse(BaseModel):
    status: str
    progress_percent: float = 0.0
    current_epoch: Optional[int] = 0
    total_epochs: Optional[int] = 0
    latest_metrics: Optional[Dict[str, Any]] = None
    history: Optional[List[Dict[str, Any]]] = []
    message: str = ""
    error: Optional[str] = None
