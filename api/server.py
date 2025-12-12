# api/server.py
from __future__ import annotations

import io
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

from inference.detector import Detector
from inference.tracker import SimpleIOUTracker
from inference.fusion import tracks_to_dicts
from api.schemas import (
    DetectResponse,
    HealthResponse,
    MetricsResponse,
    BBox,
)

app = FastAPI(title="Edge AI Detection API")

# Global nesneler (basit, production'da daha sofistike yapılabilir)
_detector: Optional[Detector] = None
_tracker: Optional[SimpleIOUTracker] = None
_total_requests: int = 0


def get_detector() -> Detector:
    global _detector
    if _detector is None:
        # Varsayılan: PyTorch backend, latest.pt
        _detector = Detector(
            backend="pytorch",
            model_path=str(
                Path(__file__).resolve().parents[1] / "models" / "latest.pt"
            ),
            imgsz=640,
            device="cpu",  # GPU ortamında "cuda" olarak değiştirilebilir
        )
    return _detector


def get_tracker() -> SimpleIOUTracker:
    global _tracker
    if _tracker is None:
        _tracker = SimpleIOUTracker(iou_thresh=0.5, max_missed=10)
    return _tracker


def read_image_from_upload(file: UploadFile) -> np.ndarray:
    contents = file.file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    img_np = np.array(image)
    # PIL → RGB, OpenCV → BGR
    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    return img_np


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok", detail="API is running.")


@app.post("/detect", response_model=DetectResponse)
def detect(file: UploadFile = File(...)):
    global _total_requests
    _total_requests += 1

    img = read_image_from_upload(file)
    detector = get_detector()
    det = detector(img)

    boxes = []
    for i in range(len(det.boxes)):
        x1, y1, x2, y2 = det.boxes[i].tolist()
        score = float(det.scores[i])
        class_id = int(det.class_ids[i])
        boxes.append(
            BBox(
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                score=score,
                class_id=class_id,
                track_id=None,
            )
        )

    resp = DetectResponse(
        boxes=boxes,
        inference_time_ms=det.time_ms,
        backend=detector.backend,
        avg_latency_ms=detector.avg_latency_ms,
    )
    return resp


@app.get("/metrics", response_model=MetricsResponse)
def metrics():
    det = get_detector()
    return MetricsResponse(
        total_requests=_total_requests,
        avg_latency_ms=det.avg_latency_ms,
    )
