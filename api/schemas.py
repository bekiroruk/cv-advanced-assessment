# api/schemas.py
from __future__ import annotations

from typing import List

from pydantic import BaseModel


class BBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    class_id: int
    track_id: int | None = None


class DetectResponse(BaseModel):
    boxes: List[BBox]
    inference_time_ms: float
    backend: str
    avg_latency_ms: float


class HealthResponse(BaseModel):
    status: str
    detail: str


class MetricsResponse(BaseModel):
    total_requests: int
    avg_latency_ms: float
