
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


def iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    box1, box2: [x1, y1, x2, y2]
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h

    area1 = max(0.0, (box1[2] - box1[0])) * max(0.0, (box1[3] - box1[1]))
    area2 = max(0.0, (box2[2] - box2[0])) * max(0.0, (box2[3] - box2[1]))
    union = area1 + area2 - inter + 1e-6

    return float(inter / union)


def nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_thresh: float = 0.5,
) -> List[int]:
    """
    Basit NMS.
    boxes: (N, 4)
    scores: (N,)
    return: tutulacak index listesi
    """
    if len(boxes) == 0:
        return []

    idxs = scores.argsort()[::-1]
    keep = []

    while len(idxs) > 0:
        current = idxs[0]
        keep.append(int(current))
        if len(idxs) == 1:
            break
        rest = idxs[1:]
        ious = np.array([iou(boxes[current], boxes[i]) for i in rest])
        rest = rest[ious < iou_thresh]
        idxs = rest

    return keep


@dataclass
class DetectionResult:
    boxes: np.ndarray      # (N, 4) [x1,y1,x2,y2]
    scores: np.ndarray     # (N,)
    class_ids: np.ndarray  # (N,)
    time_ms: float


class AvgTimer:
    """
    Hareketli ortalama latency tutmak için küçük yardımcı sınıf.
    """
    def __init__(self, window: int = 100):
        self.window = window
        self.values: List[float] = []

    def update(self, value: float):
        self.values.append(float(value))
        if len(self.values) > self.window:
            self.values = self.values[-self.window:]

    @property
    def avg(self) -> float:
        if not self.values:
            return 0.0
        return float(sum(self.values) / len(self.values))


def now_ms() -> float:
    return time.perf_counter() * 1000.0
