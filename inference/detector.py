from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import numpy as np
from ultralytics import YOLO

from .utils import DetectionResult, AvgTimer, now_ms


BackendType = Literal["pytorch", "onnx", "tensorrt"]


class Detector:
    """
    Çoklu backend destekleyen dedektör:
      - PyTorch (YOLOv8 .pt)
      - ONNX (YOLOv8 .onnx → Ultralytics ORT backend)
      - TensorRT (Ultralytics'in TRT backend'i, TRT ortamı varsa)
    """

    def __init__(
        self,
        backend: BackendType = "pytorch",
        model_path: Optional[str] = None,
        imgsz: int = 640,
        device: str = "cpu",
    ):
        self.backend = backend
        self.imgsz = imgsz
        self.device = device
        self.timer = AvgTimer(window=100)

        root = Path(__file__).resolve().parents[1]
        models_dir = root / "models"

        if model_path is None:
            if backend == "pytorch":
                model_path = str(models_dir / "latest.pt")
            elif backend == "onnx":
                model_path = str(models_dir / "model.onnx")
            elif backend == "tensorrt":
                # Ultralytics YOLO, .engine yolunu da açabiliyor.
                model_path = str(models_dir / "model_fp16.engine")
            else:
                raise ValueError(f"Bilinmeyen backend: {backend}")

        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model bulunamadı: {self.model_path}")

        # Ultralytics backend seçimini model uzantısına göre yapıyor.
        # .pt → PyTorch
        # .onnx → ONNX Runtime
        # .engine → TensorRT
        self.model = YOLO(str(self.model_path))

        print(
            f"[Detector] backend={self.backend}, "
            f"model_path={self.model_path}, device={self.device}"
        )

        self.warmup()

    def warmup(self, num_iters: int = 2):
        import cv2

        dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
        for _ in range(num_iters):
            _ = self.model.predict(
                dummy,
                imgsz=self.imgsz,
                device=self.device,
                verbose=False,
            )

    def __call__(self, frame: np.ndarray) -> DetectionResult:
        """
        frame: BGR numpy image (H,W,3)
        """
        t0 = now_ms()
        results = self.model.predict(
            frame,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )
        t1 = now_ms()
        dt = t1 - t0
        self.timer.update(dt)

        r0 = results[0]
        boxes_xyxy = r0.boxes.xyxy.cpu().numpy() if r0.boxes is not None else np.zeros((0, 4), dtype=np.float32)
        scores = r0.boxes.conf.cpu().numpy() if r0.boxes is not None else np.zeros((0,), dtype=np.float32)
        class_ids = r0.boxes.cls.cpu().numpy().astype(np.int32) if r0.boxes is not None else np.zeros((0,), dtype=np.int32)

        return DetectionResult(
            boxes=boxes_xyxy,
            scores=scores,
            class_ids=class_ids,
            time_ms=dt,
        )

    @property
    def avg_latency_ms(self) -> float:
        return self.timer
