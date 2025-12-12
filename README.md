
# CV Advanced Assessment – Edge AI Video Analytics System

This repository implements an end-to-end **Edge AI Video Analytics pipeline** for the Dataguess AI FAE (Computer Vision) technical assessment. The goal is to simulate a realistic production-style workflow covering the entire lifecycle of a computer vision model from training to deployment.

## 🚀 Key Features

* **Model Training:** YOLOv8 training on custom data with logging and augmentations.
* **Optimization:** Model export pipeline (PyTorch → ONNX → TensorRT).
* **Multi-backend Inference:** Unified interface for PyTorch, ONNX Runtime, and TensorRT.
* **Real-time Pipeline:** Detector + tracker fusion.
* **Deployment:** FastAPI REST API for serving detections.
* **Quality Assurance:** Unit tests with `pytest` and basic monitoring utilities.

---

## 📂 Repository Structure

```text
cv-advanced-assessment/
│
├── training/
│   ├── train.py                 # Training script using Ultralytics YOLO
│   ├── dataset.yaml             # Dataset configuration
│   └── logs/...                 # Training logs and artifacts
│
├── optimization/
│   ├── export_to_onnx.py        # PyTorch to ONNX export
│   ├── build_trt_engine.py      # ONNX to TensorRT engine builder
│   ├── calibrate_int8.py        # INT8 Calibration logic
│   └── benchmarks.py            # Latency and FPS benchmarking
│
├── inference/
│   ├── detector.py              # Main inference class (Multi-backend)
│   ├── tracker.py               # Object tracking implementation
│   ├── video_engine.py          # Video processing pipeline
│   ├── fusion.py                # Fusion utilities (Detection + Tracking)
│   └── utils.py                 # Pre/Post-processing helpers
│
├── api/
│   ├── server.py                # FastAPI application
│   ├── schemas.py               # Pydantic models
│   └── docker/Dockerfile        # Containerization setup
│
├── monitoring/
│   ├── logger.py                # Custom logging setup
│   ├── fps_meter.py             # FPS and Latency metering
│   └── dashboard.py             # Metrics dashboard placeholder
│
├── tests/
│   ├── test_inference.py        # Inference sanity checks
│   ├── test_onnx_shapes.py      # Shape validation tests
│   └── test_tracker.py          # Tracker logic tests
│
├── models/
│   ├── latest.pt                # PyTorch weights
│   ├── model.onnx               # Exported ONNX model
│   ├── model_fp16.engine        # TensorRT FP16 Engine
│   ├── model_int8.engine        # TensorRT INT8 Engine
│   └── calibration.cache        # INT8 calibration cache
│
├── benchmark_results.json       # Output of benchmark scripts
├── README.md                    # Project documentation
└── report.pdf                   # Technical report

-----

## 🛠️ 1. Environment Setup

The project was developed and tested on **Python 3.13 (CPU)**.
*Note: TensorRT / pycuda are not required for the basic PyTorch/ONNX pipeline but are needed to build `.engine` files.*

```bash
# Clone repository
git clone [https://github.com/bekiroruk/cv-advanced-assessment.git](https://github.com/bekiroruk/cv-advanced-assessment.git)
cd cv-advanced-assessment

# Create and activate virtual env (Windows / PowerShell example)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

-----

## 🏋️ 2. Training

Train a YOLOv8 model on the **COCO8** mini dataset.

```bash
python training/train.py
```

**This script will:**

1.  Download/load COCO8 (if not available).
2.  Train YOLOv8 for 10 epochs on CPU.
3.  Save logs/plots to `training/logs/exp_coco8*/`.
4.  Save weights to `training/logs/exp_coco8*/weights/best.pt`.

**Artifacts to inspect:**

  * `results.png`: Training loss & mAP curves.
  * `confusion_matrix.png`: Per-class performance.
  * `train_batch*.jpg`: Augmentations and predictions.

-----

## 🔄 3. Export to ONNX

After training, export the best checkpoint to ONNX format.

```bash
python optimization/export_to_onnx.py
```

  * **Output:** `models/model.onnx`
  * The script performs a basic shape check between PyTorch and ONNX outputs to ensure consistency.

-----

## ⚡ 4. TensorRT Engines (FP16 / INT8) – Optional

> **Requirement:** These steps require a GPU + TensorRT + pycuda environment (e.g., NVIDIA Jetson or a CUDA server). They are implemented but were not executed on the CPU-only dev machine.

### 4.1 INT8 Calibration

```bash
python optimization/calibrate_int8.py
```

  * Samples \~50 images from the COCO8 train set.
  * Preprocesses them and feeds an `EntropyCalibrator`.
  * Writes an INT8 calibration cache to `models/calibration.cache`.

### 4.2 Build TensorRT Engines

```bash
python optimization/build_trt_engine.py
```

  * Parses `models/model.onnx`.
  * Builds `models/model_fp16.engine` and `models/model_int8.engine`.
  * **Dynamic Shape Profile:**
      * Min: `(1, 3, 480, 480)`
      * Opt: `(1, 3, 640, 640)`
      * Max: `(4, 3, 1280, 1280)`

-----

## 📊 5. Benchmarks

Measure PyTorch vs. ONNX Runtime performance.

```bash
python optimization/benchmarks.py
```

  * **Results:** Written to `benchmark_results.json`.
  * **Metrics:** Average latency, p50 / p95 latency, and FPS estimates.

-----

## 👁️ 6. Inference Engine

The main entry point for generic inference is `inference/detector.py`.

**Example Usage:**

```python
from inference.detector import Detector
import cv2

detector = Detector(
    backend="onnx",                 # Options: "torch", "onnx", "tensorrt"
    model_path="models/model.onnx"
)

img = cv2.imread("path/to/image.jpg")
detections = detector(img)
print(detections)
```

**Components:**

  * **Detector:** Unified preprocessing, postprocessing, and Custom NMS. Supports batching and warm-up runs.
  * **Tracker (`inference/tracker.py`):** Simple IoU-based tracker implementation (`SimpleIOUTracker`).
  * **Fusion (`inference/fusion.py`):** Utilities to fuse detector outputs with tracker IDs.
  * **Video Engine (`inference/video_engine.py`):** Skeleton for real-time video processing.

-----

## 🌐 7. FastAPI REST API

Start the API server:

```bash
uvicorn api.server:app --reload --port 8000
```

**Available Endpoints:**

  * `GET /health`: Basic health check.
  * `GET /metrics`: Returns backend name and basic latency / FPS stats.
  * `POST /detect`: Accepts an image file. Returns bounding boxes, class IDs, scores, and inference time.

-----

## 📈 8. Monitoring

Monitoring utilities are located in the `monitoring/` directory:

  * `fps_meter.py`: Rolling FPS & latency meter.
  * `logger.py`: Simple JSON logger.
  * `dashboard.py`: Placeholder for future metrics dashboard integration (e.g., Prometheus/Grafana).

-----

## 🧪 9. Tests

Run unit tests to ensure system stability.

```bash
pytest -q
```

**Test Coverage:**

  * ONNX model shapes & dynamic axes.
  * Basic detector inference sanity checks.
  * Tracker behavior (ID assignment, IoU thresholds).

-----

## 📝 10. Notes & Future Work

  * **Dataset:** Replace the toy COCO8 dataset with a real-world dataset (e.g., VisDrone, UAVDT, or a custom edge dataset).
  * **Hardware:** Run `calibrate_int8.py` and `build_trt_engine.py` on a real TensorRT-enabled edge device.
  * **Monitoring:** Integrate monitoring metrics into a live dashboard.
  * **Tracking:** Upgrade the tracker from simple IoU to **ByteTrack**, **DeepSORT**, or **OC-SORT** for more robust multi-object tracking.

-----

**Author:** Bekir Oruk
**Role:** AI Field Application Engineer (Computer Vision) – Candidate

```
```
