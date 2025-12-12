
import numpy as np

from inference.detector import Detector


def test_pytorch_detector_single_image():
    detector = Detector(backend="pytorch", model_path="models/latest.pt")

    img = np.zeros((480, 640, 3), dtype=np.uint8)
    result = detector(img)

    # Çıktı formatı temel kontroller
    assert result.boxes.ndim == 2
    assert result.scores.ndim == 1
    assert result.class_ids.ndim == 1
    assert result.boxes.shape[0] == result.scores.shape[0] == result.class_ids.shape[0]
