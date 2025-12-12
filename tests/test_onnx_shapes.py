
import os
import numpy as np
import pytest

from inference.detector import Detector


@pytest.mark.skipif(
    not os.path.exists("models/model.onnx"),
    reason="ONNX model not found",
)
def test_onnx_detector_output_shape():
    detector = Detector(backend="onnx", model_path="models/model.onnx")

    img = np.zeros((480, 640, 3), dtype=np.uint8)
    result = detector(img)

    assert result.boxes.shape[1] == 4
    assert result.scores.shape[0] == result.boxes.shape[0]
