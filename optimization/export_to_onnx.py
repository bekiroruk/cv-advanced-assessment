
from __future__ import annotations

from pathlib import Path

from ultralytics import YOLO


def export_to_onnx(
    weights_path: str = "models/latest.pt",
    onnx_path: str = "models/model.onnx",
    imgsz: int = 640,
) -> None:
    """
    Eğitilmiş YOLOv8 modelini ONNX formatına export eder.

    Dynamic shape (batch, height, width) ve opset>=12 ile export.
    """
    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(f"Model bulunamadı: {weights_path}")

    print(f"[export_to_onnx] YOLO model yükleniyor: {weights_path}")
    model = YOLO(str(weights_path))

    print("[export_to_onnx] ONNX formatına export ediliyor...")
    # Ultralytics kendi export fonksiyonunu kullanıyoruz
    onnx_file = model.export(
        format="onnx",
        imgsz=imgsz,
        dynamic=True,
        opset=12,
        simplify=True,
        project="models",
        name="onnx_export",
        exist_ok=True,
    )

    # Ultralytics 'models/onnx_export/weights/best.onnx' gibi bir şey üretecek
    onnx_file = Path(onnx_file)

    target = Path(onnx_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"[export_to_onnx] {onnx_file} → {target} kopyalanıyor...")
    target.write_bytes(onnx_file.read_bytes())

    print(f"[export_to_onnx] Tamamlandı. ONNX model: {target}")


if __name__ == "__main__":
    export_to_onnx()
