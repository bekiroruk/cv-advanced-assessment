
from pathlib import Path
import shutil

import torch
from ultralytics import YOLO


def main():
    # Proje root'unu bul (cv-advanced-assessment klasörü)
    root = Path(__file__).resolve().parents[1]

    # Modellerin saklanacağı klasör
    models_dir = root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # Log'ların saklanacağı klasör
    logs_dir = root / "training" / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Küçük ve hızlı model: YOLOv8n
    model = YOLO("yolov8n.pt")

    # Cihaz seçimi (GPU varsa cuda, yoksa cpu)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 🔴 ÖNEMLİ: Artık dataset.yaml kullanmıyoruz, direkt coco8.yaml kullanıyoruz
    # Ultralytics kendi coco8.yaml'ını ve datasını otomatik bulup kullanıyor.
    results = model.train(
        data="coco8.yaml",           # <-- BURASI dataset.yaml yerine COCO8
        epochs=10,                   # deneme için 10 epoch
        imgsz=640,
        project=str(logs_dir),       # training/logs altına yazar
        name="exp_coco8",
        cos_lr=True,                 # cosine learning rate
        amp=True,                    # mixed precision
        batch=8,
        device=device,
        workers=2,
        pretrained=True,
    )

    # En iyi ağırlığı al ve models/latest.pt olarak kopyala
    trainer = model.trainer
    best_ckpt = Path(trainer.best)
    latest_path = models_dir / "latest.pt"
    shutil.copy(best_ckpt, latest_path)

    print(f"\n[OK] Training finished.")
    print(f"Best model copied to: {latest_path}")


if __name__ == "__main__":
    main()
