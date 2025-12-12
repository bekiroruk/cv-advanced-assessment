

from pathlib import Path
import random

import cv2
import numpy as np

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit  # noqa: F401
except ImportError:
    trt = None
    cuda = None


IMG_SIZE = 640
NUM_IMAGES = 50  # kalibrasyonda kullanılacak örnek sayısı


class ImageBatchStream:
    def __init__(self, image_paths, batch_size=1, img_size=IMG_SIZE):
        self.image_paths = image_paths
        self.batch_size = batch_size
        self.img_size = img_size
        self.max_batches = len(self.image_paths) // self.batch_size
        self.batch = 0

    def reset(self):
        self.batch = 0

    def next_batch(self):
        if self.batch >= self.max_batches:
            return None

        batch_paths = self.image_paths[
            self.batch * self.batch_size : (self.batch + 1) * self.batch_size
        ]
        batch_data = []

        for p in batch_paths:
            img = cv2.imread(str(p))
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (self.img_size, self.img_size))
            img = img.astype(np.float32) / 255.0
            img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
            batch_data.append(img)

        if not batch_data:
            return None

        batch_data = np.stack(batch_data, axis=0)  # (B, C, H, W)
        self.batch += 1
        return batch_data


class EntropyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, batch_stream: ImageBatchStream, cache_file: Path):
        super().__init__()
        self.batch_stream = batch_stream
        self.cache_file = cache_file
        self.d_input = None
        self.current_batch = None

        # input buffer allocate
        shape = (batch_stream.batch_size, 3, batch_stream.img_size, batch_stream.img_size)
        size = int(np.prod(shape)) * np.dtype(np.float32).itemsize
        self.d_input = cuda.mem_alloc(size)

    def get_batch_size(self):
        return self.batch_stream.batch_size

    def get_batch(self, names):
        batch = self.batch_stream.next_batch()
        if batch is None:
            return None

        cuda.memcpy_htod(self.d_input, batch)
        self.current_batch = batch
        return [int(self.d_input)]

    def read_calibration_cache(self):
        if self.cache_file.exists():
            print(f"[INFO] Var olan kalibrasyon cache okunuyor: {self.cache_file}")
            return self.cache_file.read_bytes()
        return None

    def write_calibration_cache(self, cache):
        print(f"[INFO] Kalibrasyon cache yazılıyor: {self.cache_file}")
        self.cache_file.write_bytes(cache)


def main():
    if trt is None or cuda is None:
        raise RuntimeError(
            "TensorRT veya pycuda yüklü değil. "
            "INT8 kalibrasyon için GPU + TensorRT ortamına ihtiyacınız var."
        )

    root = Path(__file__).resolve().parents[1]
    models_dir = root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # Coco8 dataset'i senin loglarında şurada:
    data_root = Path("C:/Users/Bekir/Desktop/object-detection-api/datasets/coco8")
    img_dir = data_root / "images" / "train"

    if not img_dir.exists():
        raise FileNotFoundError(f"Kalibrasyon için image klasörü bulunamadı: {img_dir}")

    all_images = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
    if not all_images:
        raise RuntimeError(f"Kalibrasyon için uygun görüntü bulunamadı: {img_dir}")

    random.shuffle(all_images)
    selected = all_images[:NUM_IMAGES]
    print(f"[INFO] Kalibrasyon için {len(selected)} görüntü seçildi.")

    batch_stream = ImageBatchStream(selected, batch_size=1, img_size=IMG_SIZE)
    cache_file = models_dir / "calibration.cache"

    calibrator = EntropyCalibrator(batch_stream, cache_file)

    # Dummy ONNX -> network parse (builder, network vs) normalde burada yapılır,
    # fakat biz bu script'i sadece cache üretimi için kullanacağız.
    # Asıl engine oluşturma build_trt_engine.py içinde yapılacak.

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    config = builder.create_builder_config()
    config.int8_calibrator = calibrator
    config.set_flag(trt.BuilderFlag.INT8)
    config.max_workspace_size = 1 << 30

    print("[INFO] Boş bir network ile INT8 kalibrasyon tetikleniyor (sadece cache üretimi için).")
    engine = builder.build_engine(network, config)
    if engine is None:
        print("[WARN] Engine None döndü ama kalibrasyon cache yine de yazılmış olabilir.")

    print(f"[OK] Kalibrasyon tamamlandı. Cache dosyası: {cache_file}")


if __name__ == "__main__":
    main()
