

from pathlib import Path

import numpy as np

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit  # noqa: F401
except ImportError:
    trt = None
    cuda = None

ENGINE_FP16_NAME = "model_fp16.engine"
ENGINE_INT8_NAME = "model_int8.engine"
CALIB_CACHE_NAME = "calibration.cache"


def build_engine(
    onnx_path: Path,
    engine_path: Path,
    fp16: bool = False,
    int8: bool = False,
    calib_cache: Path | None = None,
):
    if trt is None or cuda is None:
        raise RuntimeError(
            "TensorRT veya pycuda yüklü değil. "
            "Bu script'i çalıştırmak için TensorRT + pycuda kurulu bir GPU ortamına ihtiyacınız var."
        )

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)

    parser = trt.OnnxParser(network, logger)

    print(f"[INFO] ONNX modeli yükleniyor: {onnx_path}")
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            print("[ERROR] ONNX parse hatası:")
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise RuntimeError("ONNX parse başarısız.")

    config = builder.create_builder_config()
    config.max_workspace_size = 1 << 30  # 1 GB

    profile = builder.create_optimization_profile()
    input_tensor = network.get_input(0)
    input_name = input_tensor.name
    print(f"[INFO] Input tensor: {input_name}, shape={input_tensor.shape}")

    # Dinamik batch, yüksekliği ve genişliği için profil
    min_shape = (1, 3, 480, 480)
    opt_shape = (1, 3, 640, 640)
    max_shape = (4, 3, 1280, 1280)
    profile.set_shape(input_name, min=min_shape, opt=opt_shape, max=max_shape)
    config.add_optimization_profile(profile)

    if fp16:
        if not builder.platform_has_fast_fp16:
            print("[WARN] Platform fast FP16 desteklemiyor ama yine de FP16 flag'i set ediliyor.")
        config.set_flag(trt.BuilderFlag.FP16)

    if int8:
        if not builder.platform_has_fast_int8:
            print("[WARN] Platform fast INT8 desteklemiyor ama yine de INT8 flag'i set ediliyor.")
        config.set_flag(trt.BuilderFlag.INT8)

        if calib_cache is None or not calib_cache.exists():
            raise FileNotFoundError(
                f"INT8 kalibrasyon cache dosyası bulunamadı: {calib_cache}. "
                "Önce calibrate_int8.py script'ini çalıştırmalısınız."
            )

        class CacheCalibrator(trt.IInt8EntropyCalibrator2):
            def __init__(self, cache_path: Path):
                super().__init__()
                self.cache_path = cache_path

            def get_batch_size(self):
                return 1

            def get_batch(self, names):
                # INT8 engine oluştururken cache kullanacağız,
                # bu yüzden runtime'da gerçek batch üretmemize gerek yok.
                return None

            def read_calibration_cache(self):
                print(f"[INFO] Kalibrasyon cache okunuyor: {self.cache_path}")
                return self.cache_path.read_bytes()

            def write_calibration_cache(self, cache):
                # Burada yeniden yazmaya gerek yok, zaten hazır.
                pass

        calibrator = CacheCalibrator(calib_cache)
        config.int8_calibrator = calibrator

    print("[INFO] TensorRT engine derleniyor...")
    engine = builder.build_engine(network, config)
    if engine is None:
        raise RuntimeError("TensorRT engine oluşturulamadı.")

    print(f"[INFO] Engine diske yazılıyor: {engine_path}")
    with open(engine_path, "wb") as f:
        f.write(engine.serialize())

    print(f"[OK] TensorRT engine oluşturuldu: {engine_path}")


def main():
    root = Path(__file__).resolve().parents[1]
    models_dir = root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    onnx_path = models_dir / "model.onnx"
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX modeli bulunamadı: {onnx_path}")

    fp16_engine_path = models_dir / ENGINE_FP16_NAME
    int8_engine_path = models_dir / ENGINE_INT8_NAME
    calib_cache_path = models_dir / CALIB_CACHE_NAME

    print("[INFO] PyTorch → ONNX → TensorRT pipeline (FP16 & INT8)")

    # FP16 engine
    print("\n[STEP] FP16 TensorRT engine oluşturma")
    try:
        build_engine(
            onnx_path=onnx_path,
            engine_path=fp16_engine_path,
            fp16=True,
            int8=False,
        )
    except RuntimeError as e:
        print(f"[WARN] FP16 engine oluşturulamadı: {e}")

    # INT8 engine
    print("\n[STEP] INT8 TensorRT engine oluşturma")
    try:
        build_engine(
            onnx_path=onnx_path,
            engine_path=int8_engine_path,
            fp16=False,
            int8=True,
            calib_cache=calib_cache_path,
        )
    except (RuntimeError, FileNotFoundError) as e:
        print(f"[WARN] INT8 engine oluşturulamadı: {e}")

    print("\n[INFO] build_trt_engine.py tamamlandı.")


if __name__ == "__main__":
    main()
