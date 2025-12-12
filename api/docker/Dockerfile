# api/docker/Dockerfile

# Örnek baz imaj: NVIDIA TensorRT runtime
FROM nvcr.io/nvidia/tensorrt:23.12-py3

WORKDIR /app

# Sistem paketleri
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-dev python3-pip git \
    && rm -rf /var/lib/apt/lists/*

# Python bağımlılıkları (örnek)
COPY ../requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Projeyi kopyala
COPY .. /app

# FastAPI + Uvicorn
EXPOSE 8000

ENV PYTHONUNBUFFERED=1

CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000"]
