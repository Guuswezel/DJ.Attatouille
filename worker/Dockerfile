FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libsndfile1 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Harmonix is provided by the maintained all-in-one-infer package. It runs the
# original Harmonix models with a pure-PyTorch attention backend, avoiding the
# unmaintained NATTEN/madmom native dependency chain of the legacy allin1
# package. The base Compose file stays CPU-compatible; docker-compose.gpu.yml
# switches this index to a CUDA wheel on an NVIDIA Linux host.
ARG INSTALL_HARMONIX=false
ARG TORCH_VERSION=2.5.1
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
RUN if [ "$INSTALL_HARMONIX" = "true" ]; then \
      pip install --no-cache-dir --index-url "$TORCH_INDEX_URL" "torch==$TORCH_VERSION" "torchaudio==$TORCH_VERSION" && \
      pip install --no-cache-dir all-in-one-infer; \
    fi
COPY app ./app
COPY tests ./tests
EXPOSE 8090
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8090"]
