FROM node:24-bookworm-slim AS node-runtime

FROM python:3.11-slim-bookworm
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libsndfile1 && rm -rf /var/lib/apt/lists/*
COPY --from=node-runtime /usr/local/bin/node /usr/local/bin/node
COPY requirements.txt .
ARG TORCH_VERSION=2.5.1
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir --index-url "$TORCH_INDEX_URL" "torch==$TORCH_VERSION" && \
    pip install --no-cache-dir -r requirements.txt
COPY dj_train ./dj_train
COPY tests ./tests
ENTRYPOINT ["python", "-m", "dj_train.cli"]
CMD ["status"]
