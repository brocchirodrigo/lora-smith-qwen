FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake build-essential git curl ca-certificates \
 && curl -LsSf https://astral.sh/uv/install.sh | sh \
 && ln -s /root/.local/bin/uv /usr/local/bin/uv \
 && rm -rf /var/lib/apt/lists/* \
 && useradd -m -u 1000 trainer

WORKDIR /app

COPY --chown=trainer:trainer . .

USER trainer
ENV PATH="/home/trainer/.local/bin:/usr/local/bin:$PATH"

RUN make setup

VOLUME ["/app/models", "/app/data"]
