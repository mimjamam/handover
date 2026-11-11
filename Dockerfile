# One small container: the API also serves the demo page.
# Fits in 512 MB of memory (no PyTorch; 8-bit ONNX embedding model).
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /usr/local/bin/uv

RUN useradd -m -u 1000 user && mkdir -p /home/user/app && chown user:user /home/user/app
USER user
ENV HOME=/home/user \
    PATH=/home/user/app/.venv/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    HF_HUB_OFFLINE=0 \
    UV_COMPILE_BYTECODE=1 \
    UV_NO_CACHE=1 \
    HANDOVER_LOW_MEMORY=1 \
    PYTHONUNBUFFERED=1
WORKDIR /home/user/app

# Dependencies first, so code changes do not reinstall them.
COPY --chown=user pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Download the embedding model at build time: no slow first request,
# no failure if the download site is unreachable at start-up.
RUN python -c "from huggingface_hub import hf_hub_download as d; d('intfloat/multilingual-e5-small', 'sentencepiece.bpe.model'); d('Xenova/multilingual-e5-small', 'onnx/model_quantized.onnx')"
ENV HF_HUB_OFFLINE=1

COPY --chown=user src ./src
COPY --chown=user artifacts ./artifacts
RUN uv sync --frozen --no-dev

# Hosts such as Render pass the port in $PORT; 7860 is the local default.
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
  CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://localhost:{os.environ.get(\"PORT\", \"7860\")}/health')" || exit 1

CMD ["sh", "-c", "uvicorn handover.api:app --host 0.0.0.0 --port ${PORT:-7860}"]
