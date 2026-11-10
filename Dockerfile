# One container, two processes: the API (port 8000, internal) and the
# web page (port 7860, public). Built for a free Hugging Face Docker Space.
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /usr/local/bin/uv

RUN useradd -m -u 1000 user && mkdir -p /home/user/app && chown user:user /home/user/app
USER user
ENV HOME=/home/user \
    PATH=/home/user/app/.venv/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    UV_COMPILE_BYTECODE=1 \
    UV_NO_CACHE=1 \
    PYTHONUNBUFFERED=1
WORKDIR /home/user/app

# Dependencies first, so code changes do not reinstall them.
COPY --chown=user pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Download the embedding model at build time: no slow first request,
# no failure if the download site is unreachable at start-up.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('intfloat/multilingual-e5-small', device='cpu')"

COPY --chown=user src ./src
COPY --chown=user artifacts ./artifacts
COPY --chown=user start.sh ./
RUN uv sync --frozen --no-dev

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["bash", "start.sh"]
