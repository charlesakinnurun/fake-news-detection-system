# syntax=docker/dockerfile:1

# ---------------------------------------------------------------------------
# Shared runtime. The package is installed once, then specialised per target.
# The model artifact is intentionally NOT baked into the image: mount it at
# /app/artifacts (see docker-compose.yml) so it can be versioned and rolled
# back independently of the application image.
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HOME=/home/appuser

WORKDIR /app

RUN useradd --create-home --uid 10001 appuser

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY configs ./configs

RUN python -m pip install --upgrade pip \
    && python -m pip install ".[api]"

# ---------------------------------------------------------------------------
# API image
# ---------------------------------------------------------------------------
FROM base AS api

USER appuser
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"

CMD ["uvicorn", "fake_news_detection.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# ---------------------------------------------------------------------------
# Streamlit image
# ---------------------------------------------------------------------------
FROM base AS app

RUN python -m pip install ".[app]"

COPY app ./app

USER appuser
EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=3)"

CMD ["streamlit", "run", "app/streamlit_app.py", \
     "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
