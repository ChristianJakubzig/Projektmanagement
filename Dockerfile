# Multi-stage build für kleinere Images
FROM python:3.11-slim as base

# Umgebungsvariablen
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Arbeitsverzeichnis setzen
WORKDIR /app

# Build-Dependencies (werden später entfernt)
FROM base as builder
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    g++ \
    cmake \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Rust installieren (nur für Build)
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y
ENV PATH="/root/.cargo/bin:$PATH"

# Python Dependencies installieren
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install langchain-chroma

# Final stage - nur Runtime
FROM base as final

# Nur Runtime-Dependencies
RUN apt-get update && apt-get install -y \
    poppler-utils \
    tesseract-ocr \
    libjpeg62-turbo \
    libpq5 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Python packages von builder kopieren
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# App-Code kopieren
COPY ./app /app

# Logs und Data Ordner erstellen
RUN mkdir -p /app/logs /app/uploads

# Port exponieren
EXPOSE 8000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Standard-Command für Development
CMD ["tail", "-f", "/dev/null"]