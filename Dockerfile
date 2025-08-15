# Multi-stage build für kleinere Images (dein ursprüngliches Design)
FROM python:3.11-bullseye as base

# Proxy-Konfiguration für TH Wildau
ENV http_proxy=http://proxy.th-wildau.de:8080 \
    https_proxy=http://proxy.th-wildau.de:8080 \
    HTTP_PROXY=http://proxy.th-wildau.de:8080 \
    HTTPS_PROXY=http://proxy.th-wildau.de:8080 \
    no_proxy=localhost,127.0.0.1,.th-wildau.de \
    NO_PROXY=localhost,127.0.0.1,.th-wildau.de

# Umgebungsvariablen
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Arbeitsverzeichnis setzen
WORKDIR /app

# Build-Dependencies (werden später entfernt)
FROM base as builder

# pip Proxy-Konfiguration
RUN pip config set global.proxy http://proxy.th-wildau.de:8080

# apt Proxy-Konfiguration (falls noch nötig)
RUN echo 'Acquire::http::Proxy "http://proxy.th-wildau.de:8080";' > /etc/apt/apt.conf.d/01proxy && \
    echo 'Acquire::https::Proxy "http://proxy.th-wildau.de:8080";' >> /etc/apt/apt.conf.d/01proxy

# Build-Dependencies installieren (bullseye hat schon die meisten)
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    g++ \
    cmake \
    git \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# SQLite3 von Source kompilieren (neueste Version für ChromaDB)
RUN cd /tmp && \
    wget https://www.sqlite.org/2024/sqlite-autoconf-3450000.tar.gz && \
    tar xzf sqlite-autoconf-3450000.tar.gz && \
    cd sqlite-autoconf-3450000 && \
    ./configure --prefix=/usr/local && \
    make && make install && \
    ldconfig && \
    cd / && rm -rf /tmp/sqlite-autoconf-*

# Rust installieren (nur für Build)
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:$PATH"

# SQLite3-Pfad für Python setzen
ENV LD_LIBRARY_PATH="/usr/local/lib:$LD_LIBRARY_PATH"

# Python Dependencies installieren
COPY requirements.txt .
# App-Code für -e . Installation kopieren
COPY ./app /app
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install langchain-chroma

# Final stage - nur Runtime
FROM base as final

# pip Proxy-Konfiguration auch für final stage
RUN pip config set global.proxy http://proxy.th-wildau.de:8080

# apt Proxy-Konfiguration für final stage
RUN echo 'Acquire::http::Proxy "http://proxy.th-wildau.de:8080";' > /etc/apt/apt.conf.d/01proxy && \
    echo 'Acquire::https::Proxy "http://proxy.th-wildau.de:8080";' >> /etc/apt/apt.conf.d/01proxy

# Runtime-Dependencies (bullseye package names)
RUN apt-get update && apt-get install -y \
    poppler-utils \
    tesseract-ocr \
    libjpeg62-turbo \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# SQLite3 Libraries von builder kopieren
COPY --from=builder /usr/local/lib/libsqlite3* /usr/local/lib/
COPY --from=builder /usr/local/bin/sqlite3 /usr/local/bin/
RUN ldconfig

# SQLite3-Pfad für Python setzen
ENV LD_LIBRARY_PATH="/usr/local/lib:$LD_LIBRARY_PATH"

# Python packages von builder kopieren
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# App-Code von builder kopieren (bereits mit -e . installiert)
COPY --from=builder /app /app

# Logs und Data Ordner erstellen
RUN mkdir -p /app/logs /app/uploads

# Port exponieren
EXPOSE 8000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Standard-Command für Development
CMD ["tail", "-f", "/dev/null"]