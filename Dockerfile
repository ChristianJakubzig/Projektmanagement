# Benutze das offizielle Python-Image als Basis
FROM python:3.11-slim

# Proxy-Umgebungsvariablen setzen
ENV http_proxy=http://proxy.th-wildau.de:8080
ENV https_proxy=http://proxy.th-wildau.de:8080
ENV HTTP_PROXY=http://proxy.th-wildau.de:8080
ENV HTTPS_PROXY=http://proxy.th-wildau.de:8080

# Setze Umgebungsvariablen
ENV DEBIAN_FRONTEND=noninteractive \
    PATH="/root/.cargo/bin:$PATH"

# APT für Proxy konfigurieren
RUN echo 'Acquire::http::Proxy "http://proxy.th-wildau.de:8080";' > /etc/apt/apt.conf.d/01proxy && \
    echo 'Acquire::https::Proxy "http://proxy.th-wildau.de:8080";' >> /etc/apt/apt.conf.d/01proxy

# Installiere grundlegende Pakete für Python, C++ und andere benötigte Tools
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    g++ \
    clang \
    cmake \
    git \
    wget \
    curl \
    poppler-utils \
    tesseract-ocr \
    libjpeg-dev \
    zlib1g-dev \
    libtiff5-dev \
    libopenjp2-7-dev \
    libpq-dev \
    python3-dev \
    bzip2 \
    tar \
    gzip \
    ca-certificates \
    postgresql-client \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && apt-get clean

# Installiere Rust & Cargo (mit Proxy)
RUN curl --proxy http://proxy.th-wildau.de:8080 https://sh.rustup.rs -sSf | sh -s -- -y && \
    /bin/bash -c "source $HOME/.cargo/env"  

# Setze das Arbeitsverzeichnis auf /app
WORKDIR /app

# Pip aktualisieren und Abhängigkeiten installieren
COPY requirements.txt .
RUN pip install --proxy http://proxy.th-wildau.de:8080 --upgrade pip && \
    pip install --proxy http://proxy.th-wildau.de:8080 -r requirements.txt && \
    pip install --proxy http://proxy.th-wildau.de:8080 langchain-chroma

# Kopiere den gesamten Code in den Container
COPY . /app

# Exponiere den Port für FastAPI
EXPOSE 8000

# Standardbefehl (falls du Python starten möchtest)
CMD ["tail", "-f", "/dev/null"]