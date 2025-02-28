# Benutze das offizielle Python-Image als Basis
FROM python:3.8-slim

# Setze Umgebungsvariablen
ENV DEBIAN_FRONTEND=noninteractive \
    PATH="/root/.cargo/bin:$PATH"

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
    && apt-get clean

# Installiere Rust & Cargo
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y && \
    /bin/bash -c "source $HOME/.cargo/env"  

# Setze das Arbeitsverzeichnis auf /app
WORKDIR /app

# Pip aktualisieren und Abhängigkeiten installieren
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Kopiere den gesamten Code in den Container
COPY . /app

# Standardbefehl (falls du Python starten möchtest)
CMD ["tail", "-f", "/dev/null"]




