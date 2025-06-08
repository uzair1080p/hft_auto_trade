FROM python:3.10-slim

WORKDIR /app

# ---------- system toolchain ----------
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        make \
        wget \
        curl \
        libffi-dev \
        libssl-dev \
        python3-dev \
        git && \
    rm -rf /var/lib/apt/lists/*

# ---------- build & register TA-Lib ----------
RUN wget -q http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar -xzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib && \
    ./configure --prefix=/usr && \
    make && \
    make install && \
    cd .. && rm -rf ta-lib ta-lib-0.4.0-src.tar.gz && \
    ln -s /usr/lib/libta_lib.so.0 /usr/lib/libta_lib.so && \
    ldconfig

# ---------- python deps ----------
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ---------- project ----------
COPY . .

ENTRYPOINT ["bash"]