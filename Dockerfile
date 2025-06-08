FROM python:3.10-slim

WORKDIR /app

# Install system-level dependencies including TA-Lib
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    curl \
    wget \
    make \
    libffi-dev \
    libssl-dev \
    python3-dev \
    git \
    && wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz \
    && tar -xvzf ta-lib-0.4.0-src.tar.gz \
    && cd ta-lib && ./configure --prefix=/usr && make && make install \
    && cd .. && rm -rf ta-lib* \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

ENTRYPOINT ["bash"]