FROM python:3.10-slim

WORKDIR /app

# Install build tools and TA-Lib dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    wget \
    curl \
    libffi-dev \
    libssl-dev \
    python3-dev \
    git \
    && wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz \
    && tar -xvzf ta-lib-0.4.0-src.tar.gz \
    && cd ta-lib && ./configure --prefix=/usr && make && make install \
    && cd .. && rm -rf ta-lib* \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables so linker finds libta_lib.so
ENV LD_LIBRARY_PATH="/usr/lib:$LD_LIBRARY_PATH"
ENV TA_LIBRARY_PATH="/usr/lib"

# Install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app source
COPY . .

# Default entrypoint
ENTRYPOINT ["bash"]