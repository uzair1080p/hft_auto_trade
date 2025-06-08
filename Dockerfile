FROM python:3.10-slim

WORKDIR /app

# Install system dependencies and build TA-Lib from source
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    wget \
    curl \
    libffi-dev \
    libssl-dev \
    python3-dev \
    git \
    make \
    && wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz \
    && tar -xvzf ta-lib-0.4.0-src.tar.gz \
    && cd ta-lib && ./configure --prefix=/usr/local && make && make install \
    && cd .. && rm -rf ta-lib* \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables to help find the TA-Lib shared object
ENV LD_LIBRARY_PATH="/usr/local/lib:$LD_LIBRARY_PATH"
ENV TA_LIBRARY_PATH="/usr/local/lib"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy the full app source code
COPY . .

# Default entrypoint
ENTRYPOINT ["bash"]