FROM python:3.10-slim

WORKDIR /app

# Step 1: Install build tools and dependencies
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
    && rm -rf /var/lib/apt/lists/*

# Step 2: Build and install TA-Lib C library
RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar -xvzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib && \
    ./configure --prefix=/usr && \
    make && \
    make install && \
    cd .. && \
    rm -rf ta-lib*

# Step 3: Ensure linker can find it
ENV LD_LIBRARY_PATH="/usr/lib:$LD_LIBRARY_PATH"
RUN ldconfig

# Step 4: Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Step 5: Copy source code
COPY . .

# Default shell
ENTRYPOINT ["bash"]