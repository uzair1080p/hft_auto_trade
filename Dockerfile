# Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies and timezone support
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    curl \
    tzdata \
    && rm -rf /var/lib/apt/lists/*

# Set timezone to UTC (Binance uses UTC)
ENV TZ=UTC
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Default entrypoint
ENTRYPOINT ["bash"]