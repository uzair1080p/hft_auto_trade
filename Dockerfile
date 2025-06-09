# ---------- base image ----------
FROM python:3.10-slim

WORKDIR /app

# ---------- system deps ----------
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential gcc \
    && rm -rf /var/lib/apt/lists/*

# ---------- python deps ----------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---------- project ----------
COPY . .

# make sure the shell script is executable inside the image
RUN chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]