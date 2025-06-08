# ---------- base image ----------
    FROM python:3.10-slim

    WORKDIR /app
    
    # ---------- system deps (only what we still need) ----------
    RUN apt-get update && apt-get install -y --no-install-recommends \
            build-essential gcc && \
        rm -rf /var/lib/apt/lists/*
    
    # ---------- python deps ----------
    COPY requirements.txt .
    RUN pip install --no-cache-dir -r requirements.txt
    
    # ---------- project code ----------
    COPY . .
    
    ENTRYPOINT ["bash"]