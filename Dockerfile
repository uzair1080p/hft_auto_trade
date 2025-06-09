# ---------- base image ----------
    FROM python:3.10-slim

    # Allow the build to choose a different requirements list (defaults to requirements.txt)
    ARG REQS=requirements.txt
    
    WORKDIR /app
    
    # ---------- system deps ----------
    RUN apt-get update && apt-get install -y --no-install-recommends \
            build-essential gcc \
        && rm -rf /var/lib/apt/lists/*
    
    # ---------- python deps ----------
    # Copy whichever requirements file was specified by the build arg
    COPY ${REQS} /tmp/requirements.txt
    RUN pip install --no-cache-dir -r /tmp/requirements.txt && \
        rm /tmp/requirements.txt
    
    # ---------- project ----------
    COPY . .
    
    # make sure the shell script is executable inside the image
    RUN chmod +x /app/entrypoint.sh
    
    ENTRYPOINT ["/app/entrypoint.sh"]