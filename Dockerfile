FROM python:3.10-slim

WORKDIR /app

# Install TA-Lib dependencies and build tools
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    && wget https://launchpad.net/~mario-mariomedina/+archive/ubuntu/talib/+files/libta-lib0_0.4.0-oneiric1_amd64.deb \
    && wget https://launchpad.net/~mario-mariomedina/+archive/ubuntu/talib/+files/ta-lib0-dev_0.4.0-oneiric1_amd64.deb \
    && dpkg -i libta-lib0_0.4.0-oneiric1_amd64.deb \
    && dpkg -i ta-lib0-dev_0.4.0-oneiric1_amd64.deb \
    && rm *.deb \
    && apt-get remove -y wget \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

COPY . .

ENTRYPOINT ["bash"]