FROM python:3.12-slim

# tc (iproute2) is needed inside client containers for netem; procps for debugging.
RUN apt-get update && apt-get install -y --no-install-recommends \
        iproute2 procps \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ /app/
ENV PYTHONUNBUFFERED=1
