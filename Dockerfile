# Python 3.11 slim image
FROM python:3.11-slim

WORKDIR /app

# Install audio dependencies
RUN apt-get update && apt-get install -y ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY . .

# Run server
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-10000}
