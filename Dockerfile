# Use a Python base image
FROM python:3.9-slim

# Install system dependencies (required for librosa and WAV processing)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy Python dependencies file
COPY src/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ /app/

# Mount data directory (defined in docker-compose.yml)
VOLUME /app/data

# Entrypoint command (e.g., run the mapping script)
CMD ["python", "main.py"]