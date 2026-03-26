# Dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies needed for Pillow (image processing)
RUN apt-get update && apt-get install -y \
    libjpeg-dev \
    libpng-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Set Python env variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Copy requirements first (for Docker layer caching)
COPY requirements.txt .

# ⚡ Install CPU-only PyTorch first (saves ~600MB vs the default CUDA version)
RUN pip install --no-cache-dir torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu

# Install the rest of your dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy your app and model files into the container
COPY app.py .
COPY models/ ./models/

# Expose port (Railway injects $PORT automatically)
EXPOSE 8000

# Start the FastAPI server
CMD uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}