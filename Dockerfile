FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install minimal OS dependencies for OpenCV and RealESRGAN
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy Python requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create folders for file uploads/results
RUN mkdir -p static/uploads static/results weights && \
    chmod -R 755 static/uploads static/results weights

# Environment configs
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Expose the app port
EXPOSE 10000

# Start with Gunicorn (single worker to avoid memory spike)
CMD ["gunicorn", "--bind", "0.0.0.0:10000", "--workers", "1", "--threads", "1", "--timeout", "300", "app:app"]
