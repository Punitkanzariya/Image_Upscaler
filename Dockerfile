FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies (note: no libgthread)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* && apt-get clean

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir torch==2.0.1 torchvision==0.15.2


# Copy app files
COPY . .

# Create folders and set permissions
RUN mkdir -p static/uploads static/results weights && \
    chmod -R 755 static/uploads static/results weights

# Environment
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Port
EXPOSE 10000

# Start app with Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:10000", "--workers", "1", "--timeout", "600", "--max-requests", "100", "--max-requests-jitter", "10", "app:app"]
