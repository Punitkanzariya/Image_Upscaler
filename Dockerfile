# Use official Python image
FROM python:3.10-slim

# Set environment
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy project files
COPY . .

# Set default port for Flask
ENV PORT=8080

# Tell Fly.io to run this command
CMD ["gunicorn", "app:app", "--workers", "1", "--threads", "1", "--timeout", "120", "--bind", "0.0.0.0:8080"]
