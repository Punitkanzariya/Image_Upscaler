# Use official lightweight Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copy Python requirements
COPY requirements.txt .

# ✅ Install torch first to avoid numpy runtime error
RUN pip install --no-cache-dir torch torchvision

# ✅ Install other dependencies
RUN pip install --no-cache-dir -r requirements.txt

# ✅ Reinstall numpy (important fix for PyTorch)
RUN pip install --no-cache-dir --force-reinstall numpy

# Copy the rest of the app
COPY . .

# Set the port environment variable (Render uses PORT)
ENV PORT 10000

# Start the Flask app with Gunicorn
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:10000", "--workers=1", "--threads=1", "--timeout=120"]
