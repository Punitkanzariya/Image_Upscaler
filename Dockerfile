# Use official lightweight Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# ✅ Install torch + torchvision first
RUN pip install --no-cache-dir torch torchvision

# ✅ Then install your other packages
RUN pip install --no-cache-dir -r requirements.txt

# ✅ Downgrade NumPy to a stable version
RUN pip install --no-cache-dir numpy==1.26.4

# Copy the rest of the app
COPY . .

# Set the port environment variable (Render uses PORT)
ENV PORT 10000

# Start the Flask app with Gunicorn
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:10000", "--workers=1", "--threads=1", "--timeout=120"]
