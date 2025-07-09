# Use lightweight Python base
FROM python:3.10-slim

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

# Copy app code
COPY . .

# Port Flask will run on
ENV PORT 10000

# Run the Flask app via Gunicorn
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:10000", "--workers=1", "--threads=1", "--timeout=120"]

