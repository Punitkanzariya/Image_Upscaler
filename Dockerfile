FROM python:3.10-slim

WORKDIR /app

# ✅ Install required system packages including libgthread
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# ✅ Install PyTorch + torchvision
RUN pip install --no-cache-dir torch torchvision

# ✅ Install all other packages
RUN pip install --no-cache-dir -r requirements.txt

# ✅ Downgrade numpy for PyTorch compatibility
RUN pip install --no-cache-dir numpy==1.26.4

COPY . .

ENV PORT=10000

CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:10000", "--workers=1", "--threads=1", "--timeout=120"]
