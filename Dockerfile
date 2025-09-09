FROM python:3.12-slim

# Install system dependencies including build tools and headers
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    ffmpeg \
    libsm6 \
    libxext6 \
    libffi-dev \
    libssl-dev \
    libpq-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /usr/src/app

# Copy application files
COPY . .

# Upgrade pip, setuptools, and wheel
RUN pip install --upgrade pip setuptools wheel

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Run the application
CMD ["python3", "manage.py", "runserver", "0.0.0.0:8000"]
