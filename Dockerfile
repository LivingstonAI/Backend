FROM python:3.12-slim

# Update package list and install necessary system packages
RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 && apt-get clean

# Set working directory
WORKDIR /usr/src/app

# Copy application files
COPY . .

# Upgrade pip and install setuptools
RUN pip install --upgrade pip setuptools

# Install required Python packages
RUN pip install -r requirements.txt

# Run the application
CMD ["python3", "manage.py", "runserver", "0.0.0.0:8000"]
