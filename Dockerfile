# Use an official Python runtime as a parent image
FROM python:3

# Install system dependencies for building Python packages and enabling OpenCV
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    ffmpeg libsm6 libxext6

# Set the working directory inside the container
WORKDIR /usr/src/app

# Copy the local codebase into the container
COPY . /usr/src/app

# Upgrade pip to the latest version to avoid compatibility issues
RUN pip install --upgrade pip

# Install Python dependencies from requirements.txt
RUN pip install -r requirements.txt

# Specify the command to run the Django server
CMD ["python3", "manage.py", "runserver", "0.0.0.0:8000"]
