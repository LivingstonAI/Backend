FROM python:3.9

# Update package list and install necessary system packages
RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 && apt-get clean

# Copy application files
COPY . /usr/src/app

# Set the working directory
WORKDIR /usr/src/app

# Upgrade pip and install setuptools using pip
RUN pip install --upgrade pip setuptools

# Install required Python packages from requirements.txt
RUN pip install -r requirements.txt

# Command to run the application
CMD ["python3", "manage.py", "runserver", "0.0.0.0:8000"]
