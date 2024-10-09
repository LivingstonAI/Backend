FROM python:3

# Update package list and install necessary system packages
RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 && apt-get clean

# Set working directory
WORKDIR /usr/src/app

# Copy application files
COPY . /usr/src/app

# Upgrade pip and install virtualenv
RUN pip install --upgrade pip
RUN pip install virtualenv

# Create a virtual environment
RUN virtualenv venv
# Activate the virtual environment and install dependencies
RUN . venv/bin/activate && pip install -r requirements.txt

# Command to run the application
CMD ["venv/bin/python", "manage.py", "runserver", "0.0.0.0:8000"]
