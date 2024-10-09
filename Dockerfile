FROM python:3

# Install distutils and other necessary packages
RUN apt-get update && apt-get install -y setuptools ffmpeg libsm6 libxext6 && apt-get clean

# Copy application files
COPY . /usr/src/app

# Set the working directory
WORKDIR /usr/src/app

# Install required Python packages
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Command to run the application
CMD ["python3", "manage.py", "runserver", "0.0.0.0:8000"]
