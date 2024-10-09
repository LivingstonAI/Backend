FROM python:3
COPY . /usr/src/app
WORKDIR /usr/src/app
RUN pip install -r requirements.txt
# This below is to enable cv2 to work
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
CMD ["python3", "manage.py", "runserver", "0.0.0.0:8000"]
