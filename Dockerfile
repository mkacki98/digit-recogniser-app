FROM python:3.9.15
COPY . /app
WORKDIR /app

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN pip3 install -r requirements.txt
RUN pip3 install torch torchvision torchaudio tensorboard 

ENTRYPOINT [ "python" ]

CMD ["app.py"]