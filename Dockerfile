FROM nvidia/cuda:11.6.1-runtime-ubuntu20.04

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN apt-get install -y python3-pip
WORKDIR /app

COPY requirements-test.txt requirements-test.txt
RUN pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
RUN pip install -r requirements-test.txt


COPY data data
COPY models models
COPY options options
COPY util util
COPY test.py test.py
COPY train.py train.py

RUN ls