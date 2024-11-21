FROM python:3.8-slim


RUN apt-get update && apt-get install -y \
    build-essential \
    pkg-config \
    libhdf5-dev


WORKDIR /app

COPY app/requirements.txt .
RUN pip install -r requirements.txt

COPY app/ .

# CMD ["python", "mnist.py"]   => compose.yml 에서 덮어쓰기 때문에 삭제

# (1) docker build -t mnist-image .
# (2) docker run --name mnist-container mnist-image



# docker-compose
# 도커마무리 => Airflow

# Q. 컨테이너에있는 모델 폴더를 옮기려면
# - docker cp mnist-container:/app/model/trained_model.h5 ./trained_model.h5