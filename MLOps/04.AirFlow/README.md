path = /Users/jooeunsong/Documents/FC_Projects/04.AirFlow


# Airflow 

## 1. Dockerfile 설정
(1) 코드 설정

(2) dags 폴더 생성 
> mkdir dags

(2) 도커 이미지 빌드 
> docker build -t airflow-image . 
(3) 도커 컨테이너 실행
> docker run --name airflow-container -d -p 8080:8080 airflow-image

-d: 백그라운드 실행
-p 8080:8080 => 호스트의 8080포트와 컨테이너의 8080포트 맵핑 

(5) 컨테이너 접속 후 어드민 계정 생성
> docker exec -it airflow-container /bin/bash
-it: interactive terminal (가상 터미널 접속)

> airflow users create \
    --username admin \
    --firstname jenn \
    --lastname song \
    --role Admin \
    --email jsong.hcbig@gmail.com \
    --password 1234

(6) docker-compose 정의

(7) docker-compose 실행
> docker-compose up --build


(8) docker-compose 컨테이너 접속 후 어드민 생성 
> docker exec -it airflow-dags-container /bin/bash
> airflow users create --username admin --firstname jenn --lastname song--role Admin --email jsong.hcbig@gmail.com --password 1234

