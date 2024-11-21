FROM python:3.10-slim

# 환경 변수 설정  -> 추후에 airflow 관련된 파일들이 생성됨 
ENV AIRFLOW_HOME=/usr/local/airflow

# 시스템 업데이트 및 필요한 패키지 설치
RUN apt-get update && \
    apt-get install -y gcc libc-dev

# Airflow 및 Slack Provider 설치
RUN pip install apache-airflow apache-airflow-providers-slack pandas

# Airflow 홈 디렉터리 생성 및 작업 디렉터리 설정
RUN mkdir -p $AIRFLOW_HOME
WORKDIR $AIRFLOW_HOME

# Airflow 데이터베이스 초기화
RUN airflow db init

# dags 폴더 복사
COPY dags/ $AIRFLOW_HOME/dags/

# 포트 노출 - airflow ui 접속 가능하게 
EXPOSE 8080

# Airflow 웹 서버 및 스케줄러 실행, 그리고 관리자 계정 생성
CMD airflow webserver -p 8080 & \
    airflow scheduler

    # airflow webserver: UI 서버
    # airflow scheduler: Worker 실행 서버