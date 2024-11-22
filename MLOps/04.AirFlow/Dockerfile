# Docker Container에도 파이썬을 포함하여 모듈, 라이브러리 설치 가능 
# 파이썬으로 시작. 파이썬 버전 3.10 기준. 버전 변경 가능
FROM python:3.10-slim       

# 환경 변수 설정  -> 추후에 airflow 관련된 파일들이 생성됨 
ENV AIRFLOW_HOME=/opt/airflow
ENV MLFLOW_HOME=/opt/mlflow
ENV PYTHONPATH="${PYTHONPATH}:${AIRFLOW_HOME}"

# 시스템 업데이트 및 필요한 패키지 설치
RUN apt-get update && \
apt-get install -y --no-install-recommends \
gcc \
libc-dev \
libpq-dev \
git \
&& apt-get clean \
&& rm -rf /var/lib/apt/lists/*

# requirements.txt 파일을 불러와서 패키지 설치
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 필요한 디렉터리 생성 및 권한 설정
RUN mkdir -p $AIRFLOW_HOME $MLFLOW_HOME && \
    chmod -R 755 $AIRFLOW_HOME $MLFLOW_HOME
 
# MLFlow, Airflow 홈 디렉터리 설정
WORKDIR $AIRFLOW_HOME

# dags, mlflow 폴더 복사
COPY dags/ $AIRFLOW_HOME/dags/
COPY mlflow/ $MLFLOW_HOME/

# 포트 노출 - airflow ui, mlflow ui, fastapi 접속 가능하게 
# 8080: airflow ui
# 5000: mlflow ui
# 8000: fastapi

EXPOSE 8080 5000 8000  


# Create startup script
RUN echo '#!/bin/bash\n\
airflow db init && \n\
airflow users create --username jesong --password 0000 --firstname jenn --lastname song --role Admin --email jsong.hcbiz@gmail.com && \n\
airflow webserver -p 8080 & \n\
airflow scheduler & \n\
mlflow server --host 0.0.0.0 --port 5000 & \n\
uvicorn main:app --host 0.0.0.0 --port 8000\n\
' > /start.sh && chmod +x /start.sh

# 서비스 실행
CMD ["/start.sh"]
