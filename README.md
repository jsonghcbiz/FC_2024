# FC_2024
Practice codes used for FC Bootcamp

## MLflow
(1) 가상환경 구축
> python3.10 -m venv .venv
> source .venv/bin/activate


(2) MLflow 설치
> pip install mlflow (> pip install mlflow==2.16.2)

(3) MLflow UI 실행 (홈페이지 접속)
> mlflow ui
> http://127.0.0.1:5000

- 필요시
> pip install --upgrade pip
> pip install setuptools

(3) 프로젝트 생성
> mlflow server --backend-store-uri sqlite:///mlflow.db

## FastAPI
- 첫 번째, 데이터 타입 체크가 가능 => pydantic
- 두 번째, 비동기 방식으로 구동 => asyncio => python 3.5 버전에서 async/await
- 동기와 비동기의 차이는? ex) 스타벅스 카페 - 맨 앞에 있는 사람이 자바칩 프라프치노 벤티 얼음 많이 초콜렛 많이 시럽 추가 - 그 다음 사람이 기다려야 하면
  - 동기: 알바생 1명
  - 비동기: 알바생 N명 (부하 많이 걸리는 커피를 주문해도, 다음 알바생이 주문을 받을 수 있는 구조)
- 벤치마킹 지수를 보시면 FastAPI 상위권에 위치.

### (1) FastAPI 설치
> pip install "fastapi[standard]" # 표준 설치
> pip install tensorflow   # 모델 설치
또는 
> pip install fastapi uvicorn  # 기본 설치

### (2) Web 서버 실행
> fastapi dev main.py    # 개발 서버 실행
또는 
> uvicorn main:app --reload  # (구방식) 실제 서버 실행

### (3) 홈페이지 접속
> http://127.0.0.1:8000    # 기본 페이지
> http://127.0.0.1:8000/docs  # API 문서 페이지

---
## TinyBert Model
- 초소형 언어모델 (huggingface에 있는 모델을 다운받아서 학습까지 진행 예정)

### (1) 데이터 확보

### (2) 모델 다운로드

### (3) 모델 학습

### (4) 우리만의 언어모델 => Twitter Disater => 최신 트위터글들을 몇백개 불러와서 학습. 1시간 단위로 실시간 학습 => 모델 성능

---
- Teachable Machine

---

## Docker
- 컨테이너 안에 필요로 하는 것들을 담아두는 공간
- 개념은 VM과 같다. 다른 점은 VM 물리적 공간을 분리한다는 것. Docker Container 호스트OS 위에서 돌아갑니다. (물리 공간 분리X => 가볍다.)

### (1) Docker Desktop 설치
- Linux: https://docs.docker.com/desktop/install/linux-install/
- Mac: https://docs.docker.com/desktop/install/mac-install/
- => docker cli 명령어 사용 가능
> docker version

### (2) Docker Hub 회원가입
- https://app.docker.com/signup

### (3) Dockerfile 생성

### (4) Dockerfile 빌드
1. 이미지 빌드
   > docker build -t hello-docker .

2. 이미지가 잘 빌드되었는지 확인
   > docker images

3. 빌드된 이미지를 실행 (컨테이너를 띄운다)
   > docker run hello-docker

4. 파이썬 코드 업데이트 후 도커 이미지를 실행했는데, 로그는 그대로인 경우
   - 로컬에서 작업했기 때문에 컨테이너에 최신 파일이 업로드 되었다고 볼 수 없다
   - 파일이 업데이트가 된 경우에는 이미지를 다시 빌드해줘야 한다.


---

## Airflow

### (1) Dockerfile 설정
1. 코드 작성

2. dags 폴더 생성
> mkdir dags

3. 도커 이미지 빌드
> docker build -t airflow-image .

4. 도커 컨테이너 실행
> docker run --name airflow-container -d -p 8080:8080 airflow-image
- > -d => 컨테이너 백그라운드 실행
- > -p 8080:8080 => 호스트의 8080포트와 컨테이너의 8080포트를 맵핑

### (2) 컨테이너 접속 후 어드민 계정 생성
1. 컨테이너 접속
> docker exec -it airflow-container /bin/bash
    - it: interacive terminal (가상터미널)

2. 어드민 계정 생성 명령어 실행
> airflow users create --username admin --firstname inseop --lastname kim --role Admin --email inseop@gmail.com --password 123
또는
> airflow users create
--username jesong
--firstname jenn
--lastname song
--role Admin
--email jsong.hcbiz@gmail.com
--password 1234

### (3) Docker Compose 사용
1. docker-compose.yml 파일 생성

2. docker-compose 정의

3. docker-compose 실행
> docker compose up --build

4. docker-compose 컨테이너 접속 후 어드민 생성

### Q&A

- Q.docker compose down 하면 백그라운드도 같이 down되나요?
    - down 명령어는 컨테이너를 중지 및 삭제하는 것이 포함되어 있음