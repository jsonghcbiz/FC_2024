# FC_2024
Practice codes used for FC Bootcamp

# 라이브러리 관리 팁 
- requirements.txt가 나중에 발생할 수 있는 라이브러리 의존성을 해결할 수 있음 
- poetry 사용 시 라이브러리 버전 관리 용이

---

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
- -t '이름' => 이미지 이름 설정

4. 도커 컨테이너 실행
> docker run --name airflow-container -d -p 8080:8080 airflow-image
- > -d => 컨테이너 백그라운드 실행
- > -p 8080:8080 => 호스트의 8080포트와 컨테이너의 8080포트를 맵핑
- > --name '이름' => 컨테이너 이름 설정
- > airflow-image => 앞에서 설정한 이미지 이름

5. 컨테이너 접속 후 어드민 계정 생성
> docker exec -it airflow-container /bin/bash
   - it: interacive terminal (가상터미널)
   - airflow-container => 컨테이너 이름
   - /bin/bash => 쉘 실행

6. 어드민 계정 생성 명령어 실행
> airflow users create --username admin --firstname jensong --lastname song --role Admin --email jsong.hcbiz@gmail.com --password 1234
또는
> airflow users create \
--username jesong \
--firstname jenn \
--lastname song \
--role Admin \
--email jsong.hcbiz@gmail.com \
--password 1234

### (3) Docker Compose 사용
1. VScode에서 docker-compose.yml 파일 생성

2. docker-compose.yml 파일 정의

3. docker-compose 실행
> docker compose up --build
- docker compose: docker-compose.yml 파일을 사용하여 컨테이너 실행
- up: 컨테이너 실행
- --build: 이미지 빌드

4. docker-compose 컨테이너 접속 후 어드민 생성
- > docker exec -it airflow-dags-container /bin/bash
- > airflow users create --username admin --firstname jensong --lastname song --role Admin --email jsong.hcbiz@gmail.com --password 1234

---

## ML 파이프라인 구축 


---

### Q&A

- Q.docker compose down 하면 백그라운드도 같이 down되나요?
    - down 명령어는 컨테이너를 중지 및 삭제하는 것이 포함되어 있음

docker build 할때 dockerfile 에 airflow admin 계정 생성 명령어까지 포함시켜두면 매번 새로 빌드할때 기본으로 관리자 계정이 만들어지게 할 수도 있는건가요?

- 네!

docker 구동중인 컴퓨터를 끌 때는 stop 했다 추후 다시 start 하면 컨테이너를 유지할 수 있고, 또는 변경된 컨테이너를 새로 이미지로 저장 후 불러와 사용할수 있다는듯한데, 실제로는 어떤 방식이 더 자주 사용되나요?

- 실제로는 컴퓨터를 안끄죠 => EC2, GCP 서버를 호스팅해서 올릴테니깐요!
- EC2 => docker설치한 다음 pull (private docker container하면 => 로그인한다음 (credential) => 개인계정에 등록된 cotainer를 pull 받을 수 있음)

### Q.

방금 기존 파일 수정하다가 놓친 거 같은데, DAG 에서 sklearn 의 모듈이 없다는 거 어떻게 해결하셨나요 ?

> docker exec -it airflow-dags-container /bin/bash

- 터미널 접속

> pip install scikit-learn

- 파이썬이 설치되어 있으니깐

## Q&A

### Q.docker compose down 하면 백그라운드도 같이 down되나요?

- down 명령어는 컨테이너를 중지 및 삭제하는 것이 포함되어 있습니다.

### Q.Docker build 후 순서가 어떻게 되나요? Compose yml파일 가지고 compose up 한 뒤에 컨테이너 런 하는게 맞을까요? Compose 안하고 run 한 경우엔 그냥 나갔다가 다시 compose 하고 하면 되는거죠?

- docker-compose up --build
- docker-compose down

docker 만으로는 했을 때는 한계가 많았기 때문에
docker-compose 명령어

### Q.저도 기존 dags 들만 드고 새로 만든건 안보입니다

- 시간이 걸립니다.. (20분)

### Q.

Mlflow/ wandb
Airflow/ Make => GPT
이런 식으로 대응되는 느낌이군요, 전자들로도 기능은 충분한데 후자들이 아무래도 좀 더 편리한 부분도 있는것 같아요.

- make로 유저들이 사용하고 있는 서비스있어? => 10번 중에 1번은 꼭 문제가 생깁니다.
- 성공여부 실패한 사람들 다시 보내고.

### Q.

그렇군요, wandb는 그래도 연구쪽에서 많이 쓴다고 들었는데, 저도 n8n 시도해봤을때 생각보다 애매하게 에러도 많고 기능도 좀 그랬었는데 노코드 툴 자체는 심화해 사용하기 좀 그런가 보네요. => MVP 툴

- n8n
- 개인이 간단한 자동화

### Q.

알려주신대로 Docker 재시작 및 파일 밖에 뺐다 다시 넣었는데도 아직 등록이 안되네요 ㅠ 생각보다 오래걸리는군요

- 기다려보셔야 할 것 같아요.
- 오늘 저녁에라도 연락주시면 줌켜서 방법을 알려드리겠씁니다! :)

### Q.

airflow에 올라올때 active가 아니라 paused로 올라옵니다.

- 트리거를 주시면 됩니다.



# container 접속 후 mlflow ui 명령어로 mlflow 서버 실행


# 이제 오늘까지 수업한 걸로 담주 월에 OT로 프로젝트 시작하나요?
# - 네

# (1) MLFLOW - 모델 실험 설계
# (2) FastAPI - 모델 서빙
# (3) AirFlow - 워크플로우 자동화

# Readme도 최종 업데이트 내용 포함해서 업데이트 해주실 수 있을까요? 복습하는데 도움될 것 같습니다.

# 송호연 - 인프라에 대한

# SageMaker => 3~5명 => 1명
# 기존 시니어들이 주니어 3명 델꼬 했는데, 주니어 3명 = GPT

# Airflow users create —username 하는 부분을 Dockerfile 에 포함시켜두었는데 
# (CMD airflow web server 부분 뒤에), 
# 생성이 안되는것 같네요. 계정 생성까지는 도커에서 자동화하긴 어려운걸까요? => 됩니다! 타이밍의 문제.

# 신규 데이터 저장형태와 저장소와 관련한 가이드는 없는건가요?
# - 컨테이너 | csv, db (클라우드)
# 선택과 집중
# - 논문을 2~3개 -> 모델만들어줘 
# - 내 포지션을 어떻게 가져갈 것인가?


# 앞으로 2주간 이거 세개만 하면 되는 건가요!? => 내 인생 = 포폴 => 연봉
# MLFlow
# FastAPI 서빙
# Airflow - 워크플로우 자동화
# AI Engineer => 러닝커브

# 고통 = 성장 , 고통 = 고통
# - 새로운 게 매일 쏟아진다.

# Build 할때 docker file 정보를 사용하는거고, compose 시는 yml을 사용하는거죠? 
# 그래서 Compose up, down 을 통해서는 yml파일 설정을 튜닝해 조정 가능하는거고

# 참 Docker file 에 EXPOSE 5000 도 해줘야 하는거 맞죠?

# +강의
# Mlflow ui 서버 실행 명령어도 도커파일에 추가해두면 되겠네요. -> 뼈대 프레임 -> 튜닝

## Recommend

- slack => 명령어 => 트리거 돌아가면서 => 모델 파일을 슬랙으로 넘겨주고, 배포 완료 되었습니다 (정확도 몇% 향상되었습니다)
- mlflow => 벤치마크 툴 => 리퀘스트 테스트 진행 (docker, 로컬에서도 해보고.) => 기록으로 남겨서 면접볼 때 활용
- 200페이지 설계문서 => AB180


