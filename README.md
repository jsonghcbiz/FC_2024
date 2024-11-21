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