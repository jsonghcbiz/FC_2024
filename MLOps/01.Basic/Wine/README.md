# 1. 프로젝트 폴더 생성
> cd Desktop
> cd upstage5_mlops
> upstage5_mlops code . # VSC가 열림
# 2. 가상환경 구축
(mac)
> python3.10 -m venv .venv (안되면 3.10 버전 이름 지우고 실행)
> source .venv/bin/activate
(win)
> python3.10 -m venv .venv (안되면 3.10 버전 이름 지우고 실행)
> .venv/Scripts/activate (안 되면 Set-ExecutionPolicy RemoteSigned -Scope CurrentUser 하고 실행)
- poetry
- Docker => pip install -r requriments.txt 설치
# 3. MLflow 설치
> pip install mlflow==2.16.2
> mlflow ui
from sklearn.datasets import load_iris # mlflow 의존성으로 함께 sklearn이 설치됩니다.
import pandas as pd
iris = load_iris()
iris.data
iris.target
iris.feature_names
pd.DataFrame(iris.data, columns=iris.feature_names) (편집됨) 






# 1. ML flow

### (1) 가상환경 설정

(mac)

> python -m venv .venv
> source .venv/bin/activate

(win)

> .venv/Scripts/activate

### (2) Install

> pip install mlflow

### (3) ML flow UI start

> mlflow ui
> http://127.0.0.1:5000

debug)
pkg_resources Not found

> pip install --upgrade pip
> pip install setuptools