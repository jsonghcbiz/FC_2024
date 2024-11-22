## mlflow 시작은 
# 터미널 1: MLflow 서버 시작
# > mlflow server --host 127.0.0.1 --port 5000

# 터미널 2: 스크립트 실행
# > python mlflow_temp_one.py

import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import numpy as np
import datetime

# mlflow 버전 확인 => mlflow.__version__

# 데이터 로드
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['Label'] = iris.target

# 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=123)

# 데이터 스케일링 (train과 test 따로 적용)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # fit_transform은 train에만
X_test_scaled = scaler.transform(X_test)        # test는 transform만


# MLflow 서버 설정
try:
    mlflow.set_tracking_uri('http://127.0.0.1:5000')
    print(f"MLflow Tracking URI: {mlflow.get_tracking_uri()}")
except Exception as e:
    print(f"Error setting MLflow tracking URI: {e}")

# 실험 생성
try:
    exp_name = 'iris-classification-single'
    exp = mlflow.get_experiment_by_name(exp_name)
    if exp is None:
        exp_id = mlflow.create_experiment(exp_name)
        exp = mlflow.get_experiment_by_name(exp_name)
    else:
        exp_id = exp.experiment_id

    # 실험 정보 출력
    print("\nExperiment Info:")
    print(f"Name: {exp_name}")
    print(f"Experiment ID: {exp_id}")
    print(f"Artifact Location: {exp.artifact_location}")
    print(f"Creation Time: {pd.to_datetime(exp.creation_time, unit='ms')}")
    
    mlflow.set_experiment(exp_name)
except Exception as e:
    print(f"Error creating/setting experiment: {e}")


# 기록을 위한 2가지 방법
# - (1) mlflow.start_run(), mlflow.end_run() => 내가 진짜 필요한 값만 기록하고 싶을 때 (3개)
# - (2) mlflow.autolog() => 자기가 알아서 중요한 메트릭,파라미터를 기록해 줍니다. 속도가 조금 느립니다. (11개)

# 자동 기록
mlflow.autolog()


# 단일 모델 학습 및 평가
with mlflow.start_run(run_name='iris-logistic-regression'):
    # 모델 정의 및 학습
    model = LogisticRegression(max_iter=1000, random_state=123)
    model.fit(X_train_scaled, y_train)
    
    # 예측
    train_pred = model.predict(X_train_scaled)
    test_pred = model.predict(X_test_scaled)
    
    # 성능 평가
    train_accuracy = accuracy_score(y_train, train_pred)
    test_accuracy = accuracy_score(y_test, test_pred)
    
    # 결과 출력
    print("\nLogistic Regression Results:")
    print(f"Train Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
