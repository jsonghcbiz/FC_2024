import pandas as pd   # 판다스
import numpy as np    # 넘파이
import requests       # 요청: 웹 페이지 요청
import json           # 자바스크립트 객체 표현

from sklearn.datasets import load_iris  # 붓꽃 데이터셋
from sklearn.model_selection import train_test_split  # 데이터 분할
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression  # 로지스틱 회귀
from sklearn.metrics import accuracy_score  # 정확도: 예측 결과와 실제 결과 비교
from sklearn.preprocessing import StandardScaler  # 표준화
from sklearn.ensemble import RandomForestClassifier  # 랜덤 포레스트
from sklearn.svm import SVC

import mlflow   # mlflow 라이브러리
import mlflow.sklearn  # mlflow와 연동
from mlflow.tracking import MlflowClient  # mlflow 클라이언트: 서버와 통신

# 데이터 준비
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['Label'] = iris.target

# 데이터 전처리
scaler = StandardScaler()
X_scaled = scaler.fit_transform(iris.data)

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, iris.target, test_size=0.2, random_state=123)



# mlflow 서버 설정
mlflow.set_tracking_uri("http://127.0.0.1:5000")  # dev.upstage.com:88003  
mlflow.set_experiment("iris_classification") 


mlflow.autolog()

models = {
    "LogisticRegression": LogisticRegression(
        max_iter=10, # 최대 반복 횟수
        C=1.0, # 규제강도
        solver='lbfgs',
        random_state=123
    ),
    "RandomForest" : RandomForestClassifier(
        n_estimators=100, # 트리의갯수
        random_state=123, 
        min_samples_leaf=2, 
        max_depth=10
    )
}


# Initialize variables for tracking the best model
best_accuracy = 0
best_model_name = None
best_model = None

# mlflow 실행
with mlflow.start_run(run_name="iris_classification"):
    for model_name, model in models.items():    
        with mlflow.start_run(run_name=model_name, nested=True):
            model.fit(X_train, y_train)
            # 예측
            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)
            train_accuracy = accuracy_score(y_train, train_pred)
            test_accuracy = accuracy_score(y_test, test_pred)
            print(f"{model_name} Train Accuracy: {train_accuracy}, Test Accuracy: {test_accuracy}")

            # 모델 파라미터 로깅
            mlflow.log_param("model_name", model_name)
            mlflow.log_params(model.get_params())
            mlflow.log_metric("train_accuracy", train_accuracy)
            mlflow.log_metric("test_accuracy", test_accuracy)

            print(f"Model {model_name} saved")
            print(f"Train Accuracy: {train_accuracy:.4f}")
            print(f"Test Accuracy: {test_accuracy:.4f}")

        # 최고 정확도 모델 추적
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_model_name = model_name
            best_model = model

print(f"\nBest Model: {best_model_name} with test accuracy: {best_accuracy:.4f}")



