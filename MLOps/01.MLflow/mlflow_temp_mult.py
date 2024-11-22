## mlflow 시작은 
# 터미널 1: MLflow 서버 시작
# > mlflow server --host 127.0.0.1 --port 5000

# 터미널 2: 스크립트 실행
# > python mlflow_temp_mult.py

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
from sklearn.model_selection import GridSearchCV
import numpy as np

# mlflow 버전 확인  => mlflow.__version__

# 데이터 로드
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['Label'] = iris.target

# 데이터 스케일링
scaler = StandardScaler()
X_scaled = scaler.fit_transform(iris.data)

# 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, iris.target, test_size=0.2, random_state=123)


# 여러 모델 학습
models = {
    'LogisticRegression': {
        'model': LogisticRegression(),
        'params': {
            'max_iter': [100, 200],
            'C': [1],
            'solver': ['lbfgs'], 
            'random_state': [123]
        }
    },
    'RandomForestClassifier': {
        'model': RandomForestClassifier(),
        'params': {
            'n_estimators': [100],
            'max_depth': [10, 20, 30],
            'random_state': [123]
        }
    },
    'SVC': {
        'model': SVC(),
        'params': {
            'C': [1],
            'kernel': ['linear'],
            'random_state': [123]
        }
    }
}


# mlflow 서버 설정
try: 
    mlflow.set_tracking_uri('http://127.0.0.1:5000')  # dev.upstage.com:88003
    print(f"MLflow Tracking URI: {mlflow.get_tracking_uri()}")
except Exception as e:
    print(f"Error setting MLflow Tracking URI: {e}")
# 실험 생성
try: 
    exp_name = 'iris-classification-test'
    exp = mlflow.get_experiment_by_name(experiment_name=exp_name)
    if exp is None:
        exp_id = mlflow.create_experiment(experiment_name=exp_name)
        exp = mlflow.get_experiment_by_name(experiment_name=exp_name)
    else: 
        exp_id = exp.experiment_id
    print(f"Artifact Location: {exp.artifact_location}")
    print(f"Creation Time: {exp.creation_time}")
    mlflow.set_experiment(exp_name)
except Exception as e:
    print(f"Error creating experiment: {e}")


mlflow.autolog()

# 모델 학습 & mlflow 기록
for model_name, model_info in models.items():
    with mlflow.start_run(run_name=model_name):
        
        # 그리드 서치
        grid_search = GridSearchCV(
            estimator = model_info['model'], 
            param_grid = model_info['params'], 
            cv=3, # 3-fold cross validation
            scoring='accuracy'
            )
        
        # 모델 학습
        grid_search.fit(X_train, y_train)
        train_pred = grid_search.predict(X_train)
        test_pred = grid_search.predict(X_test)
        
        # 평가
        train_accuracy = accuracy_score(y_train, train_pred)
        test_accuracy = accuracy_score(y_test, test_pred)

        # # 수동 기록
        # mlflow.sklearn.log_model(grid_search, model_name)
        # mlflow.log_params(grid_search.best_params_)
        # mlflow.log_metric('best_cv_score', grid_search.best_score_)
        # mlflow.log_metric('train_accuracy', train_accuracy)
        # mlflow.log_metric('test_accuracy', test_accuracy)



        print(f"\n최적 파라미터: {grid_search.best_params_}")
        print(f"최적 평가 점수: {grid_search.best_score_:.4f}")
        print(f"Train Accuracy: {train_accuracy:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")



# 기록을 위한 2가지 방법
# - (1) mlflow.start_run(), mlflow.end_run() => 내가 진짜 필요한 값만 기록하고 싶을 때 (3개)
# - (2) mlflow.autolog() => 자기가 알아서 중요한 메트릭,파라미터를 기록해 줍니다. 속도가 조금 느립니다. (11개)

