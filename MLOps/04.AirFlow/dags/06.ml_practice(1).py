import pandas as pd
import joblib
from datetime import datetime

from airflow import DAG
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ML 파이프라인 
## (1) 데이터 준비   ** context로 데이터 전달할 경우, json 형태로 전달해야 함. 
def prepare_data(**context):
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target, name='target')

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=123)
    

    context['ti'].xcom_push(key='X_train', value=X_train.to_json())
    context['ti'].xcom_push(key='X_test', value=X_test.to_json())
    context['ti'].xcom_push(key='y_train', value=y_train.to_json(orient='records'))
    context['ti'].xcom_push(key='y_test', value=y_test.to_json(orient='records'))


## (2) 모델 학습 (모델은 파일로 저장). 
def train_model(**context):
    ti = context['ti']

    X_train = pd.read_json(ti.xcom_pull(key='X_train'))
    y_train = pd.read_json(ti.xcom_pull(key='y_train', type='series'))
    
    rf_clf = RandomForestClassifier(
        n_estimators=100,
        random_state=123
    )
    rf_clf.fit(X_train, y_train)

    model_path = '/tmp/random_forest_model.pkl'    # tmp로 만들었다가, 다 사용하면 나중에 삭제 
    joblib.dump(rf_clf, model_path)

    context['ti'].xcom_push(key='model_path', value=model_path)
    

## (3) 모델 평가 (파일 형태의 모델을 로드해서 평가 )
def evaluate_model(**context):
    ti = context['ti']
    model_path = ti.xcom_pull(key='model_path')
    model = joblib.load(model_path)    

    X_test = pd.read_json(ti.xcom_pull(key='X_test'))
    y_test = pd.read_json(ti.xcom_pull(key='y_test', type='series'))

    predict = model.predict(X_test)
    print(f'predictions: {predict}')

    accuracy = accuracy_score(y_test, predict)
    print(f'Model accuracy: {accuracy}')


## (4) DAG 정의
dag = DAG(
    'iris_ml_pipeline_single_model',
    default_args=default_args,
    description='A simple ML pipeline with RandomForestClassifier',
    start_date=datetime(2024, 11, 22),
    schedule_interval='@daily', 
    catchup=False
)
