import pandas as pd
import joblib
from datetime import datetime

from airflow import DAG
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from airflow.operators.python import PythonOperator

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
    y_train = pd.read_json(ti.xcom_pull(key='y_train', typ='series'))
    
    rf_clf = RandomForestClassifier(
        n_estimators=100,
        random_state=123)
    rf_clf.fit(X_train, y_train)

    model_path = '/tmp/random_forest_model.pkl'    # tmp로 만들었다가, 다 사용하면 나중에 삭제 
    joblib.dump(rf_clf, model_path)  # 모델 저장. dump: 모델을 파일로 저장

    context['ti'].xcom_push(key='model_path', value=model_path)
    

## (3) 모델 평가 (파일 형태의 모델을 로드해서 평가 )
def evaluate_model(**context):
    ti = context['ti']
    model_path = ti.xcom_pull(key='model_path')
    model = joblib.load(model_path)    

    X_test = pd.read_json(ti.xcom_pull(key='X_test'))
    y_test = pd.read_json(ti.xcom_pull(key='y_test', typ='series'))

    predict = model.predict(X_test)
    print(f'predictions: {predict}')

    accuracy = accuracy_score(y_test, predict)
    print(f'Model accuracy: {accuracy}')


## (4) DAG 정의
dag = DAG(
    'iris_ml_pipeline_single_model',
    description='A simple ML pipeline with RandomForestClassifier',
    start_date=datetime(2024, 11, 22),
    schedule_interval='@daily', 
    catchup=False
)

prepare_data_task1 = PythonOperator(
    task_id='prepare_data',
    python_callable=prepare_data,
    provide_context=True,
    dag=dag
)
train_model_task2 = PythonOperator(
    task_id='train_model',
    python_callable=train_model,
    provide_context=True,
    dag=dag
)
evalutate_model_task3 = PythonOperator(
    task_id='evaluate_model',
    python_callable=evaluate_model,
    provide_context=True,
    dag=dag
)

prepare_data_task1 >> train_model_task2 >> evalutate_model_task3


# docker exec -it airflow-dags-container /bin/bash
# pip install mlflow==2.16.2
# pip install scikit-learn
# - 파이썬이 설치되어 있으니깐