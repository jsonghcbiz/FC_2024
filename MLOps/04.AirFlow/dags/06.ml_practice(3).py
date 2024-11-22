import os
import pandas as pd
from datetime import datetime

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib
import mlflow
import mlflow.sklearn  # sklearn 모델을 로깅할 때 사용

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.slack.operators.slack_webhook import SlackWebhookOperator



os.environ['NO_PROXY'] = '*'  # mac에서 airflow로 외부 요청할 때 이슈가 있음. 하여 해당 코드 추가 필요

# MLflow 설정 (실제 환경에 맞게 수정)
mlflow.set_tracking_uri("http://127.0.0.1:5000")  # MLflow 서버 URI => mlflow ui
mlflow.set_experiment("Iris_Model_Training")  # 실험 이름 설정

default_args = {
    'owner': 'admin',
    'start_date': datetime(2024, 11, 22),
    # 'schedule_interval': '@daily',
    'retries': 1,
}

# 1. 데이터 준비 함수
def prepare_data(**context):
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    
    # XCom을 사용하여 데이터를 전달
    context['ti'].xcom_push(key='X_train', value=X_train.to_json())
    context['ti'].xcom_push(key='X_test', value=X_test.to_json())
    context['ti'].xcom_push(key='y_train', value=y_train.to_json(orient='records'))
    context['ti'].xcom_push(key='y_test', value=y_test.to_json(orient='records'))

# 2. 모델 학습 함수
def train_model(model_name, **context):
    ti = context['ti']
    X_train = pd.read_json(ti.xcom_pull(key='X_train'))
    y_train = pd.read_json(ti.xcom_pull(key='y_train'), typ='series')

    mlflow.autolog()
    # MLflow 실험 실행
    with mlflow.start_run(run_name=model_name):
        if model_name == 'RandomForest':
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif model_name == 'GradientBoosting':
            model = GradientBoostingClassifier(random_state=42)
        elif model_name == 'SVM':
            model = SVC()
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        # 모델 학습
        model.fit(X_train, y_train)

        # 모델 로깅
        mlflow.sklearn.log_model(model, model_name)
        mlflow.log_param("model_name", model_name)
        
        # 모델을 파일로 저장
        model_path = f'/tmp/{model_name}_model.pkl' # S3에도 업로드
        joblib.dump(model, model_path)

        context['ti'].xcom_push(key=f'model_path_{model_name}', value=model_path)

# 3. 모델 평가 함수
def evaluate_model(model_name, **context):
    ti = context['ti']
    model_path = ti.xcom_pull(key=f'model_path_{model_name}')
    model = joblib.load(model_path)

    X_test = pd.read_json(ti.xcom_pull(key='X_test'))
    y_test = pd.read_json(ti.xcom_pull(key='y_test'), typ='series')

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    # MLflow에 메트릭 로깅
    with mlflow.start_run(run_name=model_name):
        mlflow.log_metric("accuracy", accuracy)
    
    print(f"{model_name} Model Accuracy: {accuracy}")

    context['ti'].xcom_push(key=f'performance_{model_name}', value=accuracy)

# 4. 최고 성능 모델 선택
def select_best_model(**context):
    ti = context['ti']

    rf_performance = ti.xcom_pull(key='performance_RandomForest')
    gb_performance = ti.xcom_pull(key='performance_GradientBoosting')
    svm_performance = ti.xcom_pull(key='performance_SVM')

    performances = {
        'RandomForest': rf_performance,
        'GradientBoosting': gb_performance,
        'SVM': svm_performance
    }

    best_model = max(performances, key=performances.get)
    best_performance = performances[best_model]

    print(f"Best Model: {best_model} with accuracy {best_performance}")
    context['ti'].xcom_push(key='best_model', value=best_model)

# 5. Slack 메시지 전송 함수
def send_slack_notification(**context):
    ti = context['ti']
    best_model = ti.xcom_pull(key='best_model')
    rf_performance = ti.xcom_pull(key='performance_RandomForest')
    gb_performance = ti.xcom_pull(key='performance_GradientBoosting')
    svm_performance = ti.xcom_pull(key='performance_SVM')
    
    message = (
        f"🏆 **Best Model:** *{best_model}*\n\n"
        f"📊 **Model Performances:**\n"
        f"- 🌲 **Random Forest Accuracy:** {rf_performance}\n"
        f"- 🚀 **Gradient Boosting Accuracy:** {gb_performance}\n"
        f"- 🖥️ **SVM Accuracy:** {svm_performance}"
    )
    
    slack_notification = SlackWebhookOperator(
        task_id='send_slack_notification_task',
        slack_webhook_conn_id='slack_webhook',
        message=message,
        dag=context['dag']
    )
    
    # Slack 메시지를 실제로 전송
    slack_notification.execute(context=context)

# DAG 정의
dag = DAG(
    'iris_ml_training_pipeline_with_mlflow',
    default_args=default_args,
    description='A machine learning pipeline with MLflow logging on Iris dataset',
    schedule_interval='@daily',
    catchup=False
)

# Task 정의
prepare_data_task = PythonOperator(
    task_id='prepare_data',
    python_callable=prepare_data,
    provide_context=True,
    dag=dag,
)

train_rf_task = PythonOperator(
    task_id='train_random_forest',
    python_callable=train_model,
    op_kwargs={'model_name': 'RandomForest'},
    provide_context=True,
    dag=dag,
)

train_gb_task = PythonOperator(
    task_id='train_gradient_boosting',
    python_callable=train_model,
    op_kwargs={'model_name': 'GradientBoosting'},
    provide_context=True,
    dag=dag,
)

train_svm_task = PythonOperator(
    task_id='train_svm',
    python_callable=train_model,
    op_kwargs={'model_name': 'SVM'},
    provide_context=True,
    dag=dag,
)

evaluate_rf_task = PythonOperator(
    task_id='evaluate_random_forest',
    python_callable=evaluate_model,
    op_kwargs={'model_name': 'RandomForest'},
    provide_context=True,
    dag=dag,
)

evaluate_gb_task = PythonOperator(
    task_id='evaluate_gradient_boosting',
    python_callable=evaluate_model,
    op_kwargs={'model_name': 'GradientBoosting'},
    provide_context=True,
    dag=dag,
)

evaluate_svm_task = PythonOperator(
    task_id='evaluate_svm',
    python_callable=evaluate_model,
    op_kwargs={'model_name': 'SVM'},
    provide_context=True,
    dag=dag,
)

select_best_model_task = PythonOperator(
    task_id='select_best_model',
    python_callable=select_best_model,
    provide_context=True,
    dag=dag,
)

# Slack 메시지 전송 Task
slack_notification_task = PythonOperator(
    task_id='send_slack_notification',
    python_callable=send_slack_notification,
    provide_context=True,
    dag=dag
)

# Task 의존성 설정
prepare_data_task >> [train_rf_task, train_gb_task, train_svm_task]
train_rf_task >> evaluate_rf_task
train_gb_task >> evaluate_gb_task
train_svm_task >> evaluate_svm_task
[evaluate_rf_task, evaluate_gb_task, evaluate_svm_task] >> select_best_model_task >> slack_notification_task