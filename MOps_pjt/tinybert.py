
# 라이브러리 설정
import os
import pandas as pd
import joblib
from datetime import datetime
from datasets import load_dataset, Dataset

import sklearn.model_selection
from sklearn.metrics import accuracy_score


from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
import torch

import mlflow
import mlflow.sklearn  # sklearn 모델을 로깅할 때 사용

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.slack.operators.slack_webhook import SlackWebhookOperator

## 외부 요청 이슈 해결
os.environ['NO_PROXY'] = '*'  # mac에서 airflow로 외부 요청할 때 이슈가 있음. 하여 해당 코드 추가 필요

## MLflow 환경 설정 (실제 환경에 맞게 수정)
mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', "http://127.0.0.1:5000"))
# mlflow.set_tracking_uri("http://127.0.0.1:5000")  # MLflow 서버 URI => mlflow ui
mlflow.set_experiment("IMDB_Model_Training")  # 실험 이름 설정

default_args = {
    'owner': 'admin',
    'start_date': datetime(2024, 11, 25),
    'retries': 1,
}


## 데이터 불러오고 전처리 진행 
def prepare_data(**context):
    data_path = 'IMDB_PJT/airflow/dags/IMDB_Dataset.csv'
    imdb_db = pd.read_csv(data_path, index_col=0)

    dataset = Dataset.dataset.from_pandas(imdb_db)
    dataset = sklearn.model_selection.train_test_split(test_size=0.2)
    
    label2id = {'positive': 1, 'negative': 0}  # 라벨 값 설정
    dataset = dataset.map(lambda x: {'label': label2id[x['sentiment']]}) # 라벨 값 매핑


    # XCom을 사용하여 데이터를 함수 간 전달
    context['ti'].xcom_push(key='train_dataset', value=dataset['train'].to_json())
    context['ti'].xcom_push(key='test_dataset', value=dataset['test'].to_json())
    context['ti'].xcom_push(key='label2id', value=label2id)

## 모델 학습 및 mlflow 로깅
def train_model(model_name, **context):
    ti = context['ti']
    train_dataset = pd.read_json(ti.xcom_pull(key='train_dataset'))
    test_dataset = pd.read_json(ti.xcom_pull(key='test_dataset'))
    label2id = ti.xcom_pull(key='label2id')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name = 'huawei-noah/TinyBERT_General_4L_312D'
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)

    def tokenize_batch(batch):
        return tokenizer(batch['review'], padding='max_length', truncation=True, max_length=300)
    
    train_dataset = train_dataset.map(tokenize_batch, batched=True, batch_size=None)
    test_dataset = test_dataset.map(tokenize_batch, batched=True, batch_size=None)

    train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
    test_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

    id2label = {0: 'negative', 1: 'positive'}
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=2, 
        id2label=id2label, 
        label2id=label2id
    ).to(device)


    mlflow.autolog()
    
    # 학습 모델 선택 및 학습
    with mlflow.start_run(run_name=model_name):
        model_path = f'/tmp/{model_name}_model.pkl'
        trainer = Trainer(
            model=model,
            args=TrainingArguments(output_dir=model_path, num_train_epochs=3),
            train_dataset=train_dataset,
            eval_dataset=test_dataset
        )
        torch.save(model.state_dict(), model_path)
        # 모델 학습
        trainer.train()

        mlflow.log_artifact(model_path, artifact_path="model")

        context['ti'].xcom_push(key=f'model_path_{model_name}', value=model_path)

## 모델 평가 및 mlflow 로깅
def evaluate_model(model_name, **context):
    ti = context['ti']
    model_path = ti.xcom_pull(key=f'model_path_{model_name}')
    test_dataset = pd.read_json(ti.xcom_pull(key='test_dataset'))
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name = 'huawei-noah/TinyBERT_General_4L_312D'
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model.to(device)

    predictions = model.predict(test_dataset)
    accuracy = accuracy_score(test_dataset['label'], predictions)

    # MLflow에 메트릭 로깅
    with mlflow.start_run(run_name=model_name):
        mlflow.log_params({"model_name": model_name})
        mlflow.log_metric("accuracy", accuracy)

    context['ti'].xcom_push(key=f'performance_{model_name}', value=accuracy)

## Slack 메시지 전송
def send_slack_notification(**context):
    ti = context['ti']
    # best_model = ti.xcom_pull(key='best_model')

    performance = ti.xcom_pull(key='performance_TinyBERT_General_4L_312D')
    message = (
        f"📊 **Model Performances:**\n"
        f"- 🌲 **Accuracy:** {performance}\n"
    )
    
    slack_notification = SlackWebhookOperator(
        task_id='send_slack_notification_task',
        slack_webhook_conn_id='slack_webhook',
        message=message,
        dag=context['dag']
    )
    # Slack 메시지를 실제로 전송
    slack_notification.execute(context=context)

## DAG 정의
dag = DAG(
    'IMDB_Model_Training',
    default_args=default_args,
    description='A TinyBERT model training pipeline with MLflow on IMDB dataset',
    schedule_interval='@daily',
    catchup=False
)

## Task 정의
prepare_data_task = PythonOperator(
    task_id='prepare_data',
    python_callable=prepare_data,
    provide_context=True,
    dag=dag,
)

train_model_task = PythonOperator(
    task_id='train_model',
    python_callable=train_model,
    op_kwargs={'model_name': 'TinyBERT'},
    dag=dag,
)

evaluate_model_task = PythonOperator(
    task_id='evaluate_model',
    python_callable=evaluate_model,
    op_kwargs={'model_name': 'TinyBERT'},
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

## Task 의존성 설정
prepare_data_task >> train_model_task >> evaluate_model_task >> slack_notification_task







