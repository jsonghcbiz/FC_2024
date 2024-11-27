# 라이브러리 설정
import os
import pandas as pd
import numpy as np
import requests
from datetime import datetime
from datasets import load_dataset, Dataset, load_from_disk
import joblib

# 평가 지표 라이브러리
import evaluate
from evaluate import load
import torch

# mlflow 라이브러리
import mlflow
import mlflow.pytorch

# transformers 라이브러리
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments, pipeline   

# airflow 라이브러리
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.slack.operators.slack_webhook import SlackWebhookOperator


# 환경 변수 설정
os.environ['NO_PROXY'] = '*'  # airflow로 외부 요청할 때 이슈가 있음. 하여 해당 코드 추가 필요
# 모델 준비
model_name = 'huawei-noah/TinyBERT_General_4L_312D'
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# 데이터 경로 설정
# current_path = os.getcwd()
current_path = os.getcwd()
data_path = os.path.join(current_path, 'data', 'IMDB_Dataset.csv')
dataset_path = os.path.join(current_path, 'data', 'tk_dataset')
output_path = os.path.join(current_path, 'train_dir')
model_path = os.path.join(current_path, 'tinybert_saved')
model_saved_path = os.path.join(model_path, 'model_20241128_001841.pkl')
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
dataset_saved_path = os.path.join(dataset_path, 'processed_dataset_20241128_001841.json')
mlflow.set_tracking_uri('http://127.0.0.1:5005')
mlflow.set_experiment('IMDB_Model_Training_1127')

dataset = load_from_disk(dataset_saved_path)
label2id = {0: 'negative', 1: 'positive'}
id2label = {0: 'negative', 1: 'positive'}
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, 
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id
)

# Airflow 기본 설정
default_args = {
    'owner': 'admin',
    'start_date': datetime(2024, 11, 27),
    'retries': 1,
    'env': {
        'NO_PROXY': '*',   # airflow로 외부 요청할 때 
        'PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION': 'python'   # 파이썬 버전 이슈 해결
    }
}

def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    return tokenizer

def compute_accuracy(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    accuracy = evaluate.load('accuracy')
    return accuracy.compute(predictions=predictions, 
                           references=labels)


def evaluate_model(model, model_saved_path, dataset):
    # ti = kwargs['ti']


    trainer = Trainer(
        model=model,
        tokenizer=load_tokenizer(),
        compute_metrics=compute_accuracy
    )
    # 모델 평가
    evaluation_result = trainer.evaluate(dataset['test'])
    mlflow.log_metrics({
        "test_loss": evaluation_result['eval_loss'],
        "test_accuracy": evaluation_result['eval_accuracy']
    })
    print(f'모델 평가 완료')
    return evaluation_result

def slack_notification(evaluation_result, model_saved_path, dataset_saved_path):
    message = f"""
*Model Training Completed*
• Dataset: {os.path.basename(dataset_saved_path)}
• Model: {os.path.basename(model_saved_path)}
• Training Results:
  - Test Loss: {evaluation_result['eval_loss']:.4f}
  - Test Accuracy: {evaluation_result['eval_accuracy']:.4f}
• Training Duration: {evaluation_result['eval_runtime']:.2f} seconds
• Samples Per Second: {evaluation_result['eval_samples_per_second']:.2f}
    """
    slack_webhook_url = 'https://hooks.slack.com/services/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'
    payload = {
        'text': message
    }
    response = requests.post(slack_webhook_url, json=payload)
    print(f'슬랙 알림 완료')


if __name__ == "__main__":

    # dataset, label2id, dataset_saved_path = prepare_data()
    # model, model_saved_path = train_model(dataset, label2id)
    evaluation_result = evaluate_model(model, model_saved_path, dataset)
    slack_notification(evaluation_result, model_saved_path, dataset_saved_path)    
