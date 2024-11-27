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
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

mlflow.set_tracking_uri('http://127.0.0.1:5005')
mlflow.set_experiment('IMDB_Model_Training_1127')


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

# 데이터 전처리 함수 정의
def prepare_data():
    # 데이터 로드
    df = pd.read_csv(data_path, index_col=0)
    dataset = Dataset.from_pandas(df)
    dataset = dataset.train_test_split(test_size=0.2)
    print(f'데이터 로드 완료')
    # 라벨 딕셔너리 생성
    label2id = {'positive': 1, 'negative': 0}

# 토크나이저 적용
    tokenizer = load_tokenizer()
    dataset = dataset.map(
        lambda x: tokenizer(
            x['review'], # 토크나이저 적용할 데이터
            padding='max_length', # 최대 길이 이상 데이터는 잘라냄
            truncation=True, # 최대 길이 이상 데이터는 잘라냄
            max_length=300, # 최대 길이 설정
            return_tensors=None # mlflow만 사용할 시 None으로,  'pt'는 pytorch tensor로 반환하는 것을 의미
        ), 
        batched=True
    )
# 라벨 적용
    dataset = dataset.map(lambda x: {'label': label2id[x['sentiment']]})
    dataset_saved_path = os.path.join(dataset_path, f'processed_dataset_{timestamp}.json')
    dataset.save_to_disk(dataset_saved_path)
    # return dataset, label2id
    print(f'데이터 전처리 완료')
    return dataset, label2id, dataset_saved_path

# 모델 학습 및 평가 함수 정의
def train_model(dataset, label2id):
    # ti = kwargs['ti']

    os.makedirs(output_path, exist_ok=True)
    

    # 라벨 설정. label2id 딕셔너리 키 값을 뒤집음
    id2label = {0: 'negative', 1: 'positive'}
    # 모델 설정
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id
    )
    # 학습 설정
    args = TrainingArguments(
        output_dir=output_path,
        overwrite_output_dir=True,
        num_train_epochs=2,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=2e-5,
        eval_strategy='epoch'
    )

    # Trainer 설정
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        tokenizer=load_tokenizer(),
        compute_metrics=compute_accuracy    
    )
    train_result = trainer.train()
    
    mlflow.autolog()
    # mlflow에 모델 저장
    mlflow.pytorch.log_model(model, model_name)
    mlflow.log_params({"model_name": model_name}) 
    model_saved_path = os.path.join(model_path, f'model_{timestamp}.pkl')
    os.makedirs(model_path, exist_ok=True)
    joblib.dump(model, model_saved_path)
    print(f'모델 학습 완료')
    return model, model_saved_path


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

    dataset, label2id, dataset_saved_path = prepare_data()
    model, model_saved_path = train_model(dataset, label2id)
    evaluation_result = evaluate_model(model, model_saved_path, dataset)
    slack_notification(evaluation_result, model_saved_path, dataset_saved_path)    
