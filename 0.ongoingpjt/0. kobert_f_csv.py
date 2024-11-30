# pip install 'git+https://github.com/SKTBrain/KoBERT.git#egg=kobert_tokenizer&subdirectory=kobert_hf'
############### ------------------------------라이브러리 설정------------------------------ ###############
#----------------------------------------------------------------------------------------------------#
# 기본 라이브러리
import pandas as pd                 # 기본 데이터 처리
import numpy as np                  # 숫자 처리
import os                           # 시스템 경로 설정
import sys                          # 시스템 경로 설정
import time                         # 시간 설정
import pytz                         # 시간대 설정
from datetime import datetime       # 시간 설정
import requests                     # 외부 요청

# 데이터 처리
from datasets import Dataset,  DatasetDict                          # 데이터 처리
from sklearn.feature_extraction.text import CountVectorizer         # 벡터 처리
from sklearn.model_selection import KFold, train_test_split         # 데이터 분할
import matplotlib.pyplot as plt                                     # 시각화 처리
import matplotlib.font_manager as fm                                # 폰트 설정
import seaborn as sns                                               # 시각화 처리
import re

# MLflow 라이브러리
import mlflow                                   # mlflow 설정
import mlflow.pytorch                           # mlflow 설정. pytorch 모델 저장
from mlflow.tracking import MlflowClient        # mlflow 서버 접속

# 평가 지표 처리
import evaluate                                               # 평가 지표 처리 (예: accuracy, recall, precision, f1-score, etc.)
from evaluate import load                                     # load: 평가 지표 로드

# 트랜스포머 처리
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BertModel
from transformers import Trainer, TrainingArguments, pipeline   
import torch                                                            # 텐서 처리
import torch.nn.functional as F                                         # 텐서 처리. F: 텐서 함수 처리
from torch.nn import CrossEntropyLoss                                   # 텐서 처리. CrossEntropyLoss: 교차 엔트로피 손실 함수
from torch.utils.data import DataLoader         
import pickle                        # 텐서 처리. DataLoader: 데이터 로더

# 모델 토크나이저 처리
from kobert_tokenizer import KoBERTTokenizer


# from gluonnlp import nlp, Vocab
# # airflow 라이브러리
# from airflow import DAG
# from airflow.operators.python import PythonOperator
# from airflow.providers.slack.operators.slack_webhook import SlackWebhookOperator



############### ------------------------------기본 환경 설정------------------------------ ###############
#----------------------------------------------------------------------------------------------------#

# 환경 변수 설정
os.environ['NO_PROXY'] = '*'                                    # 환경 변수 설정 airflow로 외부 요청할 때 이슈가 있음. 하여 해당 코드 추가 필요
plt.rcParams['font.family'] = 'NanumGothic'                     # 폰트 설정

# 한국 시간 설정
def get_kst_time():
    kst = pytz.timezone('Asia/Seoul')
    return datetime.now(kst).strftime('%m%d_%H%M')
timestamp = get_kst_time()

# 경로 설정
current_path = os.getcwd()
data_path = os.path.join(current_path, 'data', 'cgv_review_sentiment_01.csv')    # cgv_review_sentiment_01.csv 또는 megabox_review_sentiment.csv
model_path = f'kobert_{timestamp}'
tokenizer_path = os.path.join(current_path, 'kobert_token')
# dataset_path = os.path.join(current_path, 'data', 'tk_dataset')
# output_path = os.path.join(current_path, 'train_dir')

# 모델 설정
model_name = 'skt/kobert-base-v1'   # 또는 monologg/kobert
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# mlflow 설정
mlflow.set_tracking_uri('http://127.0.0.1:5000')
mlflow.set_experiment('Megabox_Kobert')



############### ------------------------------코드 작성 ----------------------------- ###############
#-------------------------------------------------------------------------------------------------#



# 토크나이저 로드
def load_tokenizer():
    tokenizer = KoBERTTokenizer.from_pretrained(
        model_name, 
        sp_model_kwargs={'nbest_size': -1, 'alpha': 0.6, 'enable_sampling': True})
    # tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False) # 토크나이저 설정
    return tokenizer

# 평가 지표 설정
def compute_accuracy(predictions):
    predict = np.argmax(predictions.predictions, axis=1)  
    accuracy = evaluate.load('accuracy')
    return accuracy.compute(predictions=predict, 
                           references=predictions.label_ids)

# Airflow 기본 설정
default_args = {
    'owner': 'admin',
    'start_date': datetime(2024, 11, 29),
    'retries': 1,
    'env': {
        'NO_PROXY': '*',   # airflow로 외부 요청할 때 
        'PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION': 'python'   # 파이썬 버전 이슈 해결
    }
}


##### 데이터 로드 함수 정의 #####
def data_load(sample_size):
    df = pd.read_csv(data_path)
    # df = pd.read_csv(data_path)
    print("컬럼 확인:", df.columns)  # Check the columns
    print("데이터 타입 확인:", df.dtypes)  # Check data types
    print("데이터 상위 몇 행 확인:", df.head())  # Display first few rows
    
    # df = df.sample(n=sample_size, random_state=123)
    df = df.sample(n=min(sample_size, len(df)), random_state=123)
    print(f'데이터 샘플링 완료: {len(df)} rows')

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=123)
    train_df = Dataset.from_pandas(train_df)
    test_df = Dataset.from_pandas(test_df)
    dataset = ({
        'train': train_df,
        'test': test_df
    })
    print(f'데이터 로드 완료')
    return dataset

##### 데이터 전처리 함수 #####
def data_preprocess(dataset):
    start_time = time.time()
    def clean_text_function(examples):
        clean_text = []
        ## 텍스트 정제 ##
        for text in examples['review']:
            if text is None:
                clean_text.append('')
            else:
                text = re.sub(r'<.*?>', '', text)  # HTML 태그 제거
                text = re.sub(r'[^\w\s]', '', text)  # 특수문자 제거
                text = re.sub(r'\d+', '', text)  # 숫자 제거
                text = text.lower()  # 소문자로 변환
                text = text.strip()  # 문자열 양쪽 공백 제거
                text = text.replace('br', '')  # 'br' 태그 제거

                clean_text.append(text)
        return {'review': clean_text}
    
    dataset_clean = {}
    for split in dataset:
        dataset_clean[split] = dataset[split].map(clean_text_function, batched=True)
    print(f'데이터 정제 완료')
    
    ## 토크나이저 적용 ##
    tokenizer = load_tokenizer()
    dataset_tokenized = {}
    for split in dataset_clean:
        dataset_tokenized[split] = dataset_clean[split].map(
            lambda x: tokenizer(
                x['review'],
                padding='max_length',
                truncation=True,
                max_length=300,
                return_tensors=None
            ),
            batched=True
        )

    ## 라벨 딕셔너리 생성 ##
    label2id = {'1': 1, '0': 0}
    dataset_labeled = {}
    for split in dataset_tokenized:
        dataset_labeled[split] = dataset_tokenized[split].map(
            lambda x: {'label': label2id[str(x['sentiment'])]}
        )
    dataset_labeled = DatasetDict(dataset_labeled)
    # dataset_saved_path = os.path.join(dataset_path, f'processed_dataset_{timestamp}.json')
    # dataset_labeled.save_to_disk(dataset_saved_path)
    print(f'데이터 전처리 완료')
    end_time = time.time()
    preprocess_time = end_time - start_time
    return dataset_labeled, label2id, preprocess_time


##### 모델 설정 #####
class CustomBertClassifier(torch.nn.Module):
    def __init__(self, model_name, num_labels):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = torch.nn.Dropout(0.1)
        self.classifier = torch.nn.Linear(self.bert.config.hidden_size, num_labels)
        
    def forward(self, input_ids=None, attention_mask=None, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]  # [CLS] token representation
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
            
        return torch.nn.functional.log_softmax(logits, dim=1) if loss is None else (loss, logits)
    
##### 모델 학습 및 평가 함수 #####
def train_eval_model(dataset_labeled):
    # os.makedirs(output_path, exist_ok=True)

    ## 라벨 설정 ##
    id2label = {0: '0', 1: '1'}
    label2id = {'0': 0, '1': 1}
    
    ## 모델 설정 ##
    model = CustomBertClassifier(
        model_name,
        num_labels=len(label2id)
    )
    model.to(device)

    ## 학습 설정 ##
    args = TrainingArguments(
        output_dir=model_path,
        overwrite_output_dir=True,
        num_train_epochs=2,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=2e-5,
        eval_strategy='epoch', 
        save_strategy='steps',       
        save_steps=500, 
        save_total_limit=3
    )

    ## Trainer 설정 ##
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset_labeled['train'],
        eval_dataset=dataset_labeled['test'],
        tokenizer=load_tokenizer(),
        compute_metrics=compute_accuracy    
    )
    
    # trainer.save_model(model_path)

    ## mlflow 실행 ##
    mlflow.autolog()
    with mlflow.start_run():
        start_time = time.time()
        ## 모델 학습 ##
        trainer.train()
        end_time = time.time()
        training_time = end_time - start_time
        print(f'모델 학습 완료')

        ## 모델 평가 및 예측 ##
        evaluation = trainer.evaluate()
        predictions = trainer.predict(dataset_labeled['test'])
        accuracy_score = compute_accuracy(predictions)
        
        ## 평가 결과 저장 ##
        evaluation_result = {
            "model_name": model_name,
            "eval_loss": evaluation['eval_loss'],                # 평가 손실
            "eval_accuracy": evaluation['eval_accuracy'],        # 평가 정확도
            "predict_accuracy": accuracy_score['accuracy'],      # 예측 정확도
            "eval_runtime": evaluation['eval_runtime'],          # 평가 실행 시간
            "training_time_seconds": training_time               # 학습 시간
        }

        print(f'모델 평가 완료')

        ## mlflow 지표 저장 ##
        mlflow.log_metrics({'test_accuracy': accuracy_score['accuracy'],
                            'eval_loss': evaluation['eval_loss'],
                            'eval_accuracy': evaluation['eval_accuracy'],
                            'eval_runtime': evaluation['eval_runtime']})
        
        ## mlflow 파라미터 저장 ##
        mlflow.log_params({
            'model': model_name,
            'dataset': 'ratings',
            'sample_size': sample_size,
            'epochs': trainer.args.num_train_epochs,
            'batch_size': trainer.args.per_device_train_batch_size,
            'timestamp': timestamp
        })
        # mlflow.set_tags({'dataset': 'ratings',
        #                  'model': 'kobert',
        #                  'timestamp': timestamp})
    
        trainer.save_model(model_path)

        ## pkl로 모델 저장 ##
        model_save_pkl = os.path.join(model_path, f'model_{timestamp}.pkl') # pkl로도 저장
        with open(model_save_pkl, 'wb') as f:
            pickle.dump(model, f)
        mlflow_model_name = model_name.replace("/", "_")
        # joblib.dump(model, model_save_pkl) 

        ## mlflow에 모델 저장 ##
        mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path="model",
            registered_model_name=mlflow_model_name
        )
        print(f'모델 저장 완료: {model_save_pkl}')


    print(f'모델 학습 및 평가 완료')
    return model, model_path, model_name, evaluation_result

##### 모델 로드 함수 #####
# def load_saved_model(model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
#     model = CustomBertClassifier(model_name, num_labels=2)
#     model.load_state_dict(torch.load(model_path, map_location=device))
#     model.to(device)
#     # return model
# loaded_model = load_saved_model('path/to/saved/model.pkl')


##### 슬랙 알림 함수 #####
def slack_notification(evaluation_result, model_path, data_path, sample_size, preprocess_time):
    message = f"""
* Dataset 정보 *
- 데이터 경로: {os.path.basename(data_path)}
- 데이터 샘플링 개수: {sample_size}
- 학습 데이터 개수: {len(dataset['train'])}
- 평가 데이터 개수: {len(dataset['test'])}
- 데이터 전처리 시간: {preprocess_time:.2f} seconds

* Model 정보 *
- 모델명: {model_name}
- 저장 경로: {model_path}

* 학습 결과 *
- 평가 손실: {evaluation_result['eval_loss']:.4f}
- 평가 정확도: {evaluation_result['eval_accuracy']:.4f}
- 예측 정확도: {evaluation_result['predict_accuracy']:.4f}
- 학습 시간: {evaluation_result['training_time_seconds']:.2f} seconds

* 퍼포먼스 *
- 평가 실행 시간: {evaluation_result['eval_runtime']:.2f} seconds
- 초당 샘플 개수: {evaluation_result['eval_runtime'] / evaluation_result['predict_accuracy']:.2f}

* MLFlow 정보 *
- 실행 경로: {mlflow.get_artifact_uri()}
    """
    slack_webhook_url = 'https://hooks.slack.com/services/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'
    payload = {
        'color': '#00FF00',
        'text': message,
        'mrkdwn_in': ['text']
    }
    response = requests.post(slack_webhook_url, json=payload)
    print(f'슬랙 알림 완료')



##### 메인 함수 #####
if __name__ == "__main__":
    sample_size = 30000
    dataset = data_load(sample_size)
    dataset_labeled, label2id, preprocess_time = data_preprocess(dataset)
    model, model_path, model_name, evaluation_result = train_eval_model(dataset_labeled)
    slack_notification(evaluation_result, model_path, data_path, sample_size, preprocess_time)    
