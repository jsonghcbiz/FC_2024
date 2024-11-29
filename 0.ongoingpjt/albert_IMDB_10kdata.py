############### ------------------------------라이브러리 설정------------------------------ ###############
 #----------------------------------------------------------------------------------------------------#
# 기본 라이브러리
import os                           # 시스템 경로 설정
import pandas as pd                 # 기본 데이터 처리
import numpy as np                  # 숫자 처리
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

# MLflow 라이브러리
import mlflow                                   # mlflow 설정
import mlflow.pytorch                           # mlflow 설정. pytorch 모델 저장
from mlflow.tracking import MlflowClient        # mlflow 서버 접속

# 자연어 처리
from wordcloud import WordCloud                                 # 워드 클라우드 처리
from collections import Counter                                 # 카운터 처리
import re                                                       # 정규 표현식 처리
import nltk                                                     # 자연어 처리
from nltk.tokenize import word_tokenize                         # 토큰화 처리
from nltk.corpus import stopwords                               # 불용어 처리
from nltk.stem import WordNetLemmatizer, PorterStemmer          # 표제어 처리

# 평가 지표 처리
import evaluate                                               # 평가 지표 처리 (예: accuracy, recall, precision, f1-score, etc.)
from evaluate import load                                     # load: 평가 지표 로드

# 텐랜스포머 처리
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW, Trainer, TrainingArguments, pipeline
import torch                                                            # 텐서 처리
import torch.nn.functional as F                                         # 텐서 처리. F: 텐서 함수 처리
from torch.nn import CrossEntropyLoss                                   # 텐서 처리. CrossEntropyLoss: 교차 엔트로피 손실 함수
from torch.utils.data import DataLoader                                 # 텐서 처리. DataLoader: 데이터 로더

import joblib                                                           # 모델 저장. joblib: 모델 저장 라이브러리


# from tensorflow.keras.preprocessing.sequence import pad_sequences     # 텐서 처리. pad_sequences: 시퀀스 패딩 처리
# from tensorflow.keras.preprocessing.text import Tokenizer             # 텐서 처리. Tokenizer: 텍스트 토크나이저

# # airflow 라이브러리
# from airflow import DAG
# from airflow.operators.python import PythonOperator
# from airflow.providers.slack.operators.slack_webhook import SlackWebhookOperator


############### ------------------------------기본 환경 설정------------------------------ ###############
#----------------------------------------------------------------------------------------------------#

# 환경 변수 설정
os.environ['NO_PROXY'] = '*'                                    # 환경 변수 설정 airflow로 외부 요청할 때 이슈가 있음. 하여 해당 코드 추가 필요
plt.rcParams['font.family'] = 'NanumGothic'                     # 폰트 설정
timestamp = datetime.now().strftime('%Y%m%d_%H%M')

# 경로 설정
current_path = os.getcwd()                                      # 현재 경로 설정
data_path = os.path.join(current_path, 'data', 'IMDB_Dataset.csv')
model_path = 'albert_lemmi'
tokenizer_path = os.path.join(current_path, 'tokenizer')
# dataset_path = os.path.join(current_path, 'data', 'tk_dataset')
# output_path = os.path.join(current_path, 'train_dir')

# 모델 
model_name = 'albert-base-v2'
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# 언어 처리 설정 (NLTK)
nltk_data_path = os.path.join(current_path, 'nltk_data')       # nltk 데이터 경로 설정
nltk.data.path.append(nltk_data_path)
try:
    nltk.download('punkt', download_dir=nltk_data_path)
    nltk.download('punkt_tab', download_dir=nltk_data_path)
    nltk.download('stopwords', download_dir=nltk_data_path)
    nltk.download('wordnet', download_dir=nltk_data_path)
    nltk.download('averaged_perceptron_tagger', download_dir=nltk_data_path)
    nltk.download('maxent_ne_chunker', download_dir=nltk_data_path)
    nltk.download('words', download_dir=nltk_data_path)
    print('NLTK 데이터 다운로드 완료')
except Exception as e:
    print(f'NLTK 데이터 다운로드 중 오류 발생: {e}')


# mlflow 설정
mlflow.set_tracking_uri('http://127.0.0.1:5000')
mlflow.set_experiment('IMDB_ALBERT_1129_10kdata')



############### ------------------------------코드 작성 ----------------------------- ###############
#-------------------------------------------------------------------------------------------------#

# 한국 시간 설정
def get_kst_time():
    kst = pytz.timezone('Asia/Seoul')
    return datetime.now(kst).strftime('%Y-%m-%d %H:%M')
timestamp = get_kst_time()

# 토크나이저 로드
def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    return tokenizer

# ML 평가 지표 설정
def compute_accuracy(predictions):
    predict = np.argmax(predictions.predictions, axis=1)
    accuracy = evaluate.load('accuracy')
    return accuracy.compute(predictions=predict,
                           references=predictions.label_ids)
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


##### 데이터 로드 함수 #####
def data_load(sample_size=10000):
    # 데이터 로드
    df = pd.read_csv(data_path)
    print("컬럼 확인:", df.columns.tolist())
    
    df = df.sample(n=sample_size, random_state=123)
    print(f'데이터 샘플링 완료')

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=123)
    train_df = Dataset.from_pandas(train_df)
    test_df = Dataset.from_pandas(test_df)  

    dataset = {
        'train': train_df,
        'test': test_df
    }
    print(f'데이터 분할 완료')

    return dataset



##### 데이터 전처리 함수 #####
def data_preprocess(dataset):
    start_time = time.time()
    def clean_text_function(examples):
        clean_text = []
        lemmatizer = WordNetLemmatizer()
        stop_words = set(stopwords.words('english'))
        ## 텍스트 정제 ##
        for text in examples['review']:
            try:
                text = re.sub(r'<.*?>', '', text)               # HTML 태그 제거
                text = re.sub(r'[^\w\s]', '', text)             # 특수문자 제거
                text = re.sub(r'\d+', '', text)                 # 숫자 제거
                text = text.lower()                             # 소문자로 변환
                text = text.strip()                             # 문자열 양쪽 공백 제거
                text = text.replace('br', '')                   # 'br' 태그 제거
                words = word_tokenize(text)
                ## 표제어 추출 ##
                lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
                text = ' '.join(lemmatized_words)
                ## 불용어 제거 ##
                filtered_words = [word for word in words if word not in stop_words]
                text = ' '.join(filtered_words)

                clean_text.append(text)
            except Exception as e:
                print(f'데이터 전처리 중 오류 발생: {e}')
        return {'review': clean_text}
    
    dataset_clean = {}
    for split in dataset: #split: train, test
        dataset_clean[split] = dataset[split].map(clean_text_function, batched=True)
    print(f'텍스트 정제 완료')

    ## 토크나이저 적용 ##
    tokenizer = load_tokenizer()
    dataset_tokenized = {}
    for split in dataset_clean: #split: train, test
        dataset_tokenized[split] = dataset_clean[split].map(
            lambda x: tokenizer(
                x['review'], # 토크나이저 적용할 데이터
                padding='max_length', # 최대 길이 이상 데이터는 잘라냄
                truncation=True, # 최대 길이 이상 데이터는 잘라냄
                max_length=300, # 최대 길이 설정
                return_tensors=None # mlflow만 사용할 시 None으로,  'pt'는 pytorch tensor로 반환하는 것을 의미
            ),
            batched=True
        )
    ## 라벨 적용 ##
    label2id = {'positive': 1, 'negative': 0}
    dataset_labeled = {}
    for split in dataset_tokenized: #split: train, test
        dataset_labeled[split] = dataset_tokenized[split].map(
            lambda x: {'label': label2id[x['sentiment']]}
        )
    ## 데이터 저장 ##
    dataset_labeled = DatasetDict(dataset_labeled)
    # dataset_saved_path = os.path.join(dataset_path, f'processed_dataset_{timestamp}.json')
    # dataset_labeled.save_to_disk(dataset_saved_path)
    print(f'데이터 전처리 완료')
    end_time = time.time()
    preprocess_time = end_time - start_time
    return dataset_labeled, label2id, preprocess_time



##### 모델 학습 및 평가 함수 #####
def train_evaluate_model(dataset_labeled):
    ## 라벨 설정 ##
    id2label = {0: 'negative', 1: 'positive'}
    label2id = {'negative': 0, 'positive': 1}

    ## 모델 설정 ##
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id
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
        eval_strategy='epoch'
        # load_best_model_at_end=True,
        # # early_stopping_patience=3,
        # metric_for_best_model='accuracy'
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
        ## 평가 결과 추출 ##
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
        ## 모델 저장 ##
        # trainer.save_model(model_path)

        ## 모델 태그 설정 ##
        mlflow.set_tags({'dataset': 'IMDB',
                         'model': 'albert',
                         'timestamp': timestamp})
        ## 모델 저장 ##
        mlflow.pytorch.log_model(
            model,
            artifact_path=model_path,
            registered_model_name='albert_imdb')

        ## pkl로 모델 저장 ##       
        model_save_pkl = os.path.join(model_path, f'model_{timestamp}.pkl') # pkl로도 저장
        torch.save(model, model_save_pkl)
        # joblib.dump(model, model_save_pkl)
        print(f'모델 저장 완료: {model_save_pkl}')
    print(f'모델 학습 및 평가 완료')
    return model, model_path, model_name, evaluation_result, training_time



##### 슬랙 알림 함수 #####
def slack_notification(evaluation_result, model_name, data_path, sample_size, training_time):
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
    sample_size = 10000
    dataset = data_load(sample_size)
    dataset_labeled, label2id, preprocess_time = data_preprocess(dataset)
    model, model_path, model_name, evaluation_result, training_time  = train_evaluate_model(dataset_labeled)
    slack_notification(evaluation_result, model_name, data_path, sample_size, training_time) 
