# 기본 라이브러리
import pandas as pd
import os
import re
import json
from tqdm import tqdm
import gc
from pathlib import Path
import logging
from contextlib import contextmanager
from multiprocessing import Pool, cpu_count
import math
from collections import defaultdict
import numpy as np

# 시간 관련 라이브러리
from datetime import datetime
import pytz

# 파일 관련 라이브러리
import yaml 
import shutil
from glob import glob
from pprint import pprint
from itertools import product
# 모델 관련 라이브러리
import torch
import pytorch_lightning as pl
from rouge import Rouge # 모델의 성능을 평가하기 위한 라이브러리입니다.

from torch.utils.data import Dataset , DataLoader
from torch.optim.lr_scheduler import LambdaLR
from transformers import AutoTokenizer, BartForConditionalGeneration, BartConfig
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import Trainer, TrainingArguments
from transformers import EarlyStoppingCallback
from transformers import TrainerCallback

# wandb 라이브러리
import wandb # 모델 학습 과정을 손쉽게 Tracking하고, 시각화할 수 있는 라이브러리입니다.
korea_time = datetime.now(pytz.timezone('Asia/Seoul'))
kr_time = korea_time.strftime("%m%d_%H%M")

from torch.serialization import add_safe_globals
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

# from sklearn.model_selection import KFold

# Add this near the top of the file, after the imports
os.environ["TOKENIZERS_PARALLELISM"] = "True"
os.environ["TOKENIZERS_PARALLELISM_THREADS"] = "4" 
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # CUDA 오류 추적을 위한 동기 실행 활성화


########################################################
# 기본 설정
########################################################
tokenizer = AutoTokenizer.from_pretrained("digit82/kobart-summarization")
model_name = "digit82/kobart-summarization"

my_name = "JES"
team_name = "jsong-hcbiz-na"
team_project = "fc_test2"
device = torch.device('cuda:0' if torch.cuda.is_available()  else 'cpu')

DATA_DIR = "/data/ephemeral/home/baseline/data"
OUTPUT_DIR = "/data/ephemeral/home/NLP_JES/output"
LOGS_DIR = os.path.join(OUTPUT_DIR, 'logs')
CACHE_DIR = os.path.join(OUTPUT_DIR, 'cache')

# 로깅 디렉토리 지정

error_log_file_path = os.path.join(LOGS_DIR, 'oom_error_log.csv')

########################################################
# 메모리 관련 설정
########################################################
class ResourceManager:
    """Enhanced resource management with better error handling"""
    def __init__(self, logger):
        self.logger = logger
        self.initial_memory = self.get_current_memory()
        
    @staticmethod
    def get_current_memory():
        """Get current GPU memory usage"""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated()
        return 0
    
    @staticmethod
    def clean_cuda_memory():
        """Clean CUDA memory"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
    
    @contextmanager
    def cuda_memory_check(self):
        """Context manager for monitoring CUDA memory"""
        try:
            initial_memory = self.get_current_memory()
            yield
        finally:
            current_memory = self.get_current_memory()
            leaked = current_memory - initial_memory
            if leaked > 0:
                self.logger.warning(f"Memory leak detected: {leaked/1024**2:.2f}MB")
                torch.cuda.empty_cache()
    
    def log_resource_usage(self):
        """Log current resource usage"""
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated()/1024**2
            memory_cached = torch.cuda.memory_reserved()/1024**2
            self.logger.info(f"GPU Memory: Allocated={memory_allocated:.2f}MB, Cached={memory_cached:.2f}MB")

def clean_resources():
    """모든 실험 종료 후 리소스를 정리합니다."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.synchronize()
    wandb.finish()

# def delete_model(model):
#     del model
#     clear_gpu_cache()
#     gc.collect()  # Python의 GC가 메모리 해제를 도울 수 있음
#     print("모델과 메모리 해제 완료!!")

########################################################
# config 설정
########################################################

config_data = {
    "general": {
        "data_path": DATA_DIR,          # 데이터 경로 지정
        "output_dir": OUTPUT_DIR,        # 모델 최종 출력 값 저장 경로 지정
        "model_name": model_name,                                   # 모델 이름 지정
        "run_time": kr_time                                         # 실행 시간 지정
    },
    "tokenizer": {
        "encoder_max_len": 512,                                     # 인코더 최대 길이 지정. 초과 토큰은 잘라내거나 패딩 처리
        "decoder_max_len": 100,                                     # 디코더 최대 길이 지정
        "bos_token": f"{tokenizer.bos_token}",                      # 시작 토큰 지정
        "eos_token": f"{tokenizer.eos_token}",                      # 종료 토큰 지정
        # 특정 단어들이 분해되어 tokenization이 수행되지 않도록 special_tokens을 지정해줍니다.
        "special_tokens": ['#Person1#', '#Person2#', '#Person3#', '#Person4#', '#Person5#', '#Person6#', '#Person7#', '#PhoneNumber#', '#Address#', '#PassportNumber#']
    },
    "training": {
        # 학습 처리 관련 설정
        "overwrite_output_dir": True,                                # 출력 디렉토리 덮어쓰기 여부 지정     
        "seed": 42,                                                  # 랜덤 시드 지정
        "num_train_epochs": 12,                                      # 학습 에폭 수 지정
        "per_device_train_batch_size": 32,                           # 학습 배치 크기 지정
        "per_device_eval_batch_size": 32,                            # 평가 배치 크기 지정
        
        # 학습률, 스케줄러 관련 설정
        "learning_rate": 1e-5,                                       # 학습률 지정
        "warmup_ratio": 0.1,                                         # 스케줄러 초기 학습률 지정. 학습 초기 학습률을 선형적으로 증가하도록 비율 지정
        "weight_decay": 0.01,                                        # 가중치 감쇠 지정. 학습 중 가중치를 줄이는 기법으로, 과적합을 방지하는 데 도움
        "lr_scheduler_type": 'cosine',                               # 학습률 스케줄러 유형 지정
        "optim": 'adamw_torch',                                      # 최적화 알고리즘 지정
        "gradient_accumulation_steps": 1,                            # 그래디언트 누적 스텝 지정. GPU 메모리 부족 시 사용
        "early_stopping_patience": 3,                               # 조기 중단 패턴 지정. 평가 지표가 개선되지 않은 경우 중단
        "early_stopping_threshold": 0.001,                          # 조기 중단 임계값 지정
        
        # 평가 및 모델 저장 관련 설정
        "evaluation_strategy": 'epoch',                              # 평가 기준 지정
        "save_strategy": 'no',                                    # 모델 가중치 저장 기준 지정
        "save_total_limit": None,                                       # 학습 중 저장할 모델 수 지정
        "greater_is_better": True,                                  # 모델 성능 평가 지표 지정
        "load_best_model_at_end": False,                              # 최적 모델 마지막에 로드 여부 지정
        "fp16": True,                                                # 16비트 부동 소수점 사용 여부 지정
        
        # 로깅 관련 설정
        "logging_dir": LOGS_DIR,                                    # 로깅 디렉토리 지정
        "logging_strategy": "epoch",                                # 로깅 기준  지정
        "logging_steps": 1,                                         # 로깅 스텝 지정
        "report_to": "wandb",                                        # wandb 사용 여부 지정
        
        # 예측 관련 설정
        "predict_with_generate": True,                              # 평가 시 디코더 출력 생성 여부 지정
        "metric_for_best_model": "combined_score",                  # 최적 모델 평가 지표 지정 (combined_score는 rouge-1, rouge-2, rouge-l의 F-1 score를 평균한 값 *100)
        "generation_max_length": 100,                               # 평가 시 생성할 최대 길이 지정
        "generation_num_beams": 4,
        "no_repeat_ngram_size": 2,
        "length_penalty": 1.0,
        "repetition_penalty": 1.0,
        
        "do_train": True,                                           # 학습 여부 지정
        "do_eval": True,                                            # 평가 여부 지정
        "label_smoothing_factor": 0.1,                              # 라벨 스무딩 지정
        "max_grad_norm": 1.0,                                        # 그래디언트 클리핑 지정

        "num_workers": 4,                                               # num_workers 추가
        "pin_memory": torch.cuda.is_available(),                        # pin_memory 추가
        "tokenizer_num_proc": min(4, cpu_count()),  # CPU 코어 수를 고려하여 설정

    },
    "evaluation": {
        "generation_params": {
            "num_beams": 4,
            "no_repeat_ngram_size": 2,
            "length_penalty": 1.0,
            "max_length": 120,                  # 고정
            "min_length": 30,                   # 고정
            "repetition_penalty": 1.0,
            "temperature": 0.7,
            "top_p": 0.9,
            "do_sample": True
        }
    },
    # (선택) wandb 홈페이지에 가입하여 얻은 정보를 기반으로 작성합니다.
    "wandb": {
        "entity": "jsong-hcbiz-na",                        # wandb 엔티티 이름 지정
        "project": "fc_test2",                            # wandb 프로젝트 이름 지정
        "name": f"JES_{model_name}_{kr_time}"            # wandb run 이름 지정
    },
    "inference": {
        "ckt_path": "/data/ephemeral/home/NLP_JES/output/best_model",   # 모델의 checkpoint와 best model 저장 경로 설정
        "checkpoint_step": None,                                        # 나중에 모델 로드할 때를 위해 checkpoint 파라미터 설정 (None은 최종 모델을 로드하는 것을 의미)
        "result_path": "./prediction/",                                 # 예측 결과 저장 경로 설정
        "no_repeat_ngram_size": 2,                                      # 중복 생성 방지 위한 n-gram 크기 설정
        "early_stopping": True,                                         # 조기 중단 여부 지정
        "generate_max_length": 100,                                     # 생성 최대 길이 지정
        "num_beams": 4,                                                 # 빔 서치에서 생성할 빔 수 지정
        "batch_size" : 32,                                              # 평가 시 사용할 배치 크기 지정
        "length_penalty": 1.0,
        "repetition_penalty": 1.0,
        # 정확한 모델 평가를 위해 제거할 불필요한 생성 토큰들을 정의합니다.
        "remove_tokens": ['<usr>', f"{tokenizer.bos_token}", f"{tokenizer.eos_token}", f"{tokenizer.pad_token}"]
    },
    "bart_config": {
        "dropout": 0.2,
        "attention_dropout": 0.1,
        "activation_dropout": 0.1,
        "classifier_dropout": 0.1
        # "length_penalty": 1.0,
        # "repetition_penalty": 1.0,
        # "no_repeat_ngram_size": 2
    }
}

# config 저장 및 불러오기 
config_path = "./config.yaml"
with open(config_path, "w") as file:
    yaml.dump(config_data, file, allow_unicode=True)
with open(config_path, "r") as file:
    loaded_config = yaml.safe_load(file)

########################################################
# 캐쉬/메모리 관리 및 로깅 함수 정의
########################################################
def clean_cuda_cache():
    """CUDA 캐시를 정리하여 OOM 오류를 방지합니다."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        import gc
        gc.collect()
        print("CUDA cache cleaned")

def clean_resources():
    """리소스를 정리합니다."""
    clean_cuda_cache()
    wandb.finish()
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def setup_logging():
    """Configure logging with proper formatting and file output"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(LOGS_DIR)),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()


########################################################
# 데이터 가공 클래스 정의
########################################################
# 데이터 전처리를 위한 클래스로, 데이터셋을 데이터프레임으로 변환하고 인코더와 디코더의 입력을 생성합니다.
class Preprocess:
    def __init__(self, bos_token: str, eos_token: str, config: dict) -> None:
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.special_tokens = config['tokenizer']['special_tokens']
        self.cleaners = [
            self.remove_multiple_spaces,
            self.normalize_whitespace,
            self.normalize_special_tokens
        ]

    def remove_multiple_spaces(self, text):
        return ' '.join(text.split())
    def normalize_whitespace(self, text):
        return re.sub(r'[\s\xa0\u3000]+', ' ', text)
    def normalize_special_tokens(self, text):
        protected_tokens = {}
        for idx, token in enumerate(self.special_tokens):
            placeholder = f"__PROTECTED_TOKEN_{idx}__"
            base_token = token.replace('#', '')
            pattern = f"(?:#{base_token}#|{base_token})"
            text = re.sub(pattern, placeholder, text)
            protected_tokens[placeholder] = token

        cleaned_text = re.sub(r'[^가-힣a-zA-Z0-9.,!?\'\s]', ' ', text)
        cleaned_text = ' '.join(cleaned_text.split())  # Remove extra whitespace

        for placeholder, token in protected_tokens.items():
            cleaned_text = cleaned_text.replace(placeholder, token)
        
        return cleaned_text
    
    def clean_text(self, text):
        for cleaner in self.cleaners:
            text = cleaner(text)
        return text
    
    @staticmethod                    # 데이터셋 파일을 데이터프레임으로 변환하는 함수
    def make_set_as_df(file_path, is_train = True):
        if is_train:
            df = pd.read_csv(file_path)
            train_df = df[['fname','dialogue','summary']]
            return train_df
        else:
            df = pd.read_csv(file_path)
            test_df = df[['fname','dialogue']]
            return test_df

    def make_input(self, dataset, is_test=False):
        if is_test:
            encoder_input = dataset['dialogue'].apply(self.clean_text)      # 텍스트 정제 
            decoder_input = [self.bos_token] * len(dataset['dialogue'])
            return encoder_input.tolist(), list(decoder_input)
        else:
            encoder_input = dataset['dialogue'].apply(self.clean_text)
            decoder_input = dataset['summary'].apply(lambda x: self.bos_token + self.clean_text(str(x)))
            decoder_output = dataset['summary'].apply(lambda x: self.clean_text(str(x)) + self.eos_token)
            return encoder_input.tolist(), decoder_input.tolist(), decoder_output.tolist()

########################################################
# 데이터셋 생성 클래스 정의
########################################################
# Train에 사용되는 Dataset 클래스를 정의합니다.
class DatasetForTrain(Dataset):
    def __init__(self, encoder_input, decoder_input, labels, len):      # 데이터셋 클래스 초기화 함수
        self.encoder_input = encoder_input                              # 인코더 입력
        self.decoder_input = decoder_input                              # 디코더 입력
        self.labels = labels                                            # 디코더 출력
        self.len = len                                                  # 데이터셋 길이
    
    # 인코더(item)와 디코더(item2)를 처리 후 하나의 딕셔너리 데이터로 합치고 라벨 추가
    def __getitem__(self, idx):        
        item = {key: val[idx].clone().detach() for key, val in self.encoder_input.items()} # item[input_ids], item[attention_mask]
        item2 = {key: val[idx].clone().detach() for key, val in self.decoder_input.items()} # item2[input_ids], item2[attention_mask]
        item2['decoder_input_ids'] = item2['input_ids']             # item2: 디코더 입력 데이터
        item2['decoder_attention_mask'] = item2['attention_mask']   
        item2.pop('input_ids')              # 디코더 입력 데이터에서 input_ids 제거
        item2.pop('attention_mask')         # 디코더 입력 데이터에서 attention_mask 제거
        item.update(item2)                                              #item[input_ids], item[attention_mask] item[decoder_input_ids], item[decoder_attention_mask]
        item['labels'] = self.labels['input_ids'][idx]                  #item[input_ids], item[attention_mask] item[decoder_input_ids], item[decoder_attention_mask], item[labels]
        return item

    def __len__(self):
        return self.len

# Validation에 사용되는 Dataset 클래스를 정의합니다.
class DatasetForVal(Dataset):
    def __init__(self, encoder_input, decoder_input, labels, len):
        self.encoder_input = encoder_input
        self.decoder_input = decoder_input
        self.labels = labels
        self.len = len

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encoder_input.items()} # item[input_ids], item[attention_mask]
        item2 = {key: val[idx].clone().detach() for key, val in self.decoder_input.items()} # item2[input_ids], item2[attention_mask]
        item2['decoder_input_ids'] = item2['input_ids']
        item2['decoder_attention_mask'] = item2['attention_mask']
        item2.pop('input_ids')
        item2.pop('attention_mask')
        item.update(item2) #item[input_ids], item[attention_mask] item[decoder_input_ids], item[decoder_attention_mask]
        item['labels'] = self.labels['input_ids'][idx] #item[input_ids], item[attention_mask] item[decoder_input_ids], item[decoder_attention_mask], item[labels]
        return item

    def __len__(self):
        return self.len

# Test에 사용되는 Dataset 클래스를 정의합니다. 디코더 없이 인코더만 사용하고 각 샘플을 테스트 ID와 함께 출력
class DatasetForInference(Dataset):
    def __init__(self, encoder_input, test_id, len):
        self.encoder_input = encoder_input
        self.test_id = test_id
        self.len = len

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encoder_input.items()}
        item['ID'] = self.test_id[idx]
        return item

    def __len__(self):
        return self.len
    
########################################################
# 토큰화 함수
########################################################

class CustomTokenizer:
    def __init__(self, model_name, special_tokens=None):
        """토크나이저 초기화"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            if special_tokens:
                # Ensure special tokens are properly formatted
                special_tokens_dict = {'additional_special_tokens': special_tokens}
                num_added = self.tokenizer.add_special_tokens(special_tokens_dict)
                print(f"Added {num_added} special tokens to tokenizer")
                
                # Add special token handling
                self.special_token_ids = {
                    token: self.tokenizer.convert_tokens_to_ids(token)
                    for token in special_tokens
                }
        except Exception as e:
            print(f"토크나이저 초기화 실패: {str(e)}")
            raise

    def batch_tokenize(self, texts, padding=True, truncation=True, 
                      max_length=None, remove_tokens=None, **kwargs):
        """배치 토큰화 수행"""
        try:
            if isinstance(texts, str):
                texts = [texts]
                
            # Protect special tokens before tokenization
            protected_texts = []
            for text in texts:
                protected_text = text
                for token in self.special_token_ids.keys():
                    # Ensure special tokens are preserved exactly as is
                    protected_text = protected_text.replace(token, f" {token} ")
                protected_texts.append(protected_text)
            
            encoded = self.tokenizer(
                protected_texts,
                padding=padding,
                truncation=truncation,
                max_length=max_length,
                return_tensors="pt",
                **kwargs
            )
            
            return encoded
            
        except Exception as e:
            print(f"배치 토큰화 실패: {str(e)}")
            raise

    def __getattr__(self, name):
        """Delegate unknown attributes to the underlying tokenizer"""
        return getattr(self.tokenizer, name)

def tokenize_batch(batch, tokenizer, tokenize_kwargs):
    """Separate function for tokenizing a batch of texts"""
    return tokenizer(batch, **tokenize_kwargs)

def batch_tokenize(texts, tokenizer, max_length, num_proc=4, special_tokens=None, remove_tokens=None, **kwargs):
    """
    텍스트를 배치 단위로 토큰화하는 함수
    
    Args:
        texts: 토큰화할 텍스트 리스트
        tokenizer: 사용할 토크나이저
        num_proc: 병렬 처리에 사용할 프로세스 수
        return_tensors: 반환할 텐서 타입 ("pt" for PyTorch)
        padding: 패딩 여부
        add_special_tokens: 특수 토큰 추가 여부
        truncation: 최대 길이 초과 시 자르기 여부
        max_length: 최대 길이
        special_tokens: 추가할 특수 토큰 리스트
        remove_tokens: 제거할 토큰 리스트
    """
    try:
        print(f"토큰화 시작: {len(texts)} 개의 텍스트 토큰화 중...")
        
        # 기본 토큰화 설정
        tokenize_kwargs = {
            'padding': True,
            'truncation': True,
            'max_length': max_length,
            'add_special_tokens': True,
            'return_tensors': "pt"
        }
        
        tokenize_kwargs.update(kwargs)
        
        if special_tokens and not tokenizer.special_tokens_map.get('additional_special_tokens'):
            special_tokens_dict = {'additional_special_tokens': special_tokens}
            tokenizer.add_special_tokens(special_tokens_dict)
            print("스페셜 토큰 추가 완료")

        batch_size = max(1, len(texts) // num_proc)
        batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
        print(f"배치 크기: {batch_size}")
        
        print("병렬 토큰화 시작...")
        encoded_batches = []
        with Pool(num_proc) as p:
            # Using tqdm with regular starmap instead of istarmap
            tasks = [(batch, tokenizer, tokenize_kwargs) for batch in batches]
            encoded_batches = list(tqdm(
                p.starmap(tokenize_batch, tasks),
                total=len(batches),
                desc="토큰화 진행률"
            ))
        
        print("결과 병합 중...")
        encoded = {
            'input_ids': torch.cat([batch['input_ids'] for batch in encoded_batches]),
            'attention_mask': torch.cat([batch['attention_mask'] for batch in encoded_batches])
        }
        
        if remove_tokens:
            print("불필요한 토큰 제거 중...")
            for token in remove_tokens:
                token_id = tokenizer.convert_tokens_to_ids(token)
                if token_id != tokenizer.unk_token_id:
                    mask = encoded['input_ids'] != token_id
                    encoded['input_ids'] = encoded['input_ids'][mask].view(len(texts), -1)
                    encoded['attention_mask'] = encoded['attention_mask'][mask].view(len(texts), -1)
        
        print("토큰화 완료!")
        return encoded

    except Exception as e:
        print(f"토큰화 중 오류 발생: {str(e)}")
        raise

def special_token_handler():
    ''' normalize special tokens'''
    pass


########################################################
# 데이터셋 만들기
########################################################
@torch.no_grad()
def prepare_train_dataset(config, preprocessor, tokenizer):
    cache_dir = CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, 'tokenized_data.pt')
    
    if os.path.exists(cache_file):
        try:
            cached_data = torch.load(cache_file)
            print("-"*10, "캐시 토큰 데이터 로드 완료", "-"*10)
            # Ensure all required keys exist
            if all(k in cached_data for k in ['val_data', 'train_dataset', 'val_dataset']):
                return cached_data['val_data'], cached_data['train_dataset'], cached_data['val_dataset']
            else:
                print("캐시 데이터 형식이 맞지 않아 새로 생성합니다.")
                os.remove(cache_file)  # Remove invalid cache
        except Exception as e:
            print(f"캐시 로드 중 오류 발생: {str(e)}, 새로 생성합니다.")
            os.remove(cache_file)  # Remove invalid cache
    
    train_file_path = os.path.join(DATA_DIR,'train.csv')
    val_file_path = os.path.join(DATA_DIR,'dev.csv')

    train_data = preprocessor.make_set_as_df(train_file_path)
    val_data = preprocessor.make_set_as_df(val_file_path)

    encoder_input_train, decoder_input_train, decoder_output_train = preprocessor.make_input(train_data)
    encoder_input_val, decoder_input_val, decoder_output_val = preprocessor.make_input(val_data)
    print('-'*10, '데이터 로드 완료', '-'*10,)

    # Create CustomTokenizer instance with proper config
    custom_tokenizer = CustomTokenizer(
        config['general']['model_name'], 
        config['tokenizer']['special_tokens']
    )
    
    print('-'*10, '학습 데이터 토큰화 시작', '-'*10,)
    
    # Use the custom tokenizer's batch_tokenize method
    tokenized_encoder_inputs = custom_tokenizer.batch_tokenize(
        texts=encoder_input_train,
        max_length=config['tokenizer']['encoder_max_len'],
        remove_tokens=config['inference']['remove_tokens']
    )
    tokenized_decoder_inputs = custom_tokenizer.batch_tokenize(
        texts=decoder_input_train,
        max_length=config['tokenizer']['decoder_max_len'],
        remove_tokens=config['inference']['remove_tokens']
    )
    tokenized_decoder_ouputs = custom_tokenizer.batch_tokenize(
        texts=decoder_output_train,
        max_length=config['tokenizer']['decoder_max_len'],
        remove_tokens=config['inference']['remove_tokens']
    )
    
    train_inputs_dataset = DatasetForTrain(
        tokenized_encoder_inputs, 
        tokenized_decoder_inputs, 
        tokenized_decoder_ouputs,
        len(encoder_input_train))
    print('-'*10, '학습 데이터 토큰화 및 데이터셋 생성 완료', '-'*10,)
    
    print('-'*10, '검증 데이터 토큰화 시작', '-'*10,)
    val_tokenized_encoder_inputs = custom_tokenizer.batch_tokenize(
        texts=encoder_input_val,
        max_length=config['tokenizer']['encoder_max_len'],
        remove_tokens=config['inference']['remove_tokens']
    )
    val_tokenized_decoder_inputs = custom_tokenizer.batch_tokenize(
        texts=decoder_input_val,
        max_length=config['tokenizer']['decoder_max_len'],
        remove_tokens=config['inference']['remove_tokens']
    )
    val_tokenized_decoder_ouputs = custom_tokenizer.batch_tokenize(
        texts=decoder_output_val,
        max_length=config['tokenizer']['decoder_max_len'],
        remove_tokens=config['inference']['remove_tokens']
    )

    val_inputs_dataset = DatasetForVal(
        val_tokenized_encoder_inputs, 
        val_tokenized_decoder_inputs, 
        val_tokenized_decoder_ouputs,
        len(encoder_input_val))
    print('-'*10, '검증 데이터 토큰화 및 데이터셋 생성 완료', '-'*10,)
    
    # Save to cache with val_data included
    try:
        cache_data = {
            'val_data': val_data,
            'train_dataset': train_inputs_dataset,
            'val_dataset': val_inputs_dataset
        }
        torch.save(cache_data, cache_file)
        print("-"*10, "캐시 데이터 저장 완료", "-"*10)
    except Exception as e:
        print(f"캐시 저장 중 오류 발생: {str(e)}")
        if os.path.exists(cache_file):
            os.remove(cache_file)
    
    return val_data, train_inputs_dataset, val_inputs_dataset


########################################################
# 모델 성능 평가 함수
########################################################

# 모델 성능에 대한 평가 지표를 정의합니다. 본 대회에서는 ROUGE 점수를 통해 모델의 성능을 평가합니다.
def compute_metrics(config, tokenizer, pred, val_data):
    """모델 성능 평가를 위한 메트릭 계산 함수"""
    logger = logging.getLogger(__name__)
    rouge = Rouge()

    predictions = pred.predictions
    if isinstance(predictions, tuple):
        predictions = predictions[0]  
    
    labels = pred.label_ids
    labels[labels == -100] = tokenizer.pad_token_id
    predictions[predictions == -100] = tokenizer.pad_token_id

    # Decode with skip_special_tokens=True for normal tokens but preserve specified special tokens
    decoded_preds = [tokenizer.decode(p, skip_special_tokens=True) for p in predictions]
    decoded_labels = [tokenizer.decode(l, skip_special_tokens=True) for l in labels]

    # Normalize special tokens
    special_tokens = config['tokenizer']['special_tokens']
    for token in special_tokens:
        for i in range(len(decoded_preds)):
            decoded_preds[i] = decoded_preds[i].replace(token.lower(), token)
            decoded_preds[i] = decoded_preds[i].replace(token.upper(), token)
        for i in range(len(decoded_labels)):
            decoded_labels[i] = decoded_labels[i].replace(token.lower(), token)
            decoded_labels[i] = decoded_labels[i].replace(token.upper(), token)

    # Remove unwanted tokens
    remove_tokens = config['inference']['remove_tokens']
    for token in remove_tokens:
        decoded_preds = [p.replace(token, " ") for p in decoded_preds]
        decoded_labels = [l.replace(token, " ") for l in decoded_labels]

    # Clean up spacing
    decoded_preds = [" ".join(p.split()) for p in decoded_preds]
    decoded_labels = [" ".join(l.split()) for l in decoded_labels]

    # Print sample comparisons with dialogue
    print('\n' + '='*50 + ' SAMPLE PREDICTIONS ' + '='*50)
    for i in range(min(3, len(decoded_preds))):
        print(f"\n--- Sample {i+1} ---")
        print(f"DIALOGUE  : {val_data['dialogue'].iloc[i][:200]}...")  # Show first 200 chars of dialogue
        print(f"PRED: {decoded_preds[i].strip()}")
        print(f"GOLD: {decoded_labels[i].strip()}")
        print('-' * 100)
    print('='*120 + '\n')
    
    # Calculate ROUGE scores
    results = rouge.get_scores(decoded_preds, decoded_labels, avg=True)
    combined_score = ((results['rouge-1']['f'] + results['rouge-2']['f'] + results['rouge-l']['f']) / 3) * 100

    result = {
        'combined_score': combined_score,
        'rouge-1': results['rouge-1']['f'],
        'rouge-2': results['rouge-2']['f'],
        'rouge-l': results['rouge-l']['f'],
        'rouge-1-precision': results['rouge-1']['p'],
        'rouge-1-recall': results['rouge-1']['r'],
        'rouge-2-precision': results['rouge-2']['p'],
        'rouge-2-recall': results['rouge-2']['r'],
        'rouge-l-precision': results['rouge-l']['p'],
        'rouge-l-recall': results['rouge-l']['r']
        # METEOR, BLEU, exact match, semantic similarity metrics 추가 고려
    }

    # Log detailed metrics
    logger.info("\nEvaluation Metrics:")
    logger.info(f"Combined Score: {combined_score:.2f}")
    logger.info(f"ROUGE-1 F1: {results['rouge-1']['f']:.4f}")
    logger.info(f"ROUGE-2 F1: {results['rouge-2']['f']:.4f}")
    logger.info(f"ROUGE-L F1: {results['rouge-l']['f']:.4f}\n")

    return result
    


# 학습을 위한 trainer 클래스와 매개변수를 정의합니다.
def load_trainer_for_train(config, generate_model, tokenizer, train_inputs_dataset, val_inputs_dataset, val_data, callbacks=None):
    print('-'*10, '학습 파라미터 설정', '-'*10,)
    training_args = Seq2SeqTrainingArguments(
        # 기본 설정
                output_dir=config['general']['output_dir'],                                     
                overwrite_output_dir=config['training']['overwrite_output_dir'],
                seed=config['training']['seed'],
                num_train_epochs=config['training']['num_train_epochs'],                        
                per_device_train_batch_size=config['training']['per_device_train_batch_size'],  
                per_device_eval_batch_size=config['training']['per_device_eval_batch_size'],    
                
        # 학습률, 스케줄러 관련 설정
                learning_rate=config['training']['learning_rate'],                              
                warmup_ratio=config['training']['warmup_ratio'],                                
                weight_decay=config['training']['weight_decay'],                                
                lr_scheduler_type=config['training']['lr_scheduler_type'],
                optim =config['training']['optim'],
                gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],  # gradient accumulation steps는 모델 학습 시 사용되는 파라미터
                
        # 평가 및 모델 저장 관련 설정
                evaluation_strategy=config['training']['evaluation_strategy'],          
                save_strategy=config['training']['save_strategy'],
                save_total_limit=config['training']['save_total_limit'],                
                greater_is_better=config['training']['greater_is_better'],                    
                load_best_model_at_end=config['training']['load_best_model_at_end'],    
                fp16=config['training']['fp16'],    
                
        # 로깅 관련 설정
                logging_dir=config['training']['logging_dir'],              
                logging_strategy=config['training']['logging_strategy'],
                logging_steps=config['training']['logging_steps'],
                report_to=config['training']['report_to'],
                
        # 예측 관련 설정
                predict_with_generate=config['training']['predict_with_generate'],      
                metric_for_best_model=config['training']['metric_for_best_model'],          
                generation_max_length=config['training']['generation_max_length'],

                
        # 기타 설정
                do_train=config['training']['do_train'],
                do_eval=config['training']['do_eval'],
                label_smoothing_factor=config['training']['label_smoothing_factor'],
                max_grad_norm=config['training']['max_grad_norm']
            )

    wandb.init(
        entity=config['wandb']['entity'],
        project=config['wandb']['project'],
        name=config['wandb']['name'],
        settings=wandb.Settings(start_method="thread"),
        config=config  
    )
    
    # Evaluation을 위한 커스텀 trainer 클래스 정의
    class CustomSeq2SeqTrainer(Seq2SeqTrainer):
        def __init__(self, config, *args, **kwargs):
            self.config = config
            super().__init__(*args, **kwargs)
        
        def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
            if prediction_loss_only:
                return super().prediction_step(
                    model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys)

            has_labels = "labels" in inputs 
            inputs = self._prepare_inputs(inputs)

            gen_kwargs = {
                "max_length": self.config['evaluation']['generation_params']['max_length'],
                "min_length": self.config['evaluation']['generation_params']['min_length'],
                "num_beams": self.config['evaluation']['generation_params']['num_beams'],
                "no_repeat_ngram_size": self.config['evaluation']['generation_params']['no_repeat_ngram_size'],
                "length_penalty": self.config['evaluation']['generation_params']['length_penalty'],
                "repetition_penalty": self.config['evaluation']['generation_params']['repetition_penalty'],
                "pad_token_id": model.config.pad_token_id,
                "eos_token_id": model.config.eos_token_id,
                "bos_token_id": model.config.bos_token_id,
                "do_sample": False,  
                "early_stopping": True,  
            }

            with torch.no_grad():
                generated_tokens = model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    **gen_kwargs
                )

            if has_labels:
                labels = inputs["labels"].to(model.device, dtype=torch.long)
            else:
                labels = None

            return (None, generated_tokens, labels)



    trainer = CustomSeq2SeqTrainer(
        config=config,
        model=generate_model,
        args=training_args,
        train_dataset=train_inputs_dataset,
        eval_dataset=val_inputs_dataset,
        tokenizer=tokenizer,
        compute_metrics=lambda pred: compute_metrics(config, tokenizer, pred, val_data),  # Pass val_data
        callbacks=callbacks
    )
    print('-'*10, '학습 파라미터 설정 완료', '-'*10,)

    return trainer


def load_tokenizer_and_model_for_train(config, device):
    print('-'*10, '학습용 tokenizer와 사전 학습된 모델 불러오기', '-'*10,)
    model_name = config['general']['model_name']
    
    try:
        # 1. 먼저 토크나이저 초기화
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        original_vocab_size = len(tokenizer)
        # 2. 특수 토큰 추가
        special_tokens_dict = {
            'additional_special_tokens': config['tokenizer']['special_tokens']
        }
        num_added_tokens = tokenizer.add_special_tokens(special_tokens_dict)

        # 3. CUDA 캐시 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 4. 모델 로드 (ignore_mismatched_sizes 없이)
        generate_model = BartForConditionalGeneration.from_pretrained(model_name)
        
        # 5. 임베딩 레이어 크기 조정
        if num_added_tokens > 0:
            generate_model.resize_token_embeddings(len(tokenizer))

        # 6. GPU 메모리 설정
        if torch.cuda.is_available():
            # GPU 메모리 분할 사용 비활성화
            torch.cuda.set_device(device)
            # 메모리 할당 최적화
            torch.backends.cudnn.benchmark = True
        # 6. 디바이스 이동
        generate_model = generate_model.to(device)
        print(generate_model.config)
        print(f"Original vocab size: {original_vocab_size}")
        print(f"New vocab size: {len(tokenizer)}")
        print(f"Added {num_added_tokens} special tokens")
        print(f"Model moved to device: {device}")

        print('-'*10, '학습용 tokenizer와 사전 학습된 모델 불러오기 완료', '-'*10,)
        return generate_model, tokenizer
    except Exception as e:
        print(f"모델 로드 중 오류 발생: {str(e)}")
        # CUDA 캐시 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise



########################################################
# 학습 및 평가 함수
########################################################

def main(config, param_combination=None):
    try:
        experiment_metrics = {
            'epochs': [],
            'best_metrics': None,
            'final_metrics': None,
            'parameters': param_combination,
            'best_epoch': None
        }

        if param_combination:
            if 'training' in param_combination:
                for key, value in param_combination['training'].items():
                    config['training'][key] = value
            if 'evaluation' in param_combination and 'generation_params' in param_combination['evaluation']:
                for key, value in param_combination['evaluation']['generation_params'].items():
                    config['evaluation']['generation_params'][key] = value

        device = torch.device('cuda:0' if torch.cuda.is_available()  else 'cpu')
        print('-'*10, f'device : {device}', '-'*10,)
        print(torch.__version__)
        
        generate_model, tokenizer = load_tokenizer_and_model_for_train(config, device)

        # 학습 중 메트릭 로깅을 위한 커스텀 콜백 정의
        class EnhancedMetricsCallback(TrainerCallback):
            """Enhanced callback for tracking training metrics"""
            def __init__(self, experiment_metrics, patience=3):
                self.experiment_metrics = experiment_metrics
                self.patience = int(patience)  # Ensure integer type
                self.best_rouge = 0.0  # Ensure float type
                self.no_improve_count = 0
                self.history = []
                
            def on_evaluate(self, args, state, control, metrics, **kwargs):
                try:
                    current_rouge = float(metrics.get('eval_combined_score', 0.0))
                    
                    self.history.append({
                        'epoch': float(state.epoch),
                        'rouge': float(current_rouge),
                        'loss': float(metrics.get('eval_loss', 0.0))
                    })
                    
                    logger.info(f"\nEpoch {int(state.epoch)} Evaluation:")
                    logger.info(f"Combined ROUGE: {current_rouge:.4f}")
                    logger.info(f"ROUGE-1: {float(metrics.get('eval_rouge-1', 0.0)):.4f}")
                    logger.info(f"ROUGE-2: {float(metrics.get('eval_rouge-2', 0.0)):.4f}")
                    logger.info(f"ROUGE-L: {float(metrics.get('eval_rouge-l', 0.0)):.4f}")
                    
                    if current_rouge > self.best_rouge:
                        self.best_rouge = current_rouge
                        self.no_improve_count = 0
                        logger.info(f"New best ROUGE score: {current_rouge:.4f}")
                    else:
                        self.no_improve_count += 1
                        logger.info(f"No improvement for {self.no_improve_count} evaluations")
                        
                    if self.no_improve_count >= self.patience:
                        logger.info("Early stopping triggered")
                        control.should_training_stop = True
                        
                except Exception as e:
                    logger.error(f"Error in metrics callback: {str(e)}")
                    raise

        # 데이터셋 및 트레이너 준비
        preprocessor = Preprocess(
            config['tokenizer']['bos_token'], 
            config['tokenizer']['eos_token'],
            config 
        )
        
        # Explicitly unpack all three values
        val_data, train_inputs_dataset, val_inputs_dataset = prepare_train_dataset(config, preprocessor, tokenizer)
        
        # Verify val_data exists
        if val_data is None:
            raise ValueError("Validation data is missing")
            
        trainer = load_trainer_for_train(
            config, 
            generate_model,
            tokenizer,
            train_inputs_dataset,
            val_inputs_dataset,
            val_data,  # Pass validation data
            callbacks=[EnhancedMetricsCallback(experiment_metrics)]
        )
        
        # Train and evaluate
        train_result = trainer.train()
        eval_metrics = trainer.evaluate()
        
        # Store final metrics
        experiment_metrics['final_metrics'] = {
            'train_metrics': train_result.metrics,
            'eval_metrics': eval_metrics
        }
        
        # 실험 결과 저장
        experiment_dir = os.path.join(DATA_DIR, 'experiments')
        os.makedirs(experiment_dir, exist_ok=True)
        
        experiment_file = os.path.join(experiment_dir, f'experiment_{kr_time}.json')
        with open(experiment_file, 'w') as f:
            json.dump(experiment_metrics, f, indent=4)
        print(f"실험 결과 저장 완료 : {experiment_file}")
        wandb.log(experiment_metrics)
        return experiment_metrics
    
    except Exception as e:
        print(f"Training failed: {str(e)}")
        raise
    finally:
        clean_resources()

def run_parameter_optimization(config):
    experiment_dir = os.path.join(DATA_DIR, 'parameter_optimization')
    os.makedirs(experiment_dir, exist_ok=True)
    
    try:
        param_combinations = generate_parameter_combinations()
        logger.info(f"Generated {len(param_combinations)} parameter combinations")
        
        all_experiments = []
        best_experiment = None
        
        for i, params in enumerate(param_combinations):
            logger.info(f"\n실험 번호 {i+1}/{len(param_combinations)}")
            logger.info(f"파라미터: {json.dumps(params, indent=2)}")
            
            try:
                experiment_metrics = main(config.copy(), params)
                
                if experiment_metrics:
                    # 최적 실험 업데이트
                    if (best_experiment is None or 
                        experiment_metrics['best_metrics']['eval_metrics']['eval_combined_score'] > 
                        best_experiment['best_metrics']['eval_metrics']['eval_combined_score']):
                        best_experiment = experiment_metrics

                    all_experiments.append(experiment_metrics)
                    
                    # 각 실험 결과 저장
                    save_optimization_results(experiment_dir, all_experiments, best_experiment)
                    
            except Exception as e:
                logger.error(f"실험 {i+1} 오류: {str(e)}")
                continue
        
        return all_experiments, best_experiment
        
    except Exception as e:
        logger.error(f"파라미터 최적화 실패: {str(e)}")
        raise

def validate_parameter_types(params):
    pass

def save_optimization_results(experiment_dir, all_experiments, best_experiment):
    """Save optimization results to file"""
    results_file = os.path.join(experiment_dir, f'optimization_results.json')
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump({
            'all_experiments': all_experiments,
            'best_experiment': best_experiment
        }, f, indent=4, ensure_ascii=False)

def generate_parameter_combinations():    # 학습 파라미터 - explicit float/int types
    train_param_grid = {
        "learning_rate": [float(1e-6)],
        "gradient_accumulation_steps": [int(1), int(2)]
    }

    # 평가 파라미터
    eval_param_grid = {
        "num_beams": [int(4), int(6)],
        "no_repeat_ngram_size": [int(2), int(3)],
        "length_penalty": [float(1.0), float(1.2)]
    }
    
    # 모든 가능한 조합 가져오기
    train_keys = train_param_grid.keys()
    train_values = train_param_grid.values()
    eval_keys = eval_param_grid.keys()
    eval_values = eval_param_grid.values()

    train_combinations = list(product(*train_values))
    eval_combinations = list(product(*eval_values))
    
    # 모든 가능한 조합 생성
    param_combinations = []
    for train_combo in train_combinations:
        train_dict = dict(zip(train_keys, train_combo))
        for eval_combo in eval_combinations:
            eval_dict = dict(zip(eval_keys, eval_combo))
            param_combinations.append({
                "training": train_dict,
                "evaluation": {"generation_params": eval_dict}
            })
    
    return param_combinations


if __name__ == "__main__":
    try:
        add_safe_globals([DatasetForTrain, DatasetForVal])
        run_parameter_optimization(loaded_config)
    except KeyboardInterrupt:
        print("\n사용자에 의해 학습 중단")
        clean_resources()
    except Exception as e:
        print(f"\n학습 실패: {str(e)}")
        clean_resources()
        raise

    

# if __name__ == "__main__":
#     main(loaded_config)






