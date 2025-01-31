import pandas as pd
import os
import re
import json
import yaml
import gc 
import logging
import random
import numpy as np
from glob import glob
from tqdm import tqdm
import pytz
from datetime import datetime
from pprint import pprint
from itertools import product
import copy

import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset , DataLoader

from transformers import (AutoTokenizer, 
                          BartForConditionalGeneration, 
                          BartConfig, 
                          Seq2SeqTrainingArguments, 
                          Seq2SeqTrainer,
                          GenerationConfig)
from transformers import EarlyStoppingCallback

from rouge import Rouge # 모델의 성능을 평가하기 위한 라이브러리입니다.

import wandb
import optuna
import functools


########################################################
# 기본 설정
########################################################
DEVICE = torch.device('cuda:0' if torch.cuda.is_available()  else 'cpu')

MODEL_NAME = "digit82/kobart-summarization"
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)

DATA_DIR = "/data/ephemeral/home/baseline/data"
OUTPUT_DIR = "/data/ephemeral/home/NLP_JES/output"
CONFIG_DIR = os.path.join(OUTPUT_DIR, 'config')
LOGS_DIR = os.path.join(OUTPUT_DIR, 'logs')
CACHE_DIR = os.path.join(OUTPUT_DIR, 'cache')
ERROR_LOG_FILE = os.path.join(LOGS_DIR, 'oom_error_log.csv')

KOREA_TIME = datetime.now(pytz.timezone('Asia/Seoul'))
KR_TIME = KOREA_TIME.strftime("%m%d_%H%M")

########################################################
# 유틸 함수
########################################################
def fix_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # CUDA 연산에도 시드를 고정
    torch.backends.cudnn.deterministic = True  # deterministic 연산 모드로 설정
    torch.backends.cudnn.benchmark = False  # 벤치마크 모드 비활성화
    print(f"Random Seed Fixed: {seed}")

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(OUTPUT_DIR, f"log.txt")),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)



def clear_gpu_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        print("CUDA 캐시 비우기 완료")

def delete_model(model):
    del model
    clear_gpu_cache()
    gc.collect()  # Python의 GC가 메모리 해제를 도울 수 있음
    print("모델 삭제 및 메모리 해제 완료")

########################################################
# Wandb 관련 설정
########################################################

my_name = "JES"
team_name = "byte_busters"
team_project = "NLP-JES"
WANDB_KEY = "0021804e1b7f9c1a58208b3845d47114e005a377"
RUN_NAME = f"JES_{MODEL_NAME}_{KR_TIME}"

########################################################
# config 설정
########################################################

CONFIG_DATA = {
    "general": {
        "data_path": DATA_DIR,          # 데이터 경로 지정
        "output_dir": OUTPUT_DIR,        # 모델 최종 출력 값 저장 경로 지정
        "model_name": MODEL_NAME,                                   # 모델 이름 지정
        "run_time": KR_TIME                                         # 실행 시간 지정
    },
    "inference": {
        "batch_size" : 32,                                              # 평가 시 사용할 배치 크기 지정
        "ckt_path": "/data/ephemeral/home/NLP_JES/output/best_model",   # 모델의 checkpoint와 best model 저장 경로 설정
        "early_stopping": True,                                         # 조기 중단 여부 지정
        "generate_max_length": 100,                                     # 생성 최대 길이 지정
        "no_repeat_ngram_size": 2,                                      # 중복 생성 방지 위한 n-gram 크기 설정
        "num_beams": 4,                                                 # 빔 서치에서 생성할 빔 수 지정
        "remove_tokens": ['<usr>', f"{TOKENIZER.bos_token}", f"{TOKENIZER.eos_token}", f"{TOKENIZER.pad_token}"],
        "result_path": "./prediction/"                                 # 예측 결과 저장 경로 설정
    },
    "tokenizer": {
        "bos_token": f"{TOKENIZER.bos_token}",                      # 시작 토큰 지정
        "decoder_max_len": 100,                                     # 디코더 최대 길이 지정
        "encoder_max_len": 512,                                     # 인코더 최대 길이 지정. 초과 토큰은 잘라내거나 패딩 처리
        "eos_token": f"{TOKENIZER.eos_token}",                      # 종료 토큰 지정
        "special_tokens": [
            '#Address#', '#CarNumber#', '#CardNumber#', '#DateOfBirth#', 
            '#Email#', '#PassportNumber#', '#PhoneNumber#', '#SSN#',
            '#Person1#', '#Person2#', '#Person3#', '#Person4#', '#Person5#', '#Person6#', '#Person7#']
    },
    "training": {
        # 학습 처리 관련 설정
        "overwrite_output_dir": True,                                # 출력 디렉토리 덮어쓰기 여부 지정     
        "seed": 42,                                                  # 랜덤 시드 지정
        "num_train_epochs": 10,                                      # 학습 에폭 수 지정
        "per_device_train_batch_size": 50,                           # 학습 배치 크기 지정
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
        "max_grad_norm": 1.0                                        # 그래디언트 클리핑 지정

    },
    "evaluation": {
        "generation_params": {
            "num_beams": 4,
            "no_repeat_ngram_size": 2,
            "length_penalty": 1.0,
            "repetition_penalty": 1.0,
            "max_length": 100,                  # 고정
            "min_length": 30,                   # 고정
            "early_stopping": True,
            "temperature": 0.7,
            "top_p": 0.9,
            "do_sample": True
        }
    },
    # (선택) wandb 홈페이지에 가입하여 얻은 정보를 기반으로 작성합니다.
    "wandb": {
        "entity": team_name,                        # wandb 엔티티 이름 지정
        "project": team_project,                            # wandb 프로젝트 이름 지정
        "name": f"JES_{MODEL_NAME}_{KR_TIME}"            # wandb run 이름 지정
    }
}

# config 저장 및 불러오기 
CONFIG_PATH = os.path.join(CONFIG_DIR, f"config.yaml")
with open(CONFIG_PATH, "w") as file:  # w: write, r: read, a: append
    yaml.dump(CONFIG_DATA, file, allow_unicode=True)


########################################################
# 전처리 함수 정의
########################################################
def preprocess_text(text):
    text = ' '.join(text.split())
    special_token_pattern = r'#(Person[1-7]|PhoneNumber|Address|PassportNumber|CardNumber|CarNumber|DateOfBirth|Email|SSN)#'
    
    temp_tokens = {f"SPECIAL_TOKEN_{i}": match.group() for i, match in enumerate(re.finditer(special_token_pattern, text))}
    for marker, token in temp_tokens.items():
        text = text.replace(token, marker)
    text = re.sub(r"[!?]+$", ".", text) if not text.endswith(".") else text
    
    sentences = text.split('.')
    sentences = list(dict.fromkeys(sentences))  # 중복 제거
    text = '. '.join(s.strip() for s in sentences if s.strip()) + '.'
    # 임시 마커를 다시 특수 토큰으로 복원
    for marker, token in temp_tokens.items():
        text = text.replace(marker, token)
    
    return text.strip()

########################################################
# 데이터 준비 클래스 정의
########################################################
class Preprocess:
    def __init__(self, bos_token: str, eos_token: str, config: dict) -> None:
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.config = config  # Store the config
        self.special_token_pattern = r'#(Person[1-7]|PhoneNumber|Address|PassportNumber|CardNumber|CarNumber|DateOfBirth|Email|SSN)#'
    
    @staticmethod
    def make_set_as_df(file_path, is_train=True):
        if is_train:
            df = pd.read_csv(file_path)
            train_df = df[['fname', 'dialogue', 'summary']]
            return train_df
        else: 
            df = pd.read_csv(file_path)
            test_df = df[['fname', 'dialogue']]
            return test_df
    
    def validate_special_tokens(self, text: str) -> str:
        special_tokens = re.findall(self.special_token_pattern, text)
        for token in special_tokens:
            token_full = f'#{token}#'
            if token_full not in text:
                logger.warning(f"잘못된 특수 토큰 형식: {token}")
                text = text.replace(token, token_full)
        return text

    def make_input(self, dataset, is_test=False):
        if is_test:
            encoder_input = dataset['dialogue'].apply(self.validate_special_tokens)
            decoder_input = [self.bos_token] * len(dataset['dialogue'])
            return encoder_input.tolist(), list(decoder_input)
        else:
            encoder_input = dataset['dialogue'].apply(self.validate_special_tokens)
            processed_summary = dataset['summary'].apply(preprocess_text)
            decoder_input = processed_summary.apply(lambda x: self.bos_token + str(x))
            decoder_output = processed_summary.apply(lambda x: str(x) + self.eos_token)
            
            # # 특수 토큰 검증
            # decoder_input = decoder_input.apply(self.validate_special_tokens)
            # decoder_output = decoder_output.apply(self.validate_special_tokens)
            
            return encoder_input.tolist(), decoder_input.tolist(), decoder_output.tolist()
 
########################################################
# 데이터 준비 클래스 정의
########################################################
class DatasetForTrain(Dataset):
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

def prepare_train_dataset(config, preprocessor, data_path, tokenizer):
    train_file_path = os.path.join(DATA_DIR,'train.csv')
    val_file_path = os.path.join(DATA_DIR,'dev.csv')

    # train, validation에 대해 각각 데이터프레임을 구축합니다.
    train_data = preprocessor.make_set_as_df(train_file_path)
    val_data = preprocessor.make_set_as_df(val_file_path)

    # 특수 토큰 검증
    def validate_and_fix_special_tokens(text):
        if isinstance(text, str):
            # 특수 토큰 패턴 정의
            special_token_pattern = r'#(Person[1-7]|PhoneNumber|Address|PassportNumber|CardNumber|CarNumber|DateOfBirth|Email|SSN)#'
            
            # 특수 토큰이 있는지 확인
            special_tokens = re.findall(special_token_pattern, text)
            if not special_tokens:
                # logger.warning(f"특수 토큰이 없습니다. 텍스트: {text}")
                return text
            # 각 토큰이 올바른 형식인지 확인
            for token in special_tokens:
                token_full = f'#{token}#'
                if token_full not in text:
                    logger.warning(f"잘못된 특수 토큰 형식: {token_full}, 텍스트: {text}")
                    text = text.replace(token, token_full)
            
            return text
        return ""

    # 대화 데이터의 특수 토큰 검증
    train_data['dialogue'] = train_data['dialogue'].apply(validate_and_fix_special_tokens)
    val_data['dialogue'] = val_data['dialogue'].apply(validate_and_fix_special_tokens)

    encoder_input_train, decoder_input_train, decoder_output_train = preprocessor.make_input(train_data)
    encoder_input_val, decoder_input_val, decoder_output_val = preprocessor.make_input(val_data)
    
    # 토큰화 전에 특수 토큰 검증
    encoder_input_train = [validate_and_fix_special_tokens(text) for text in encoder_input_train]
    decoder_input_train = [validate_and_fix_special_tokens(text) for text in decoder_input_train]
    decoder_output_train = [validate_and_fix_special_tokens(text) for text in decoder_output_train]
    
    print('-'*10, '데이터 로드 완료', '-'*10,)

    # Add special token handling for tokenization
    tokenized_encoder_inputs = tokenizer(
        encoder_input_train, 
        return_tensors="pt", 
        padding=True,
        add_special_tokens=True, 
        truncation=True, 
        max_length=config['tokenizer']['encoder_max_len'], 
        return_token_type_ids=False)

    # Ensure special tokens are preserved in decoder inputs
    tokenized_decoder_inputs = tokenizer(
        decoder_input_train,
        return_tensors="pt", 
        padding=True,
        add_special_tokens=True, 
        truncation=True,
        max_length=config['tokenizer']['decoder_max_len'],
        return_token_type_ids=False)

    # Ensure special tokens are preserved in decoder outputs
    tokenized_decoder_ouputs = tokenizer(
        decoder_output_train,
        return_tensors="pt", 
        padding=True,
        add_special_tokens=True, 
        truncation=True,
        max_length=config['tokenizer']['decoder_max_len'],
        return_token_type_ids=False)

    train_inputs_dataset = DatasetForTrain(
        tokenized_encoder_inputs, 
        tokenized_decoder_inputs, 
        tokenized_decoder_ouputs,
        len(encoder_input_train))
    
    print('-'*10, '학습 데이터 데이터셋 준비 완료', '-'*10,)

    val_tokenized_encoder_inputs = tokenizer(
        encoder_input_val, 
        return_tensors="pt", 
        padding=True,
        add_special_tokens=True, 
        truncation=True, 
        max_length=config['tokenizer']['encoder_max_len'], 
        return_token_type_ids=False)
    val_tokenized_decoder_inputs = tokenizer(
        decoder_input_val, 
        return_tensors="pt", 
        padding=True,
        add_special_tokens=True, 
        truncation=True, 
        max_length=config['tokenizer']['decoder_max_len'], 
        return_token_type_ids=False)
    val_tokenized_decoder_ouputs = tokenizer(
        decoder_output_val, 
        return_tensors="pt", 
        padding=True,
        add_special_tokens=True, 
        truncation=True, 
        max_length=config['tokenizer']['decoder_max_len'], 
        return_token_type_ids=False)
    val_inputs_dataset = DatasetForVal(
        val_tokenized_encoder_inputs, 
        val_tokenized_decoder_inputs, 
        val_tokenized_decoder_ouputs,
        len(encoder_input_val))

    print('-'*10, '검증 데이터 데이터셋 준비 완료', '-'*10,)
    return val_data, train_inputs_dataset, val_inputs_dataset

def prepare_test_dataset(config,preprocessor, tokenizer):

    test_file_path = os.path.join(DATA_DIR,'test.csv')
    test_data = preprocessor.make_set_as_df(test_file_path,is_train=False)
    test_id = test_data['fname']

    print('-'*150)
    print(f'테스트 데이터:\n{test_data["dialogue"][0]}')
    print('-'*150)

    encoder_input_test , decoder_input_test = preprocessor.make_input(test_data,is_test=True)
    print('-'*10, '데이터 로드 완료', '-'*10,)

    test_tokenized_encoder_inputs = tokenizer(encoder_input_test, return_tensors="pt", padding=True,
                    add_special_tokens=True, truncation=True, max_length=config['tokenizer']['encoder_max_len'], return_token_type_ids=False,)
    test_tokenized_decoder_inputs = tokenizer(decoder_input_test, return_tensors="pt", padding=True,
                    add_special_tokens=True, truncation=True, max_length=config['tokenizer']['decoder_max_len'], return_token_type_ids=False,)

    test_encoder_inputs_dataset = DatasetForInference(test_tokenized_encoder_inputs, test_id, len(encoder_input_test))
    print('-'*10, '테스트 데이터셋 준비 완료', '-'*10,)

    return test_data, test_encoder_inputs_dataset


########################################################
# 평가 함수 정의
########################################################
def check_grammar(text):
    """문법 점수를 계산합니다."""
    score = 1.0
    if not text.strip().endswith('.'):
        score -= 0.1
    if not re.search(r'#Person\d+#', text):
        score -= 0.1
    return max(0.0, score)

def check_entity_coverage(pred, gold):
    """엔티티(화자, 장소 등) 커버리지를 계산합니다."""
    entity_pattern = r'#[A-Za-z]+\d*#'
    gold_entities = set(re.findall(entity_pattern, gold))
    pred_entities = set(re.findall(entity_pattern, pred))
    if not gold_entities:
        return 1.0
    coverage = len(pred_entities.intersection(gold_entities)) / len(gold_entities)
    return coverage

def check_completeness(text):
    """요약문의 완성도를 계산합니다."""
    min_words = 10
    words = text.split()
    score = 1.0
    if len(words) < min_words:
        score *= (len(words) / min_words)
    if not any(word.endswith('다.') for word in words):
        score -= 0.2
    return max(0.0, score)

def compute_metrics(config, tokenizer, pred, val_data):
    rouge = Rouge()
    predictions = pred.predictions
    labels = pred.label_ids

    # Debug tokenization
    print("\nDEBUG: Token IDs before decoding")
    print(f"Predictions shape: {predictions.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Sample prediction IDs: {predictions[0][:50]}")
    print(f"Sample label IDs: {labels[0][:50]}")

    predictions[predictions == -100] = tokenizer.pad_token_id
    labels[labels == -100] = tokenizer.pad_token_id

    # Explicitly keep special tokens during decoding
    decoded_preds = tokenizer.batch_decode(predictions, 
                                         skip_special_tokens=False,  # Keep special tokens
                                         clean_up_tokenization_spaces=True)
    decoded_labels = tokenizer.batch_decode(labels, 
                                          skip_special_tokens=False,  # Keep special tokens
                                          clean_up_tokenization_spaces=True)

    # Debug decoded outputs
    print("\nDEBUG: Decoded outputs")
    print(f"Sample raw prediction: {decoded_preds[0]}")
    print(f"Sample raw label: {decoded_labels[0]}")

    # Check special token presence
    special_tokens = [
        '#Person1#', '#Person2#', '#Person3#', '#Person4#', '#Person5#', '#Person6#', '#Person7#',
        '#PhoneNumber#', '#Address#', '#PassportNumber#', '#SSN#', '#CardNumber#', '#CarNumber#',
        '#DateOfBirth#', '#Email#'
    ]
    
    # Debug special token IDs
    print("\nDEBUG: Special Token IDs")
    for token in special_tokens:
        token_id = tokenizer.convert_tokens_to_ids(token)
        print(f"{token}: {token_id}")
        
    # Only remove non-special tokens
    remove_tokens = [token for token in config['inference']['remove_tokens'] 
                    if token not in special_tokens]
    
    replaced_predictions = decoded_preds.copy()
    replaced_labels = labels.copy()
    
    # remove_tokens = [token for token in config['inference']['remove_tokens'] 
    #                 if not token.startswith('#Person#') and 
    #                 not token in ['#PhoneNumber#', '#Address#', '#PassportNumber#', '#SSN#', '#CardNumber#', '#CarNumber#', '#DateOfBirth#', '#Email#']]
    replaced_labels = decoded_labels.copy()


    for token in remove_tokens:
        replaced_predictions = [sentence.replace(token," ") for sentence in replaced_predictions]
        replaced_labels = [sentence.replace(token," ") for sentence in replaced_labels]

    # Debug final outputs
    print("\nDEBUG: Final processed outputs")
    print(f"Sample processed prediction: {replaced_predictions[0]}")
    print(f"Sample processed label: {replaced_labels[0]}")

    # Add detailed special token analysis
    for idx in range(min(5, len(replaced_predictions))):
        print(f"\n=== Example {idx+1} ===")
        print(f"Original Input: {val_data['dialogue'][idx]}")
        print(f"PRED: {replaced_predictions[idx]}")
        print(f"GOLD: {replaced_labels[idx]}")
        
        
        # Count special tokens
        pred_tokens = {token: replaced_predictions[idx].count(token) for token in special_tokens}
        gold_tokens = {token: replaced_labels[idx].count(token) for token in special_tokens}
        
        print("\nSpecial Token Count:")
        print("Prediction:", {k: v for k, v in pred_tokens.items() if v > 0})
        print("Gold Label:", {k: v for k, v in gold_tokens.items() if v > 0})
                # Debug special tokens
        # pred_special = re.findall(r'#[A-Za-z]+\d*#', replaced_predictions[idx])
        # gold_special = re.findall(r'#[A-Za-z]+\d*#', replaced_labels[idx])
        # print(f"Special tokens in prediction: {pred_special}")
        # print(f"Special tokens in gold: {gold_special}")
        # print('-'*150)

    # 최종적인 ROUGE 점수를 계산합니다.
    results = rouge.get_scores(replaced_predictions, replaced_labels, avg=True)
    combined_score = ((results['rouge-1']['f'] + results['rouge-2']['f'] + results['rouge-l']['f']) / 3) * 100
    
    grammar_score = []
    entity_coverage_score = []
    completeness_score = []

    for pred, gold in zip(replaced_predictions, replaced_labels):
        grammar_score.append(check_grammar(pred))
        entity_coverage_score.append(check_entity_coverage(pred, gold))
        completeness_score.append(check_completeness(pred))

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
        'rouge-l-recall': results['rouge-l']['r'],
        # 'grammar_score': grammar_score,
        'entity_coverage_score': entity_coverage_score,
        # 'completeness_score': completeness_score
    }
    print("\n평가 지표:")
    print(f"합계 점수: {combined_score:.2f}")
    print(f"ROUGE-1 F1: {results['rouge-1']['f']:.4f}")
    print(f"ROUGE-2 F1: {results['rouge-2']['f']:.4f}")
    print(f"ROUGE-L F1: {results['rouge-l']['f']:.4f}\n")
    # print(f"문법 점수: {grammar_score}")
    # print(f"엔티티 커버리지 점수: {entity_coverage_score}")
    # print(f"완성도 점수: {completeness_score}\n")
    return result


########################################################
# 토크나이저 및 모델 로드
########################################################

def load_tokenizer_and_model_for_train(config,device):
    print('-'*10, '토크나이저 및 모델 로드', '-'*10,)
    print('-'*10, f'모델 이름 : {config["general"]["model_name"]}', '-'*10,)
    model_name = config['general']['model_name']
    bart_config = BartConfig().from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    generate_model = BartForConditionalGeneration.from_pretrained(config['general']['model_name'],config=bart_config)

    special_tokens_dict={'additional_special_tokens':[
        '#Person1#', '#Person2#', '#Person3#', '#Person4#', '#Person5#', '#Person6#', '#Person7#', 
        '#PhoneNumber#', '#Address#', '#PassportNumber#', '#SSN#', '#CardNumber#', '#CarNumber#', '#DateOfBirth#', '#Email#'
    ]}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    print(f"Added {num_added_toks} special tokens")
    
    generate_model.resize_token_embeddings(len(tokenizer)) # 사전에 special token을 추가했으므로 재구성 해줍니다.
    print("토크나이저 특수 토큰:")
    print(tokenizer.special_tokens_map)
    print("\n추가된 특수 토큰:")
    print(tokenizer.additional_special_tokens)
    generate_model.to(device)
    print(generate_model.config)

    

    print('-'*10, '토크나이저 및 모델 로드 완료', '-'*10,)
    return generate_model , tokenizer

def load_trainer_for_train(config,generate_model,tokenizer,train_inputs_dataset,val_inputs_dataset,val_data):
    print('-'*10, '학습 관련 설정', '-'*10,)
    training_args = Seq2SeqTrainingArguments(
        output_dir=config['general']['output_dir'], # model output directory
        overwrite_output_dir=config['training']['overwrite_output_dir'],
        seed=config['training']['seed'],
        num_train_epochs=config['training']['num_train_epochs'],  # total number of training epochs
        per_device_train_batch_size=config['training']['per_device_train_batch_size'], # batch size per device during training
        per_device_eval_batch_size=config['training']['per_device_eval_batch_size'],# batch size for evaluation
        
        # 학습률, 스케줄러 관련 설정
        learning_rate=config['training']['learning_rate'], # learning_rate
        warmup_ratio=config['training']['warmup_ratio'],  # number of warmup steps for learning rate scheduler
        weight_decay=config['training']['weight_decay'],  # strength of weight decay
        lr_scheduler_type=config['training']['lr_scheduler_type'],
        optim =config['training']['optim'],
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
        
        # 평가 및 모델 저장 관련 설정
        evaluation_strategy=config['training']['evaluation_strategy'], # evaluation strategy to adopt during training
        save_strategy =config['training']['save_strategy'],
        save_total_limit=config['training']['save_total_limit'], # number of total save model.
        greater_is_better=config['training']['greater_is_better'],
        load_best_model_at_end=config['training']['load_best_model_at_end'], # 최종적으로 가장 높은 점수 저장
        fp16=config['training']['fp16'],
        
        # 로깅 관련 설정
        logging_dir=config['training']['logging_dir'], # directory for storing logs
        logging_strategy=config['training']['logging_strategy'],
        logging_steps=config['training']['logging_steps'],
        report_to=config['training']['report_to'], # (선택) wandb를 사용할 때 설정합니다.
        

        # 예측 관련 설정
        predict_with_generate=config['training']['predict_with_generate'], #To use BLEU or ROUGE score
        metric_for_best_model=config['training']['metric_for_best_model'],
        generation_max_length=config['training']['generation_max_length'],
        generation_num_beams=config['training']['generation_num_beams'],
        # no_repeat_ngram_size=config['training']['no_repeat_ngram_size'],
        # length_penalty=config['training']['length_penalty'],
        # repetition_penalty=config['training']['repetition_penalty'],
        
        # 기타 설정
        do_train=config['training']['do_train'],
        do_eval=config['training']['do_eval'],
        label_smoothing_factor=config['training']['label_smoothing_factor'],
        max_grad_norm=config['training']['max_grad_norm']
        )

    # os.environ["WANDB_LOG_MODEL"]="1"
    # os.environ["WANDB_WATCH"]="1"

    print('-'*10, '학습 관련 설정 완료', '-'*10,)
    print('-'*10, '트레이너 생성', '-'*10,)


    class CustomSeq2SeqTrainer(Seq2SeqTrainer):
        def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
            if prediction_loss_only:
                return super().prediction_step(model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys)
            
            has_labels = "labels" in inputs
            inputs = self._prepare_inputs(inputs)
            # input_ids = inputs["input_ids"].to(model.device, dtype=torch.long)
            # attention_mask = inputs["attention_mask"].to(model.device, dtype=torch.long)    

            # # Create a list of allowed token IDs (including special tokens)
            # special_token_ids = []
            special_tokens = [
                '#Person1#', '#Person2#', '#Person3#', '#Person4#', '#Person5#', '#Person6#', '#Person7#',
                '#PhoneNumber#', '#Address#', '#PassportNumber#', '#SSN#', '#CardNumber#', '#CarNumber#',
                '#DateOfBirth#', '#Email#'
            ]
            special_token_ids = [tokenizer.convert_tokens_to_ids(token) for token in special_tokens]
            
            # Create forced token map for generation
            def enforce_special_tokens(batch_id, input_sequence):
                # Find special tokens in input sequence
                input_text = tokenizer.decode(inputs['input_ids'][batch_id])
                allowed_tokens = list(range(len(tokenizer)))  # Allow all tokens by default
                
                # Add special tokens that appear in input
                for token in special_tokens:
                    if token in input_text:
                        token_id = tokenizer.convert_tokens_to_ids(token)
                        if token_id not in allowed_tokens:
                            allowed_tokens.append(token_id)
                
                return allowed_tokens

            generation_kwargs = {
                'num_beams': config['evaluation']['generation_params']['num_beams'],
                'max_length': config['evaluation']['generation_params']['max_length'],
                'min_length': config['evaluation']['generation_params']['min_length'],
                # 'no_repeat_ngram_size': config['evaluation']['generation_params']['no_repeat_ngram_size'],
                'length_penalty': config['evaluation']['generation_params']['length_penalty'],
                'early_stopping': True,
                'do_sample': False,  # Disable sampling for more deterministic output
                'no_repeat_ngram_size': 0,  # Don't filter special tokens
                'bad_words_ids': None,  # Don't filter any tokens
                'forced_bos_token_id': None,
                'forced_eos_token_id': tokenizer.eos_token_id,
                'prefix_allowed_tokens_fn': enforce_special_tokens,  # Use our custom function
                'use_cache': True,
                'repetition_penalty': 1.0,  # Don't penalize special token repetition
            }

            with torch.no_grad():
                try:
                    generated_tokens = model.generate(
                        input_ids=inputs["input_ids"].to(model.device),
                        attention_mask=inputs["attention_mask"].to(model.device),
                        **generation_kwargs
                    )
                except RuntimeError as e:
                    if "OOM" in str(e):
                        print("CUDA 메모리 부족. 시도 재개...")
                        torch.cuda.empty_cache()
                        generation_kwargs.update({
                            "max_length": 64,
                            "num_beams": 1
                        })
                        generated_tokens = model.generate(
                            input_ids=inputs["input_ids"].to(model.device),
                            attention_mask=inputs["attention_mask"].to(model.device),
                            **generation_kwargs
                        )
                    else:
                        raise e

            if has_labels:
                labels = inputs["labels"].to(model.device)
            else:
                labels = None

            return (None, generated_tokens, labels)
        
    trainer = CustomSeq2SeqTrainer(
        model=generate_model,
        args=training_args,
        train_dataset=train_inputs_dataset,
        eval_dataset=val_inputs_dataset,
        tokenizer=tokenizer,
        compute_metrics=lambda pred: compute_metrics(config, tokenizer, pred, val_data),
        # callbacks = [MyCallback]
    )
    
    print('-'*10, '트레이너 생성 완료', '-'*10,)

    return trainer

def load_tokenizer_and_model_for_test(config, device):
    print('-'*10, '토크나이저 및 모델 로드', '-'*10,)
    model_name = config['general']['model_name']
    ckt_path = config['inference']['ckt_path']
    special_tokens = config['tokenizer']['special_tokens']
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model_config = BartConfig.from_pretrained(model_name)
    generate_model = BartForConditionalGeneration.from_pretrained(
        config['general']['model_name'],
        config=model_config
    )

    special_tokens_dict = {
        'additional_special_tokens': [
            '#Person1#', '#Person2#', '#Person3#', '#Person4#', '#Person5#', '#Person6#', '#Person7#',
            '#PhoneNumber#', '#Address#', '#PassportNumber#', '#SSN#', '#CardNumber#', '#CarNumber#',
            '#DateOfBirth#', '#Email#'
        ]
    }
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    print(f"Added {num_added_toks} special tokens")
    generate_model.resize_token_embeddings(len(tokenizer))

        # Verify special tokens are in vocabulary
    for token in special_tokens_dict['additional_special_tokens']:
        token_id = tokenizer.convert_tokens_to_ids(token)
        print(f"{token}: {token_id}")
        if token_id == tokenizer.unk_token_id:
            print(f"Warning: {token} not properly added to vocabulary!")
    
    sample_text = "테스트 #Person1# 와 #Person2# 가 대화를 나눴다."
    if not validate_special_tokens(tokenizer, sample_text):
        logger.warning("Special tokens validation failed in tokenizer loading!")
    
    print("토크나이저 특수 토큰:")
    print(tokenizer.special_tokens_map)
    print("\n추가된 특수 토큰:")
    print(tokenizer.additional_special_tokens)

    generate_model.to(device)

    return generate_model, tokenizer

########################################################
# 학습 함수 및 인퍼런스 함수 정의
########################################################

def train(config) :
    # 사용할 device를 정의합니다.
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available()  else 'cpu')
    print('-'*10, f'학습 시작🥸🥸🥸🥸🥸device : {DEVICE}', '-'*10,)
    generate_model , tokenizer = load_tokenizer_and_model_for_train(config, DEVICE)
    print('='*10, '학습 토크나이저 및 모델 로드 완료', '='*10)
    print(f"스페셜 토큰 확인 :", tokenizer.special_tokens_map,'-'*10)

    # 학습에 사용할 데이터셋을 불러옵니다.
    preprocessor = Preprocess(config['tokenizer']['bos_token'], config['tokenizer']['eos_token'], config) # decoder_start_token: str, eos_token: str
    data_path = DATA_DIR
    val_data, train_inputs_dataset, val_inputs_dataset = prepare_train_dataset(config,preprocessor, data_path, tokenizer)

    run_name = f"{my_name}_{config['general']['model_name']}"
    wandb.init(
        entity = team_name, 
        project = team_project, 
        name = run_name
    )
    
    trainer = load_trainer_for_train(config, generate_model, tokenizer, train_inputs_dataset, val_inputs_dataset, val_data)
    trainer.train()  

    wandb.finish()
    return generate_model, tokenizer

def inference(config, generate_model, tokenizer):
    print('-'*10, f'디바이스 : {DEVICE}', '-'*10,)
    print(torch.__version__)

    if config['inference']['ckt_path'] != "model ckt path":
        generate_model , tokenizer = load_tokenizer_and_model_for_test(config,DEVICE)
        print('='*10, '인퍼런스 토크나이저 및 모델 로드 완료', '='*10)

    preprocessor = Preprocess(config['tokenizer']['bos_token'], config['tokenizer']['eos_token'])

    test_data, test_encoder_inputs_dataset = prepare_test_dataset(config,preprocessor, tokenizer)
    dataloader = DataLoader(test_encoder_inputs_dataset, batch_size=config['inference']['batch_size'])

    summary = []
    text_ids = []
    with torch.no_grad():
        for item in tqdm(dataloader):
            text_ids.extend(item['ID'])
            generated_ids = generate_model.generate(
                input_ids=item['input_ids'].to('cuda:0'),
                attention_mask=item['attention_mask'].to('cuda:0'),
                num_beams=config['evaluation']['generation_params']['num_beams'],
                max_length=config['evaluation']['generation_params']['max_length'],
                min_length=config['evaluation']['generation_params']['min_length'],
                no_repeat_ngram_size=config['evaluation']['generation_params']['no_repeat_ngram_size'],
                length_penalty=config['evaluation']['generation_params']['length_penalty'],
                early_stopping=config['evaluation']['generation_params']['early_stopping'],
                do_sample=config['evaluation']['generation_params']['do_sample'],
                temperature=config['evaluation']['generation_params']['temperature'],
                top_p=config['evaluation']['generation_params']['top_p']
            )
            
            for ids in generated_ids:
                result = tokenizer.decode(
                    ids,
                    skip_special_tokens=False,  # 특수 토큰 유지
                    clean_up_tokenization_spaces=True
                )
                summary.append(result)

    # 정확한 평가를 위하여 노이즈에 해당되는 스페셜 토큰을 제거합니다.
    remove_tokens = config['inference']['remove_tokens']
    preprocessed_summary = summary.copy()
    for token in remove_tokens:
        preprocessed_summary = [sentence.replace(token," ") for sentence in preprocessed_summary]

    output = pd.DataFrame(
        {
            "fname": test_data['fname'],
            "summary" : preprocessed_summary,
        }
    )
    result_path = config['inference']['result_path']
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    output.to_csv(os.path.join(result_path, "output.csv"), index=False)

    return output


# OOM 오류 기록 함수
def log_oom_error(trial_number, learning_rate, batch_size, warmup_ratio, weight_decay, num_train_epochs):
    # 오류 데이터 준비
    error_data = {
        "trial_number": trial_number,
        "lr" : learning_rate,
        "batch_size": batch_size,
        "warmup_ratio": warmup_ratio,
        "weight_decay": weight_decay,
        "num_train_epochs": num_train_epochs
    }
    if not os.path.exists(ERROR_LOG_FILE):
        df = pd.DataFrame(columns=["trial_number", "lr", "batch_size", 
                                   "warmup_ratio", "weight_decay", "num_train_epochs"])
        df.to_csv(ERROR_LOG_FILE, index=False)
    df = pd.read_csv(ERROR_LOG_FILE)
    df = pd.concat([df, pd.DataFrame([error_data])], ignore_index=True)
    df.to_csv(ERROR_LOG_FILE, index=False)

best_rouge_avg = float('-inf')


def generate_parameter_combinations():
    train_param_grid = {
        "learning_rate": [1e-6],
        "gradient_accumulation_steps": [1, 2]
    }
    eval_param_grid = {
        "num_beams": [4, 6],
        "no_repeat_ngram_size": [2, 3],
        # "length_penalty": [1.0, 1.2],
        "repetition_penalty": [1.2],
        "max_length": [100],
    }

    train_keys = train_param_grid.keys()
    train_values = train_param_grid.values()
    eval_keys = eval_param_grid.keys()
    eval_values = eval_param_grid.values()

    train_combinations = list(product(*train_values))
    eval_combinations = list(product(*eval_values))

    param_combinations = []
    for train_combo in train_combinations:
        train_dict = dict(zip(train_keys, train_combo))
        for eval_combo in eval_combinations:
            eval_dict = dict(zip(eval_keys, eval_combo))
            param_combinations.append({
                "training": train_dict,
                "evaluation": {"generation_params": eval_dict}
            })
    total_combinations = len(param_combinations)
    print(f"시도할 조합 수 : {total_combinations}")
    return param_combinations

def grid_search(config):
    logger = logging.getLogger(__name__)
    print(f'GridSearch 시작!!!!!!!!!!!!!!!!!!!!!!')
    EXP_DIR = os.path.join(config['general']['output_dir'], 'parameter_optimization')
    os.makedirs(EXP_DIR, exist_ok=True)
    param_combinations = generate_parameter_combinations()
    best_score = float('-inf')
    best_params = None


    for i, param_combo in enumerate(param_combinations):
        config_copy = copy.deepcopy(config)
        for key, value in param_combo['training'].items():
            config_copy['training'][key] = value
        for key, value in param_combo['evaluation']['generation_params'].items():
            config_copy['evaluation']['generation_params'][key] = value
        # Fixed logging format
        logger.info(f"\n{i+1}번째 조합 시도 : {i+1}/{len(param_combinations)}")
        logger.info(f"현재 파라미터: {param_combo}")  # Fixed logging format

        try:
            model, tokenizer = load_tokenizer_and_model_for_train(config, DEVICE)

            print('='*10, '그리트서치 토크나이저 및 모델 로드 완료', '='*10)
            print(f"스페셜 토큰 확인 :", tokenizer.special_tokens_map)
            preprocessor = Preprocess(
                config['tokenizer']['bos_token'], 
                config['tokenizer']['eos_token'], 
                config  
            )
            data_path = DATA_DIR
            val_data, train_dataset, val_dataset = prepare_train_dataset(config_copy, preprocessor, data_path, tokenizer)
            trainer = load_trainer_for_train(config_copy, model, tokenizer, train_dataset, val_dataset, val_data)
            

            # Train and evaluate
            train_result = trainer.train()
            eval_results = trainer.evaluate()

            # Calculate average ROUGE score
            rouge1 = eval_results.get("eval_rouge-1", 0)
            rouge2 = eval_results.get("eval_rouge-2", 0)
            rougeL = eval_results.get("eval_rouge-l", 0)
            combined_score = eval_results.get("eval_combined_score", 0)
            rouge_avg = (rouge1 + rouge2 + rougeL) / 3 * 100

            # Log to wandb
            wandb.log({
                "rouge1": rouge1,   # 1-gram 정확도     
                "rouge2": rouge2,   # 2-gram 정확도
                "rougeL": rougeL,   # LCS 정확도
                "combined_score": combined_score,
                "rouge_avg": rouge_avg,
                **param_combo
            })

            print(f"평균 ROUGE 점수 : {rouge_avg}")

            # Update best parameters if current score is better
            if rouge_avg > best_score:
                best_score = rouge_avg
                best_params = param_combinations.copy()
                best_params_path= os.path.join(EXP_DIR, f"hyperparameter_tuning_best_params.json")
                with open(best_params_path, 'w') as f:  # w: write, r: read, a: append
                    json.dump(best_params, f, indent=4)
                
                # Save best model
                # model_save_path = "best_model"
                # model.save_pretrained(model_save_path)
                print(f"최고 점수 갱신 : {best_score}")
                print("최고 점수 파라미터 :", best_params)

        except torch.cuda.OutOfMemoryError:
            print(f"Combination {i+1} - CUDA Out of Memory. Logging and continuing...")
            log_oom_error(i+1, param_combo['training']['learning_rate'], 
                          param_combo['training']['batch_size'], 
                          param_combo['training']['warmup_ratio'], 
                          param_combo['training']['weight_decay'], 
                          param_combo['training']['num_train_epochs'])
        
        finally:
            # Clean up
            if 'model' in locals():
                delete_model(model)
            clear_gpu_cache()
            config_copy = None


    return best_params

def hyperparameter_tuning(config):
    run_name = f"GridSearch_{MODEL_NAME}_{KR_TIME}"  # Define run name directly
    
    wandb.init(
        entity=team_name, 
        project=team_project, 
        name=run_name, 
        config=config
    )
    
    try:
        best_config = grid_search(config)
        wandb.config.update(best_config, allow_val_change=True)
        
        # Save the best configuration
        best_config_path = os.path.join(OUTPUT_DIR, 'best_config.yaml')
        with open(best_config_path, 'w') as f:
            yaml.dump(best_config, f, allow_unicode=True)
        
        logger.info(f"Best configuration saved to {best_config_path}")
        logger.info(f"Best configuration: {best_config}")
        
    except Exception as e:
        logger.error(f"Error during hyperparameter tuning: {str(e)}")
        raise e
    finally:
        wandb.finish()
    
    return best_config

def validate_special_tokens(tokenizer, text):
    """Special token 처리가 제대로 되는지 검증"""
    # 토큰화
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)
    
    # Special token 패턴
    pattern = r'#(Person[1-7]|PhoneNumber|Address|PassportNumber|CardNumber|CarNumber|DateOfBirth|Email|SSN)#'
    
    # 원본과 디코딩된 텍스트에서 special token 추출
    original_tokens = set(re.findall(pattern, text))
    decoded_tokens = set(re.findall(pattern, decoded))
    
    # 차이점 확인
    if original_tokens != decoded_tokens:
        logger.warning(f"Special token mismatch!")
        logger.warning(f"Original: {original_tokens}")
        logger.warning(f"Decoded: {decoded_tokens}")
        
    return original_tokens == decoded_tokens


########################################################
# 메인 함수
########################################################

if __name__ == "__main__":
    fix_random_seed()
    logger = setup_logging()
    clear_gpu_cache()

    wandb.login(key=WANDB_KEY)
    with open("./config.yaml", "r") as file:
        loaded_config = yaml.safe_load(file)

    # First train with initial config
    # generate_model, tokenizer = train(loaded_config)
    
    # Then do hyperparameter tuning
    best_config = hyperparameter_tuning(loaded_config)
    
    # Optionally, train again with the best config
    if best_config:
        logger.info("Training with best configuration...")
        final_model, final_tokenizer = train(best_config)

