# 라이브러리 설정
import os
import pandas as pd
from datetime import datetime
from datasets import load_dataset, Dataset
import joblib
import numpy as np
import datetime
import evaluate
import torch
import mlflow
import mlflow.sklearn

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments

imdb_db = pd.read_csv('/data/ephemeral/home/IMDB_Dataset.csv', index_col=0)
device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
model_name = 'huawei-noah/TinyBERT_General_4L_312D'
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
accuracy = evaluate.load('accuracy')
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
model_path = f'./tinybert_model_test/tinybert_model_{timestamp}'
mlflow.set_tracking_uri('http://127.0.0.1:5000')
mlflow.set_experiment('IMDB_Model_Training')

def prepare_data():
    dataset = Dataset.from_pandas(imdb_db)
    dataset = dataset.train_test_split(test_size=0.2)
    
    label2id = {'positive': 1, 'negative': 0}
    dataset = dataset.map(lambda x: {'label': label2id[x['sentiment']]})

    dataset = dataset.map(tokenize_data, batched=True)

    return dataset, label2id

def tokenize_data(batch):
    return tokenizer(batch['review'], padding=True, truncation=True, max_length=300, return_tensors='pt')

def train_model(label2id):
    id2label = {0: 'negative', 1: 'positive'}
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id
    )
    return model

def evaluate_model(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

def train_and_evaluate(dataset, label2id):
    model = train_model(label2id)

    args = TrainingArguments(
        output_dir='train_dir',
        overwrite_output_dir=True,
        num_train_epochs=2,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        eval_strategy='epoch',
        learning_rate=2e-5
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        tokenizer=tokenizer,
        compute_metrics=evaluate_model
    )
    # torch.save(model.state_dict(), model_path)

    trainer.save_model(model_path)
    mlflow.log_artifact(model_path, artifact_path="model")
    trainer.train()
    trainer.evaluate()


if __name__ == "__main__":
    # model_name = 'huawei-noah/TinyBERT_General_4L_312D'
    dataset, label2id = prepare_data()
    mlflow.autolog()
    with mlflow.start_run(run_name=model_name):
        train_and_evaluate(dataset, label2id)
