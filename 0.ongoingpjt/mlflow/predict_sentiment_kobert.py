import torch
import os
from kobert_tokenizer import KoBERTTokenizer
import re
from transformers import BertModel
from safetensors import safe_open
from safetensors.torch import load_file

# Custom BERT Classifier class (needs to be identical to the training class)
class CustomBertClassifier(torch.nn.Module):
    def __init__(self, model_name, num_labels):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = torch.nn.Dropout(0.1)
        self.classifier = torch.nn.Linear(self.bert.config.hidden_size, num_labels)
        
    def forward(self, input_ids=None, attention_mask=None, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return torch.nn.functional.softmax(logits, dim=1)

def clean_text(text):
    text = re.sub(r'<.*?>', '', text)  # HTML 태그 제거
    text = re.sub(r'[^\w\s]', '', text)  # 특수문자 제거
    text = re.sub(r'\d+', '', text)  # 숫자 제거
    text = text.lower()  # 소문자로 변환
    text = text.strip()  # 문자열 양쪽 공백 제거
    text = text.replace('br', '')  # 'br' 태그 제거
    return text

def predict_sentiment(text, model_path, model_name='skt/kobert-base-v1'):
    # Device 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 모델 로드
    model = CustomBertClassifier(model_name, num_labels=2)
    state_dict = load_file(os.path.join(model_path, 'model.safetensors'))
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    # 토크나이저 로드
    tokenizer = KoBERTTokenizer.from_pretrained(
        model_name,
        sp_model_kwargs={'nbest_size': -1, 'alpha': 0.6, 'enable_sampling': True}
    )
    
    # 텍스트 전처리
    cleaned_text = clean_text(text)
    
    # 토크나이징
    encoded = tokenizer(
        cleaned_text,
        padding='max_length',
        truncation=True,
        max_length=300,
        return_tensors='pt'
    )
    
    # 예측
    with torch.no_grad():
        input_ids = encoded['input_ids'].to(device)
        attention_mask = encoded['attention_mask'].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probabilities = outputs.squeeze()
        predicted_class = torch.argmax(probabilities).item()
        
    # 결과 해석
    sentiment = "긍정" if predicted_class == 1 else "부정"
    confidence = probabilities[predicted_class].item()
    
    return {
        'text': text,
        'sentiment': sentiment,
        'confidence': f'{confidence:.2%}',
        'probabilities': {
            '부정': f'{probabilities[0].item():.2%}',
            '긍정': f'{probabilities[1].item():.2%}'
        }
    }

# 사용 예시
if __name__ == "__main__":
    # 모델 경로 설정 (여기에 실제 저장된 모델 경로를 입력하세요)
    # model_path = 'kobert'
    model_path = 'kobert_konply'  # 예시 경로 (실제 저장된 모델 경로로 변경 필요)
    
    # 테스트할 텍스트 예시들
    test_texts = [
        "이 영화 정말 재미있었어요! 다음에 또 보고 싶네요.",
        "최악이었어요. 시간 낭비였습니다.",
        "그저 그랬어요. 기대했던 것보다는 별로였네요."
    ]
    
    # 각 텍스트에 대해 감성 분석 수행
    for text in test_texts:
        result = predict_sentiment(text, model_path)
        print("\n=== 감성 분석 결과 ===")
        print(f"텍스트: {result['text']}")
        print(f"감성: {result['sentiment']}")
        print(f"확신도: {result['confidence']}")
        print(f"확률 분포: {result['probabilities']}") 