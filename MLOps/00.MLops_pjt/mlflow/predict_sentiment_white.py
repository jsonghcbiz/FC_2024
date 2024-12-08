import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
import pickle
from torch.nn import functional as F

# Add the CustomBertClassifier class definition
class CustomBertClassifier(torch.nn.Module):
    def __init__(self, model_name, num_labels):
        super().__init__()
        self.bert = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels)
        
    def forward(self, input_ids=None, attention_mask=None, labels=None):
        outputs = self.bert(input_ids=input_ids, 
                          attention_mask=attention_mask,
                          labels=labels)
        return outputs if labels is not None else (None, outputs.logits)

class SentimentClassifier:
    def __init__(self, model_path):
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load tokenizer
        self.model_name = 'WhitePeak/bert-base-cased-Korean-sentiment'
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            sp_model_kwargs={'nbest_size': -1, 'alpha': 0.6, 'enable_sampling': True}
        )
        
        # Load model
        model_file = os.path.join(model_path, [f for f in os.listdir(model_path) if f.endswith('.pkl')][0])
        with open(model_file, 'rb') as f:
            self.model = pickle.load(f)
        self.model.to(self.device)
        self.model.eval()  # Set model to evaluation mode

    def predict_sentiment(self, text):
        # Preprocess text
        inputs = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=300,
            return_tensors='pt'
        )
        
        # Move inputs to device
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)
        
        # Get prediction
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs[1] if isinstance(outputs, tuple) else outputs.logits
            probabilities = F.softmax(logits, dim=1)
            prediction = torch.argmax(probabilities, dim=1)
            
            # Fix: Get individual probability values correctly
            neg_prob = probabilities[0][0].item()  # Probability for negative
            pos_prob = probabilities[0][1].item()  # Probability for positive
            confidence = pos_prob if prediction.item() == 1 else neg_prob
        
        # Convert prediction to sentiment
        sentiment = "긍정" if prediction.item() == 1 else "부정"
        
        return {
            'text': text,
            'sentiment': sentiment,
            'confidence': f'{confidence:.2%}',
            'probabilities': {
                '부정': f'{neg_prob:.2%}',
                '긍정': f'{pos_prob:.2%}'
            }
        }   

# Example usage
if __name__ == "__main__":
    # Replace with your actual model path
    model_path = "white_konply"  # Update this with your model directory
    
    # Initialize classifier
    classifier = SentimentClassifier(model_path)
    
    # Test some examples
    test_texts = [
        "이 영화 정말 재미있었어요! 다음에 또 보고 싶네요.",
        "최악이었어요. 시간 낭비였습니다.",
        "그저 그랬어요. 기대했던 것보다는 별로였네요."
    ]
    

    for text in test_texts:
        result = classifier.predict_sentiment(text)
        print("\n=== 감성 분석 결과 ===")
        print(f"텍스트: {result['text']}")
        print(f"감성: {result['sentiment']}")
        print(f"확신도: {result['confidence']}")
        print(f"확률 분포: {result['probabilities']}") 

    # for text in test_texts:
        # result = classifier.predict_sentiment(text)
        # print(f"\n텍스트: {text}")
        # print(f"감성: {result['sentiment']}")
        # print(f"확신도: {result['confidence']:.2%}")
        # print(f"긍정 확률: {result['probabilities']['긍정']:.2%}")
        # print(f"부정 확률: {result['probabilities']['부정']:.2%}") 