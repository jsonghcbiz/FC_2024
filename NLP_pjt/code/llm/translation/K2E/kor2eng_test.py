import pandas as pd
import os
import time
from tqdm import tqdm
from dotenv import load_dotenv
import google.generativeai as genai
import google.generativeai.types as content_types
import json
from google.api_core import exceptions
from typing import List, Optional, Tuple, Dict
import asyncio
import multiprocessing
import logging
import random
import requests
import re
import pytz
from datetime import datetime

###############################################
# 데이터 경로 및 환경변수 세팅
###############################################
### ✅ 데이터 경로 세팅 ✅
DATA_PATH = '/data/ephemeral/home/baseline/data'
BASE_PATH = '/data/ephemeral/home/NLP_JES'
ENV_PATH = os.path.join(BASE_PATH, 'llm', '.env')
RESULT_PATH = os.path.join(BASE_PATH, 'llm', 'result', 'batch_translate')
LOG_FILE = os.path.join(RESULT_PATH, 'translation_single.log')
load_dotenv(ENV_PATH)

### ✅ 시간 세팅 ✅
KR_TIMEZONE = pytz.timezone('Asia/Seoul')
KR_NOW = datetime.now(KR_TIMEZONE)

### ✅ 로드할 데이터 범위 세팅 ✅    현재까지 0~3000까지 처리됨.
START_IDX = 0
END_IDX = 200

### ✅ 저장할 파일 이름 세팅 ✅
FILE_NAME = f'extracted_v2_eng_{START_IDX}_{END_IDX}.json'

### ✅ 배치 사이즈 세팅 ✅
BATCH_SIZE = 25     # ⚠️⚠️⚠️ output token 초과 시 데이터 추출이 제대로 안됨. 시간이 걸리더라도 배치 사이즈를 줄여야 함.

### ✅ API 리밋 초과 시 재시도 세팅 ✅
MAX_RETRIES = 5
BASE_DELAY = 15

FS_NUM_SAMPLES = 50

###############################################
# GEMINI 모델 세팅
###############################################
### ✅ GEMINI API 키 세팅 ✅
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

### ✅ GEMINI 모델 세팅 ✅
gen_model = genai.GenerativeModel('gemini-2.0-flash-exp')  
# gen_model = genai.GenerativeModel(model_name='learnlm-1.5-pro-experimental')
# gen_model = genai.GenerativeModel(model_name='gemini-2.0-flash-thinking-exp-1219')




###############################################
# 특수 토큰 세팅
###############################################
SPECIAL_TOKENS = ['#Address#', '#CarNumber#', '#CardNumber#', '#DateOfBirth#', 
            '#Email#', '#PassportNumber#', '#PhoneNumber#', '#SSN#',
            '#Person1#', '#Person2#', '#Person3#', '#Person4#', '#Person5#', '#Person6#', '#Person7#']
PERSON_TOKENS = ['#Person1#', '#Person2#', '#Person3#', '#Person4#', '#Person5#', '#Person6#', '#Person7#']

###############################################
# 데이터 로드
###############################################
def load_data(file_path=RESULT_PATH, start_idx=0, end_idx=None):
    train_df = pd.read_csv(os.path.join('/data/ephemeral/home/NLP_JES/llm/result/batch_translate','extracted_data.csv'))
    if end_idx is not None:
        train_df = train_df.iloc[start_idx:end_idx]
    else:
        train_df = train_df.iloc[start_idx:]
    return train_df

def load_reference_data(file_path=RESULT_PATH):
    reference_file = os.path.join('/data/ephemeral/home/NLP_JES/llm/result/batch_translate', f"dev_dh_eng.csv")
    reference_df = pd.read_csv(reference_file)
    return reference_df

###############################################
# Few-shot 예시 샘플링
###############################################
def few_shot_sample(reference_df):
    few_shot_samples = reference_df[:FS_NUM_SAMPLES]
    
    # We'll need to update these column names to match your actual data
    ref_korean_dialogue = few_shot_samples['korean_dialogue'].tolist()  # Changed back to 'korean_dialogue'
    ref_english_dialogue = few_shot_samples['english_dialogue'].tolist()
    
    return ref_korean_dialogue, ref_english_dialogue

def format_few_shot_examples(ref_korean_dialogue, ref_english_dialogue):
    formatted_examples = ""
    for i in range(len(ref_korean_dialogue)):
        formatted_examples += f"""
                                Example {i+1}:
                                user: {ref_korean_dialogue[i]}
                                model: {ref_english_dialogue[i]}
                                """
    return formatted_examples

###############################################
# 모델 호출 함수 (gemini_translation_stream)
###############################################
async def gemini_translation(dialogue, ref_korean_dialogue, ref_english_dialogue, model):
    history_messages = []

    system_prompt = f"""
[Role]
You are a professional English and Korean translation expert. You excel at translating given Korean Dialogue into English. You are a meticulous and format-conscious translator.

[Goal]
- Translate **all** the given Korean Dialogue into English.
- Reference the Example English Dialogues when translating.
- Output must be in the following format:
**English Dialogue:** #Person1#: First line of Dialogue
#Person1#: Second line of Dialogue
...
#Person1#: Last line of Dialogue


- English Dialogue must be the translation of the corresponding Korean Dialogue. You must translate all the dialogues.
- **The dialogue can be multiple lines long, and you must include all the lines in your translation.** Include the line break characters in the translation of the Dialogue. Do not add additional text or phrases outside the specified format.

[Dataset Information]
- A dataset of dialogue is provided.
- Each sample consists of a Korean Dialogue.
- The Korean Summary summarizes the key information and flow of the Korean Dialogue, including the topic, purpose, main characters, key events, and decisions, allowing a quick understanding of the dialogue.

[Instructions]
- **Analyze Dataset:** Analyze the characteristics of the Korean Dialogue in the provided dataset. Specifically, focus on the sentence structure, key information inclusion methods, phrase usage patterns, and keyword/phrase frequency.
- **Focus on Key Information:** Identify key information and words in each sentence of the Korean Dialogue and translate accordingly.
- **Maintain Conciseness:** Remove unnecessary details, repetitions, opinions, and background information. Focus on the core content and keep the translated text concise while preserving the original sentence structure.
- **Preserve Characters:** If characters (#Person1#, #Person2#, #Person3#, #Person4#, #Person5#, #Person6#, #Person7#, etc.) appear in each sentence, maintain them in the translated text.
- **Maintain Special Tokens:** If special tokens exist in each sentence, maintain them in the translated text. ('#Address#', '#CarNumber#', '#CardNumber#', '#DateOfBirth#', '#Email#', '#PassportNumber#', '#PhoneNumber#', '#SSN#', '#Person1#', '#Person2#', '#Person3#', '#Person4#', '#Person5#', '#Person6#', '#Person7#')
- **Maintain Sentence Count:** Make sure the translated sentences keep the same number as the original sentence count.
- Your response should only contain the translated text and match the example output format.
"""
    system_prompt_content = content_types.ContentDict(
        role="user",
        parts=[system_prompt]
    )
    history_messages.append(system_prompt_content)

    if ref_korean_dialogue and ref_english_dialogue:
        for i in range(min(len(ref_korean_dialogue), len(ref_english_dialogue))):
            example_korean_dialogue_content = content_types.ContentDict(
                role="user",
                parts=[f"**Example Korean Dialogue:**\n{ref_korean_dialogue[i]}"]
            )
            example_english_dialogue_content = content_types.ContentDict(
                role="model",
                parts=[f"**Example English Dialogue:**\n{ref_english_dialogue[i]}"]
            )
            history_messages.append(example_korean_dialogue_content)
            history_messages.append(example_english_dialogue_content)
    
    
    
    content = f"""
**DIALOGUE**
{dialogue}
    
Please translate each Korean Dialogue to English based on the output format in the prompt."""
    dialogue_content = content_types.ContentDict(
        role="user",
        parts=[content]
    )
    history_messages.append(dialogue_content)


    max_retries = 5
    base_delay = 15

    for attempt in range(max_retries):
        try:
            chat = model.start_chat(history=history_messages)
            response = chat.send_message(" ", stream=True)

            full_response = ""
            for chunk in response:
                full_response += chunk.text
            cleaned_response = full_response.replace("**Example English Dialogue:**", "").replace("**Example Korean Dialogue:**", "").replace("**English Dialogue:**", "").replace("**DIALOGUE**", "")
            return cleaned_response
            
        except exceptions.ResourceExhausted as e:
            if attempt < max_retries - 1:
                wait_time = base_delay * (2 ** attempt)  # Exponential backoff
                print(f"\n리미트 초과, 시도 {attempt + 1}/{max_retries}. {wait_time} 초 대기...")
                await asyncio.sleep(wait_time)
            else:
                print(f"최대 시도 횟수 도달. 오류: {e}")
                raise
                
        except Exception as e:
            print(f"예기치 않은 오류: {str(e)}")
            if attempt < max_retries - 1:
                wait_time = base_delay * (2 ** attempt)
                print(f"재시도 중... {wait_time} 초 대기...")
                await asyncio.sleep(wait_time)
            else:
                raise
    return ""  # Return empty string if all retries fail



async def translate(reference_df: pd.DataFrame, num_samples: int = -1):
    if isinstance(reference_df, tuple):
        reference_df = reference_df[0]
        
    reference_data = load_reference_data()
    
    test_dialogue = reference_df[:num_samples] if num_samples > 0 else reference_df
    translate_results_list = []
    start_time = time.time()
    
    model = gen_model
    
    async def process_sample(dialogue, fname, ref_korean_dialogue, ref_english_dialogue, model):
        pred_translation = await gemini_translation(
            dialogue=dialogue,
            ref_korean_dialogue=ref_korean_dialogue,
            ref_english_dialogue=ref_english_dialogue,
            model=model
        )
        return {'translation': pred_translation, 'fname': fname}  # Return both translation and fname

    for idx, row in tqdm(test_dialogue.iterrows(), total=len(test_dialogue)):
        dialogue = row['dialogue']
        fname = row['fname']
        ref_korean_dialogue, ref_english_dialogue = few_shot_sample(reference_data)
        result = await process_sample(
            dialogue,
            fname,  # Include fname in the process_sample call
            ref_korean_dialogue, 
            ref_english_dialogue, 
            model
        )
        
        translate_results_list.append({
            'fname': result['fname'],
            'korean_dialogue': dialogue,
            'english_dialogue': result['translation']
        })
        
        if (idx+1) % 100 == 0:
            end_time = time.time()
            elapsed_time = end_time - start_time
            if elapsed_time < 60:
                wait_time = 15 - elapsed_time + 5  
                print(f"시간 초과 발생 : {elapsed_time} 초")
                print(f"대기 중... {wait_time} 초 후 다음 요청 시작")
                await asyncio.sleep(wait_time)
            start_time = time.time()

        print(f"==GEMINI Translation Sample {idx}=="*5)
        print(f"fname: {fname}")
        print(f"\nKorean Dialogue: {dialogue}")
        print(f"\nEnglish Dialogue: {result['translation']}")
        print("=="*50)

        await asyncio.sleep(0.5)
        if idx % 10 == 0:  
            await asyncio.sleep(1)
                
    total_time = time.time() - start_time
    print(f"\n총 처리 시간: {total_time:.2f} seconds")

    # Save final results
    if not os.path.exists(RESULT_PATH):
        os.makedirs(RESULT_PATH)
    output_file = os.path.join(RESULT_PATH, FILE_NAME)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(translate_results_list, f, indent=4, ensure_ascii=False)
    print(f"번역 결과 저장: {output_file}")

    return translate_results_list

# Example Usage (수정된 함수에 맞게 사용 예시도 수정):
if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(LOG_FILE, mode='w', encoding='utf-8'),
                  logging.StreamHandler()])
    async def main():
        train_df = load_data(DATA_PATH, start_idx=START_IDX, end_idx=END_IDX)
        await translate(train_df)

    asyncio.run(main())
