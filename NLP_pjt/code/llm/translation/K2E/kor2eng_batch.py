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
RESULT_PATH = '/data/ephemeral/home/NLP_JES/llm/result/batch_translate'
LOG_FILE = os.path.join(RESULT_PATH, 'translation_batch.log')
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
BATCH_SIZE = 10     # ⚠️⚠️⚠️ output token 초과 시 데이터 추출이 제대로 안됨. 시간이 걸리더라도 배치 사이즈를 줄여야 함.

### ✅ API 리밋 초과 시 재시도 세팅 ✅
MAX_RETRIES = 5
BASE_DELAY = 15


###############################################
# GEMINI 모델 세팅
###############################################
### ✅ GEMINI API 키 세팅 ✅
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

### ✅ GEMINI 모델 세팅 ✅
gen_model = genai.GenerativeModel('gemini-2.0-flash-exp')  




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
    train_df = pd.read_csv(os.path.join('/data/ephemeral/home/NLP_JES/llm/result/batch_translate', 'extracted_data.csv'))
    if end_idx is not None:
        train_df = train_df.iloc[start_idx:end_idx]
    else:
        train_df = train_df.iloc[start_idx:]
    train_df['summary'] = train_df['summary'].apply(lambda x: x.replace('\n', ' '))

    val_df = pd.read_csv(os.path.join('/data/ephemeral/home/NLP_JES/llm/result/batch_translate','dev_dh_eng.csv'))
    val_df['korean_summary'] = val_df['korean_summary'].apply(lambda x: x.replace('\n', ' '))
    return train_df, val_df



###############################################
# 모델 호출 함수 (gemini_translation_stream)
###############################################
async def gemini_translation_stream(batch: List[Tuple[str, str]], model) -> List[Tuple[str, str]]:
    history_messages = []

    system_prompt = f"""
[Role]
You are a professional English and Korean translation expert. You excel at translating given Korean Dialogue and Korean Summary into English, respectively. You are a meticulous and format-conscious translator.

[Goal]
- Translate each Korean Dialogue and its corresponding Korean Summary into English.
- Output must be a JSON list of objects, with the following keys: `english_dialogue` and `english_summary`.
- Each `english_dialogue` and `english_summary` value must be the direct translation of its corresponding Korean Dialogue and Korean Summary.
- Do not add any text outside of the specified JSON format.
- Maintain the original sentence count.
- Dialogue can be multiline. Include all line breaks in the translated output.

[Dataset Information]
- A dataset of dialogue summary pairs is provided.
- Each sample consists of a Korean Dialogue and a Korean Summary.
- The Korean Summary is a summary of the key information and flow of the Korean Dialogue.

[Instructions]
- Focus on translating each Korean dialogue and its corresponding summary accurately.
- Preserve all the special tokens and characters from the original text.
- Strictly adhere to the JSON output format. 
"""

    system_prompt_content = content_types.ContentDict(
        role="user", 
        parts=[system_prompt]
    )
    history_messages.append(system_prompt_content)


    content_parts = []
    for idx, (korean_dialogue, korean_summary) in enumerate(batch, start=1):
        content_parts.append(f"""
**Korean Dialogue {idx}**
{korean_dialogue}

**Korean Summary {idx}**
{korean_summary}
"""
        )
    content_parts.append(f"Please translate each Korean Dialogue and Summary to English based on the output format in the prompt.")
    content = "\n".join(content_parts)
    
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
            

            try:
                cleaned_response = full_response.strip()
                if cleaned_response.startswith("```json"):
                    cleaned_response = cleaned_response[7:-3]
                json_response = json.loads(cleaned_response)
                print("Debug - Raw JSON response:", json_response)
                
                if isinstance(json_response, dict):
                    print("디버깅 - 번역 대화문:", json_response.get('english_dialogue'))
                    print("디버깅 - 번역 요약문:", json_response.get('english_summary'))
                    
                    if not json_response.get('english_dialogue') or not json_response.get('english_summary'):
                        print("디버깅 - 번역 필드 비어있음")
                        
                results = []

                if isinstance(json_response, dict):
                    json_response = [json_response]
                
                for item  in json_response:
                    english_dialogue = item.get("english_dialogue").strip()
                    english_summary = item.get("english_summary").strip()
                    if not english_dialogue or not english_summary:
                        logging.warning(f"JSON 응답 {item}에서 'english_dialogue' or 'english_summary' 없음")
                        results.append(("", ""))
                        continue
                    results.append((english_dialogue, english_summary))
                if len(results) != len(batch):
                    logging.warning(f"배치 크기 불일치: {len(results)} != {len(batch)}")
                    results.extend([("", "")] * (len(batch) - len(results)))
                return results
            except json.JSONDecodeError:
                logging.error(f"json 응답 형식 불일치: {full_response}")
                return [("", "")] * len(batch)
            
        except exceptions.ResourceExhausted as e:
            if attempt < max_retries - 1:
                wait_time = base_delay * (2 ** attempt)  + random.uniform(0, 1)
                print(f"\n리미트 초과, 시도 {attempt + 1}/{max_retries}. {wait_time} 초 대기...")
                await asyncio.sleep(wait_time)  
            else:
                print(f"최대 시도 횟수 도달. 오류: {e}")
                raise
                
        except requests.exceptions.RequestException as e: 
          if attempt < max_retries - 1:
              wait_time = base_delay * (2 ** attempt) + random.uniform(0, 1)
              print(f"\nAPI 오류, 시도 {attempt + 1}/{max_retries}. {wait_time:.2f} 초 대기...")
              await asyncio.sleep(wait_time)
          else:
              print(f"최대 시도 횟수 도달. API 오류: {e}")
              raise
                
        except Exception as e:
            print(f"예기치 않은 오류: {str(e)}")
            if attempt < max_retries - 1:
                wait_time = base_delay * (2 ** attempt) + random.uniform(0, 1)
                print(f"재시도 중... {wait_time:.2f} 초 대기...")
                await asyncio.sleep(wait_time)
            else:
                raise

    return [("", "")] * len(batch)

def save_intermediate_results(results: List[dict], batch_num: int) -> None:
    timestamp = KR_NOW.strftime("%m%d_%H%M%S")
    intermediate_file = os.path.join(RESULT_PATH, f"train_dh_v2_eng_{START_IDX}_{END_IDX}_batch_{batch_num}.json")
    
    try:

        for result in results:
            if not result.get("english_dialogue") or not result.get("english_summary"):
                logging.warning(f"번역 결과 비어있음: {result}")
                
        with open(intermediate_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        logging.info(f"중간 결과 저장: {intermediate_file}")
    except Exception as e:
        logging.error(f"중간 결과 저장 실패: {str(e)}")

async def translate(train_df: pd.DataFrame, batch_size: int = BATCH_SIZE, num_samples: int = -1) -> List[Dict]:
    num_samples = len(train_df) if num_samples == -1 else min(num_samples, len(train_df))

    train_dialogues = train_df['dialogue'][:num_samples].tolist()
    train_summaries = train_df['summary'][:num_samples].tolist()
    train_fnames = train_df['fname'][:num_samples].tolist()

    translate_results_list = []
    start_time = time.time()
    
    model = gen_model
    
    pairs = list(zip(train_dialogues, train_summaries, train_fnames))
    batches = [pairs[i:i+batch_size] for i in range(0, len(pairs), batch_size)]
    
    logging.info(f"총 데이터 수: {len(pairs)}")
    logging.info(f"배치 크기: {batch_size}")
    logging.info(f"총 배치 수: {len(batches)}")

    async def process_batch(batch_idx: int, batch: List[Tuple[str, str, str]]) -> List[Dict]:
        batch_start_time = time.time()
        batch_results = []
        
        logging.info(f"배치 {batch_idx+1}/{len(batches)} 처리 시작 (크기: {len(batch)})")
        
        try:
            dialogues_and_summaries = [(d, s) for d, s, _ in batch]
            results = await gemini_translation_stream(dialogues_and_summaries, model)
            
            for idx, (korean_dialogue, korean_summary, fname) in tqdm(enumerate(batch), total=len(batch), desc=f"배치 {batch_idx+1} 처리 중...", leave=False):
                if idx < len(results):
                    english_dialogue, english_summary = results[idx]

                    result_dict = {
                        "fname": fname,
                        "korean_dialogue": korean_dialogue,
                        "korean_summary": korean_summary,
                        "english_dialogue": english_dialogue,
                        "english_summary": english_summary
                    }
                    batch_results.append(result_dict)

                    print(f"==GEMINI Translation Batch {batch_idx+1}=="*5)
                    print(f"fname: {fname}")
                    print(f"\nKorean Dialogue: {korean_dialogue}")
                    print(f"\nKorean Summary: {korean_summary}")
                    print(f"\nEnglish Dialogue: {english_dialogue}")
                    print(f"\nEnglish Summary: {english_summary}")
                    print("=="*50)

                    await asyncio.sleep(0.5)
                    if (batch_idx + 1) % 10 == 0:  
                        await asyncio.sleep(1)

                    if idx % 10 == 0: 
                        logging.info(f"배치 {batch_idx+1} - {idx}/{len(batch)} 항목 처리 완료")
                else:
                    logging.error(f"배치 {batch_idx+1} - {idx}번째 항목 처리 중 오류 발생. 결과 없음")

        except Exception as e:
            logging.error(f"배치 {batch_idx+1} 처리 중 오류: {str(e)}")
        
        batch_end_time = time.time()
        batch_duration = batch_end_time - batch_start_time
        logging.info(f"배치 {batch_idx+1} 완료. 소요시간: {batch_duration:.2f}초")
        
        return batch_results



    tasks = [process_batch(i, batch) for i, batch in enumerate(batches)]
    

    batch_results = []
    all_results = []  
    total_processed = 0
    
    for future in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="배치 처리 중..."):

        
        try:
            batch_result = await future
            batch_results.extend(batch_result)
            all_results.extend(batch_result) 
            
            total_processed += len(batch_result)
            logging.info(f"현재까지 처리된 총 항목 수: {total_processed}/{len(pairs)}")
            
            if total_processed % 100 == 0:
                save_intermediate_results(batch_results, total_processed // 100)
                logging.info(f"중간 결과 저장 완료: {total_processed}개 항목")
                batch_results = [] 
        
        except Exception as e:
            logging.error(f"배치 처리 중 오류 발생: {str(e)}")
            continue

    translate_results_list = all_results  
    total_time = time.time() - start_time
    print(f"\n총 처리 시간: {total_time:.2f} seconds")

    os.makedirs(RESULT_PATH, exist_ok=True)
    output_file = os.path.join(RESULT_PATH, FILE_NAME)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(translate_results_list, f, indent=4, ensure_ascii=False)
    print(f"번역 결과 저장: {output_file}")

    return translate_results_list

if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [KST] - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(LOG_FILE, mode='w', encoding='utf-8'),
            logging.StreamHandler()
        ])
    
    # Add timezone converter to logging formatter
    logging.Formatter.converter = lambda *args: KR_NOW.timetuple()
    
    async def main():
        train_df, val_df = load_data(DATA_PATH, start_idx=START_IDX, end_idx=END_IDX)
        logging.info(f"로드된 데이터 크기: {len(train_df)}") 
        await translate(train_df, batch_size=BATCH_SIZE)

    asyncio.run(main())


