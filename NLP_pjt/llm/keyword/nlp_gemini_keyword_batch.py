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
DATA_PATH = '/data/ephemeral/home/baseline/data'
BASE_PATH = '/data/ephemeral/home/NLP_JES'
ENV_PATH = os.path.join(BASE_PATH, 'llm', '.env')
RESULT_PATH = os.path.join(BASE_PATH, 'llm', 'result', 'batch_keywords')
LOG_FILE = os.path.join(RESULT_PATH, 'keywords_batch.log')
load_dotenv(ENV_PATH)



BATCH_SIZE = 100

# Train에서 불러올 데이터 범위
START_IDX = 0
END_IDX = 100
FILE_NAME = f'train_dh_v2_eng_keywords_{START_IDX}_{END_IDX}.json'


###############################################
# GEMINI 모델 세팅
###############################################
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
gen_model = genai.GenerativeModel('gemini-2.0-flash-exp')  



# upstage : 16.347

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
def load_data(file_path=DATA_PATH, start_idx=0, end_idx=None):
    key_df = pd.read_json(os.path.join(file_path, 'train_dh_v2_eng_exp.json'))
    if end_idx is not None:
        key_df = key_df.iloc[start_idx:end_idx]
    else:
        key_df = key_df.iloc[start_idx:]
    
    # dialogue와 fname만 전처리
    key_df['english_dialogue'] = key_df['english_dialogue'].apply(lambda x: x.replace('\n', ' '))
    key_df['english_summary'] = key_df['english_summary'].apply(lambda x: x.replace('\n', ' '))
    key_df['fname'] = key_df['fname'].apply(lambda x: x.replace('\n', ' '))


    return key_df



###############################################
# 모델 호출 함수 (gemini_keywords_stream)
###############################################
async def gemini_keywords_stream(batch: List[Tuple[str, str]], model) -> List[Tuple[str, str]]:
    history_messages = []

    system_prompt = f"""
[Role]
You are a professional information extraction expert. You excel at identifying common **keywords** and **bigrams** between a given Dialogue and its corresponding Summary. You are meticulous, precise, and format-conscious in your output.

[Goal]
- Analyze each provided Dialogue and its corresponding Summary.
- Identify up to 5 keywords that best reflects the most significant content of the Dialogue and Summary. Prioritize keywords and bigrams that appear in both the Dialogue and Summary.
- Identify up to 3 bigrams that best reflects the most significant content of the Dialogue and Summary. Prioritize bigrams that appear in both the Dialogue and Summary.
- Output must be a JSON list of objects, with the following keys: `keywords` and `bigrams`.
- The `keywords` value must be a list of at most 5 strings, containing the common keywords extracted from the Dialogue and Summary.
- The `bigrams` value must be a list of at most 3 strings, containing the common bigrams extracted from the Dialogue and Summary.
- Do not add any text outside of the specified JSON format.
- Dialogue can be multiline starting with #Person1#, #Person2#, #Person3#, #Person4#, #Person5#, #Person6#, #Person7#. Consider all lines into a single string.

[Dataset Information]
- A dataset of dialogue summary pairs is provided.
- Each sample consists of a Dialogue and a Summary.
- The Summary is a summary of the key information and flow of the Dialogue.

[Instructions]
- Focus on identifying the most relevant and common keywords and bigrams shared between the Dialogue and Summary. If fewer than 5 common keywords and 3 common bigrams exist, then return only those.
- Analyze the major topics of the conversation the interactions and connections between special tokens(#Person1#, #Person2#, #Person3#, #Person4#, #Person5#, #Person6#, #Person7#, #Address#, #CarNumber#, #CardNumber#, #DateOfBirth#, #Email#, #PassportNumber#, #PhoneNumber#, #SSN#).
- Do not include the special tokens in the keywords and bigrams.
- Keywords and bigrams will be used for topic modeling and mapped to corresponding dialogues and fnames(lable) for dialogue summarization.
- Strictly adhere to the specified JSON output format. The output should be a json list of objects in the specified format.
"""

    system_prompt_content = content_types.ContentDict(
        role="user", 
        parts=[system_prompt]
    )
    history_messages.append(system_prompt_content)

    content_parts = []
    for idx, (dialogue, summary) in enumerate(batch, start=1):
        content_parts.append(f"""
**Dialogue {idx}**
{dialogue}

**Summary {idx}**
{summary}
"""
        )
    content_parts.append(f"Please extract keywords and bigrams that best reflects the most significant content of the dialogue and summary.")
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
            # response = chat.send_message(system_prompt, stream=True)
            
            full_response = ""
            for chunk in response:
                full_response += chunk.text
            

            try:
                cleaned_response = full_response.replace("**Dialogue**", "").replace("**Summary**", "")
                cleaned_response = full_response.strip()
                if cleaned_response.startswith("```json"):
                    cleaned_response = cleaned_response[7:-3]
                json_response = json.loads(cleaned_response)
                print("Debug - Raw JSON response:", json_response)
                
                if isinstance(json_response, dict):
                    print("Debug - Keywords:", json_response.get('keywords'))
                    print("Debug - Bigrams:", json_response.get('bigrams'))
                results = []
                
                if isinstance(json_response, dict):
                    json_response = [json_response]
                
                for item in json_response:
                    keywords = item.get("keywords", [])
                    if isinstance(keywords, str):
                        keywords = [k.strip() for k in keywords.split(',')]
                    elif isinstance(keywords, list):
                        keywords = [k.strip() if isinstance(k, str) else k for k in keywords]
                    else:
                        keywords = []
                    
                    # bigrams 처리 추가
                    bigrams = item.get("bigrams", [])
                    if isinstance(bigrams, str):
                        bigrams = [b.strip() for b in bigrams.split(',')]
                    elif isinstance(bigrams, list):
                        bigrams = [b.strip() if isinstance(b, str) else b for b in bigrams]
                    else:
                        bigrams = []
                        
                    results.append({"keywords": keywords, "bigrams": bigrams})
                
                if len(results) != len(batch):
                    logging.warning(f"배치 크기 불일치: {len(results)} != {len(batch)}")
                    results.extend([{"keywords": [], "bigrams": []} for _ in range(len(batch) - len(results))])
                
                return results
            except json.JSONDecodeError:
                logging.error(f"json 응답 형식 불일치: {full_response}")
                return [{"keywords": [], "bigrams": []} for _ in range(len(batch))]
            
        except exceptions.ResourceExhausted as e:
            if attempt < max_retries - 1:
                wait_time = base_delay * (2 ** attempt)  + random.uniform(0, 1)# Exponential backoff
                print(f"\n리미트 초과, 시도 {attempt + 1}/{max_retries}. {wait_time} 초 대기...")
                await asyncio.sleep(wait_time)  
            else:
                print(f"최대 시도 횟수 도달. 오류: {e}")
                raise
                
        except requests.exceptions.RequestException as e: # 일반적인 API 오류 처리
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
        return [{"keywords": [], "bigrams": []} for _ in range(len(batch))]



def save_intermediate_results(results: List[dict], batch_num: int) -> None:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    intermediate_file = os.path.join(RESULT_PATH, f"train_dh_v2_eng_{START_IDX}_{END_IDX}_batch_{batch_num}.json")
    
    try:
        # Verify results before saving
        for result in results:
            if not result:
                logging.warning(f"결과가 비어있음: {result}")
                
        with open(intermediate_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        logging.info(f"중간 결과 저장: {intermediate_file}")
    except Exception as e:
        logging.error(f"중간 결과 저장 실패: {str(e)}")

async def extract_keywords(key_df: pd.DataFrame, batch_size: int = BATCH_SIZE, num_samples: int = -1) -> List[Dict]:
    num_samples = len(key_df) if num_samples == -1 else min(num_samples, len(key_df))

    english_dialogues = key_df['english_dialogue'][:num_samples].tolist()
    english_summaries = key_df['english_summary'][:num_samples].tolist()
    key_fnames = key_df['fname'][:num_samples].tolist()

    keywords_results_list = []
    start_time = time.time()
    
    model = gen_model
    
    pairs = list(zip(english_dialogues, english_summaries, key_fnames))
    batches = [pairs[i:i+batch_size] for i in range(0, len(pairs), batch_size)]
    
    logging.info(f"총 데이터 수: {len(pairs)}")
    logging.info(f"배치 크기: {batch_size}")
    logging.info(f"총 배치 수: {len(batches)}")

    async def process_batch(batch_idx: int, batch: List[Tuple[str, str, str]]) -> List[Dict]:
        batch_start_time = time.time()
        batch_results = []
        
        logging.info(f"배치 {batch_idx+1}/{len(batches)} 처리 시작 (크기: {len(batch)})")
        
        try:
            dialogues_and_summaries = []
            for item in batch:
                dialogue = item[0] if len(item) > 0 else ""
                summary = item[1] if len(item) > 1 else ""
                dialogues_and_summaries.append((dialogue, summary))
            
            results = await gemini_keywords_stream(dialogues_and_summaries, model)
            
            for idx, item in enumerate(batch):
                if idx < len(results):
                    dialogue = item[0] if len(item) > 0 else ""
                    summary = item[1] if len(item) > 1 else ""
                    fname = item[2] if len(item) > 2 else ""
                    
                    result_dict = {
                        'fname': fname,
                        'dialogue': dialogue,
                        'summary': summary,
                        'keywords': results[idx]["keywords"] if results[idx]["keywords"] else [],
                        'bigrams': results[idx]["bigrams"] if results[idx]["bigrams"] else []
                    }
                    batch_results.append(result_dict)

                    print(f"==GEMINI Keywords Batch {batch_idx+1}=="*5)
                    print(f"fname: {fname}")
                    print(f"\nDialogue: {dialogue}")
                    print(f"\nSummary: {summary}")
                    print(f"\nKeywords: {results[idx]['keywords']}")
                    print(f"\nBigrams: {results[idx]['bigrams']}")
                    print("=="*50)

                    await asyncio.sleep(0.5)
                    if (batch_idx + 1) % 10 == 0:  
                        await asyncio.sleep(1)

                    if idx % 10 == 0:  # 배치 내 진행상황 로깅
                        logging.info(f"배치 {batch_idx+1} - {idx}/{len(batch)} 항목 처리 완료")
                else:
                    logging.error(f"배치 {batch_idx+1} - {idx}번째 항목 처리 중 오류 발생. 결과 없음")

        except Exception as e:
            logging.error(f"배치 {batch_idx+1}, 처리 중 오류: {str(e)}")
        
        batch_end_time = time.time()
        batch_duration = batch_end_time - batch_start_time
        logging.info(f"배치 {batch_idx+1} 완료. 소요시간: {batch_duration:.2f}초")
        
        return batch_results


    # Create tasks with batch indices
    tasks = [process_batch(i, batch) for i, batch in enumerate(batches)]
    
    # Use tqdm to track progress of batches
    batch_results = []
    all_results = []  # Add this to store all results
    total_processed = 0
    
    for future in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="배치 처리 중..."):

       
        try:
            batch_result = await future
            batch_results.extend(batch_result)
            all_results.extend(batch_result)  # Add this line to accumulate all results
            
            total_processed += len(batch_result)
            logging.info(f"현재까지 처리된 총 항목 수: {total_processed}/{len(pairs)}")
            
            if total_processed % 100 == 0:
                save_intermediate_results(batch_results, total_processed // 100)
                logging.info(f"중간 결과 저장 완료: {total_processed}개 항목")
                batch_results = [] 
        
        except Exception as e:
            logging.error(f"배치 처리 중 오류 발생: {str(e)}")
            continue

    keywords_results_list = all_results  
    total_time = time.time() - start_time
    print(f"\n총 처리 시간: {total_time:.2f} seconds")

    # Save final results
    os.makedirs(RESULT_PATH, exist_ok=True)
    output_file = os.path.join(RESULT_PATH, FILE_NAME)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(keywords_results_list, f, indent=4, ensure_ascii=False)
    print(f"키워드 결과 저장: {output_file}")

    return keywords_results_list

if __name__ == '__main__':
    KST = pytz.timezone('Asia/Seoul')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [KST] - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(LOG_FILE, mode='w', encoding='utf-8'),
            logging.StreamHandler()
        ])
    
    logging.Formatter.converter = lambda *args: datetime.now(KST).timetuple()
    
    async def main():
        key_df = load_data(DATA_PATH, start_idx=START_IDX, end_idx=END_IDX)  
        await extract_keywords(key_df, batch_size=BATCH_SIZE)

    asyncio.run(main())


