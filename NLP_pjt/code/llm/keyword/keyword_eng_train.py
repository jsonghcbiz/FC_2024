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
BASELINE_PATH = '/data/ephemeral/home/baseline/data'    
LLM_PATH = '/data/ephemeral/home/NLP_JES/llm'     
RESULT_PATH = os.path.join(LLM_PATH, 'result', 'batch_keywords')
LOG_FILE = os.path.join(RESULT_PATH, 'keywords_kor.log')
ENV_PATH = os.path.join(LLM_PATH, '.env')       
load_dotenv(ENV_PATH)


### ✅ 시간 세팅 ✅
KR_TIMEZONE = pytz.timezone('Asia/Seoul')
KR_NOW = datetime.now(KR_TIMEZONE)

### ✅ 로드할 데이터 범위 세팅 ✅
START_IDX = 5000
END_IDX = 10000

### ✅ 배치 사이즈 세팅 ✅
BATCH_SIZE = 100

### ✅ API 리밋 초과 시 재시도 세팅 ✅
MAX_RETRIES = 5
BASE_DELAY = 15

### ✅ 파일 이름 세팅 ✅
DATA_NAME = 'train_dh_v3.csv'
RESULT_NAME = f'v3_eng_keywords_{START_IDX}_{END_IDX}.json'

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
def load_data(file_path=BASELINE_PATH, start_idx=0, end_idx=None):
    key_df = pd.read_csv(os.path.join(file_path, DATA_NAME))
    if end_idx is not None:
        key_df = key_df.iloc[start_idx:end_idx]
    else:
        key_df = key_df.iloc[start_idx:]

    key_df['dialogue'] = key_df['dialogue'].apply(lambda x: x.replace('\n', ' '))
    key_df['summary'] = key_df['summary'].apply(lambda x: x.replace('\n', ' '))
    key_df['fname'] = key_df['fname'].apply(lambda x: x.replace('\n', ' '))


    return key_df


###############################################
# 모델 호출 함수 (gemini_keywords_stream)
###############################################
async def gemini_keywords_stream(batch: List[Tuple[str, str]], model) -> List[Tuple[str, str]]:
    history_messages = []

    system_prompt = f"""
[역할]
당신은 언어 정보 추출 전문 전문가입니다. 주어진 대화와 요약에서 모두 나타나는 공통적인 **keywords**와 **bigrams**을 식별하는 데 탁월합니다. 당신은 꼼꼼하고 정확하며, 출력 형식에 매우 주의합니다.

[목표]
- 제공된 각 대화(Dialogue)와 해당 요약(Summary)을 분석합니다.
- keywords와 bigrams을 추출할 때 불용어는 무시합니다.
- 대화와 요약의 핵심 내용을 가장 잘 나타내는 최대 5개 이내의 keywords를 식별합니다. 대화와 요약 모두에 나타나는 keywords를 우선시합니다.
- 대화와 요약의 핵심 내용을 가장 잘 나타내는 최대 3개 이내의 bigrams을 식별합니다. 대화와 요약 모두에 나타나는 bigrams을 우선시합니다.
- 출력은 `keywords`와 `bigrams` 키를 가진 객체의 JSON 배열이어야 합니다.
- `keywords` 값은 대화와 요약에서 추출된 공통 키워드를 포함하는 최대 5개 이내의 문자열 목록이어야 합니다.
- `bigrams` 값은 대화와 요약에서 추출된 공통 바이그램을 포함하는 최대 3개 이내의 문자열 목록이어야 합니다.
- 지정된 JSON 객체 배열 형식 외부에 어떠한 텍스트도 추가하지 마십시오.
- 대화(Dialogue)는 #Person1#, #Person2#, #Person3#, #Person4#, #Person5#, #Person6#, #Person7#로 시작하는 여러 줄일 수 있습니다. 대화문을 분석할 때 모든 줄을 하나의 문자열로 간주합니다.


[데이터셋 정보]
- 대화-요약 쌍의 데이터셋이 제공됩니다.
- 각 샘플은 대화(Dialogue)와 요약(Summary)으로 구성됩니다.
- 요약(Summary)은 대화의 핵심 정보와 흐름을 요약한 것입니다.

[지침]
- 각 대화와 요약에 대해, 먼저 핵심 주제를 식별하고, 핵심 주제와 맥락을 가장 잘 파악할 수 있는 keywords와 bigrams를 선별한 뒤 대화(Dialogue)와 요약(Summary)에서 공유되는 가장 관련성이 높고 공통적인 keywords와 bigrams을 선택하는 방식으로 추출합니다.
- 대화와 요약 간에 공유되는 가장 관련성이 높고 공통적인 keywords와 bigrams을 식별하는 데 집중합니다.
- 조사, 어미, 보조동사와 같이 내용상 중요하지 않은 단어들은 불용어로 처리하고 제외합니다. 의미 있는 조사의 경우는 원래 단어에 붙여 한개의 단어로 처리합니다.
- 공통 keywords가 5개 미만이거나 공통 bigrams가 3개 미만인 경우, 대화와 가장 관련성이 높은 항목만 반환합니다.
- keywords를 선택할 때 대화의 주제를 가장 잘 나타내는 명사 또는 명사구에 집중하세요.
- bigrams를 선택할 때 대화에서 핵심적인 행동, 관계 또는 주요 주제를 나타내는 표현에 집중하세요.
- 대화의 주요 주제, 상호 작용, 특수 토큰(#Person1#, #Person2#, #Person3#, #Person4#, #Person5#, #Person6#, #Person7#, #Address#, #CarNumber#, #CardNumber#, #DateOfBirth#, #Email#, #PassportNumber#, #PhoneNumber#, #SSN#) 간의 연결에 주목하여 분석합니다.
- 특수 토큰은 keywords와 bigrams에 포함하지 않습니다.
- Keywords와 bigrams은 토픽 모델링에 사용되며, 대화 요약을 위해 해당 대화와 파일 이름(레이블)에 매핑됩니다.
- 지정된 JSON 출력 형식을 엄격하게 준수하십시오. 출력은 지정된 형식의 객체 json 배열이어야 합니다.
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
    content_parts.append(f"Keywords와 bigrams을 추출하세요. 대화(Dialogue)와 요약(Summary)의 가장 중요한 내용을 가장 잘 반영하는 keywords와 bigrams을 선택하세요.")
    content = "\n".join(content_parts)
    
    dialogue_content = content_types.ContentDict(
        role="user",
        parts=[content]
    )
    history_messages.append(dialogue_content)

    
    max_retries = MAX_RETRIES
    base_delay = BASE_DELAY 

    for attempt in range(max_retries):
        try:
            chat = model.start_chat(history=history_messages)
            response = chat.send_message(" ", stream=True)
            
            full_response = ""
            for chunk in response:
                full_response += chunk.text
            

            try:
                cleaned_response = full_response.replace("**Dialogue**", "").replace("**Summary**", "")
                cleaned_response = full_response.strip()
                if cleaned_response.startswith("```json"):
                    cleaned_response = cleaned_response[7:-3]
                json_response = json.loads(cleaned_response)
                print("디버깅 - JSON 응답:", json_response)
                
                if isinstance(json_response, dict):
                    print("디버깅 - 키워드:", json_response.get('keywords'))
                    print("디버깅 - 바이그램:", json_response.get('bigrams'))
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
        return [{"keywords": [], "bigrams": []} for _ in range(len(batch))]



def save_intermediate_results(results: List[dict], batch_num: int) -> None:
    timestamp = KR_NOW.strftime("%m%d_%H%M%S")
    intermediate_file = os.path.join(RESULT_PATH, f"train_v3_keywords_{START_IDX}_{END_IDX}_batch_{batch_num}.json")
    
    try:

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

    dialogues = key_df['dialogue'][:num_samples].tolist()
    summaries = key_df['summary'][:num_samples].tolist()
    fnames = key_df['fname'][:num_samples].tolist()

    keywords_results_list = []
    start_time = time.time()
    
    model = gen_model
    
    pairs = list(zip(dialogues, summaries, fnames))
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
                    print(f"\nKeywords: {', '.join(results[idx]['keywords'])}")
                    print(f"\nBigrams: {', '.join(results[idx]['bigrams'])}")
                    
                    # print(f"\nKeywords: {results[idx]['keywords']}")
                    # print(f"\nBigrams: {results[idx]['bigrams']}")
                    print("=="*50)

                    await asyncio.sleep(0.5)
                    if (batch_idx + 1) % 10 == 0:  
                        await asyncio.sleep(1)

                    if idx % 10 == 0:  
                        logging.info(f"배치 {batch_idx+1} - {idx}/{len(batch)} 항목 처리 완료")
                else:
                    logging.error(f"배치 {batch_idx+1} - {idx}번째 항목 처리 중 오류 발생. 결과 없음")

        except Exception as e:
            logging.error(f"배치 {batch_idx+1}, 처리 중 오류: {str(e)}")
        
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
            
            if total_processed % 1200 == 0:
                save_intermediate_results(batch_results, total_processed // 1200)
                logging.info(f"중간 결과 저장 완료: {total_processed}개 항목")
                batch_results = [] 
        
        except Exception as e:
            logging.error(f"배치 처리 중 오류 발생: {str(e)}")
            continue

    keywords_results_list = all_results  
    total_time = time.time() - start_time
    print(f"\n총 처리 시간: {total_time:.2f} seconds")


    os.makedirs(RESULT_PATH, exist_ok=True)
    output_file = os.path.join(RESULT_PATH, RESULT_NAME)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(keywords_results_list, f, indent=4, ensure_ascii=False)
    print(f"키워드 결과 저장: {output_file}")

    return keywords_results_list

if __name__ == '__main__':
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [KST] - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(LOG_FILE, mode='w', encoding='utf-8'),
            logging.StreamHandler()
        ])
    
    logging.Formatter.converter = lambda *args: KR_NOW.timetuple()
    
    async def main():
        key_df = load_data(BASELINE_PATH, start_idx=START_IDX, end_idx=END_IDX)  
        await extract_keywords(key_df, batch_size=BATCH_SIZE)

    asyncio.run(main())


