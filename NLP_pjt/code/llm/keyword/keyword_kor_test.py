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
LOG_FILE = os.path.join(RESULT_PATH, 'keywords_kor_test.log')
ENV_PATH = os.path.join(LLM_PATH, '.env')   
load_dotenv(ENV_PATH)

### ✅ 시간 세팅 ✅
KR_TIMEZONE = pytz.timezone('Asia/Seoul')
KR_NOW = datetime.now(KR_TIMEZONE)

### ✅ 로드할 데이터 범위 세팅 ✅
START_IDX = 0
END_IDX = 499


### ✅ 배치 사이즈 세팅 ✅
BATCH_SIZE = 100

### ✅ API 리밋 초과 시 재시도 세팅 ✅
MAX_RETRIES = 5
BASE_DELAY = 15

### ✅파일 이름 세팅 ✅
DATA_NAME = 'test.csv'
REFERENCE_NAME = 'train_dh_v3_kor_key.csv'
FILE_NAME = f'test_kor_key_{START_IDX}_{END_IDX}.json'  # 저장 파일 이름

### ✅ Few-shot 예시 샘플링 ✅
FS_NUM_SAMPLES = 50

###############################################
# GEMINI 모델 세팅
###############################################
### ✅ GEMINI API 키 세팅 ✅
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

### ✅ GEMINI 모델 세팅 ✅
# gen_model = genai.GenerativeModel('gemini-2.0-flash-exp')  
# gen_model = genai.GenerativeModel(model_name='learnlm-1.5-pro-experimental')
gen_model = genai.GenerativeModel(model_name='gemini-2.0-flash-thinking-exp-1219')



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
    key_df['fname'] = key_df['fname'].apply(lambda x: x.replace('\n', ' '))


    return key_df

def load_reference_data(file_path=BASELINE_PATH):
    reference_df = pd.read_csv(os.path.join(file_path, REFERENCE_NAME))
    return reference_df

###############################################
# Few-shot 예시 샘플링
###############################################
def few_shot_sample(reference_df: pd.DataFrame) -> Tuple[List[str], List[str], List[str], List[str], List[str]]:
    reference_samples = reference_df[:FS_NUM_SAMPLES]
    
    ref_fname = reference_samples['fname'].tolist()
    ref_dialogue = reference_samples['dialogue'].tolist()
    ref_summary = reference_samples['summary'].tolist()
    ref_keywords = reference_samples['keywords'].tolist()
    ref_bigrams = reference_samples['bigrams'].tolist()
    
    return ref_fname, ref_dialogue, ref_summary, ref_keywords, ref_bigrams

def format_few_shot_examples(ref_fname: List[str], ref_dialogue: List[str], ref_summary: List[str], ref_keywords: List[str], ref_bigrams: List[str]) -> str:
    formatted_examples = []
    for i in range(len(ref_fname)):
        example = f"""예시 {ref_fname[i]}:
user: **Dialogue**
{ref_dialogue[i]}

model: **Reasoning:**
1) 대화(Dialogue)를 분석하여 핵심 주제와 맥락을 파악하고, 대화 요약(Summary)에 사용될 keywords와 bigrams을 추출할 준비를 합니다.
2) 주요 주제와 맥락을 가장 잘 파악할 수 있는 keywords와 bigrams을 식별하고, 대화(Dialogue)에서 높은 관련성과 중요도를 가진 keywords와 bigrams을 선택합니다.

model: **Keywords**
{ref_keywords[i]}

model: **Bigrams**
{ref_bigrams[i]}"""
        formatted_examples.append(example)
    
    return "\n".join(formatted_examples)



###############################################
# 모델 호출 함수 (gemini_keywords_stream)
###############################################
async def gemini_keywords_stream(batch: List[Tuple[str, str]], model, formatted_examples:str, ref_fname=None, ref_dialogue=None, ref_keywords=None, ref_bigrams=None) -> List[Tuple[str, str]]:
    history_messages = []

    system_prompt = f"""
[역할]
당신은 언어 정보 추출 전문 전문가입니다. 주어진 대화에서 나타나는 **keywords**와 **bigrams**을 식별하는 데 탁월합니다. 추출된 단어들은 대화 요약(Dialogue Summarization)에 사용될 것임을 기억하세요. 당신은 꼼꼼하고 정확하며, 출력 형식에 매우 주의합니다.

[목표]
- 제공된 대화(Dialogue)를 분석하여 대화 내용의 핵심 주제와 맥락을 분석합니다. 
- Keyword와 bigram을 추출할 때 불용어는 무시합니다. 조사(예: ‘을’, ‘를’, ‘에’ 등), 접사(예: ‘들’, ‘이’ 등), 보조 동사(예: ‘하다’, ‘되다’ 등)는 불용어로 간주하고 최대한 형태소 분석을 통해 의미를 파악하여 불용어를 제거합니다. 만약 Bigrams의 경우 조사가 의미가 있다면, 원래 단어에 붙여서 하나의 단어로 취급합니다.
- 대화(Dialogue)의 주요 주제, 관계, 또는 핵심 행동을 나타내는 최대 5개 이내의 keywords를 식별합니다. 'keywords' 값은 대화에서 추출된 최대 5개 이내의 문자열 목록이어야 합니다.
- 대화(Dialogue)의 주요 주제, 관계, 또는 핵심 행동을 나타내는 최대 5개 이내의 bigrams를 식별합니다. 'bigrams' 값은 대화에서 추출된 최대 5개 이내의 문자열 목록이어야 합니다.
- 출력은 `keywords`와 `bigrams` 키를 가진 객체의 JSON 배열이어야 합니다.
- 지정된 JSON 객체 배열 형식 외부에 어떠한 텍스트도 추가하지 마십시오.
- 대화(Dialogue)는 #Person1#, #Person2#, #Person3#, #Person4#, #Person5#, #Person6#, #Person7#로 시작하는 여러 줄일 수 있습니다. 대화문을 분석할 때 모든 줄을 하나의 문자열로 간주합니다.


[데이터셋 정보]
- 대화(Dialogue)데이터셋이 제공됩니다.
- 대화(Dialogue)는 나중에 요약(Summary)을 생성하는데 사용될 것입니다.

[지침]
- 각 대화(Dialogue)에 대해, 먼저 핵심 주제를 식별하고, 핵심 주제와 맥락을 가장 잘 파악할 수 있는 keywords와 bigrams를 선별한 뒤 대화(Dialogue)에서 높은 관련성과 중요도를 가진 keywords 및 bigrams을 선택하는 방식으로 추출합니다.
- 추출된 Keywords와 bigrams은 대화(Dialogue)에 대한 요약(Summary)을 생성하는데 사용될 것임을 기억하세요.
- 대화(Dialogue)의 핵심 주제와 맥락을 가장 잘 나타내는 keywords와 bigrams을 선별하세요.
- Keywords는 5개 미만이거나 bigrams는 5개 미만인 경우, 대화의 주요 주제와 맥락을 반영하는 항목만 반환합니다.
- Keywords를 선택할 때, 가장 대표적인 명사나 명사구를 중심으로 하세요.
- Bigrams를 선택할 때, 대화에서 주요 행동, 관계, 또는 주요 주제를 나타내는 표현을 중심으로 하세요.
- 대화의 주요 주제, 상호작용, 그리고 특수 토큰(#Person1#, #Person2#, #Person3#, #Person4#, #Person5#, #Person6#, #Person7#, #Address#, #CarNumber#, #CardNumber#, #DateOfBirth#, #Email#, #PassportNumber#, #PhoneNumber#, #SSN#) 사이의 관계를 중심으로 하세요.
- 특수 토큰은 keywords와 bigrams에 포함하지 않지만, 추후 맥락에 따라 요약(Summary) 생성 시 반영될 수 있습니다
- Keywords와 bigrams는 주제 모델링을 위해 사용되며, 해당 대화와 fnames(라벨)에 매핑됩니다. 대화를 요약(Dialogue Summarization)하는데 사용될 것입니다.
- 지정된 JSON 출력 형식을 엄격하게 따르세요. 출력은 지정된 형식의 JSON 객체 배열이어야 합니다.

[예시]
{formatted_examples}
"""

    system_prompt_content = content_types.ContentDict(
        role="user", 
        parts=[system_prompt]
    )
    history_messages.append(system_prompt_content)

    if ref_fname and ref_dialogue:
        for i in range(min(len(ref_fname), len(ref_dialogue))):
            example_code_content = content_types.ContentDict(
                role="user",
                parts=[f"**Code:**\n{ref_fname[i]}"]
            )
            example_korean_dialogue_content = content_types.ContentDict(
                role="user",
                parts=[f"**Dialogue:**\n{ref_dialogue[i]}"]
            )
            example_keywords_content = content_types.ContentDict(
                role="user",
                parts=[f"**Keywords:**\n{ref_keywords[i]}"]
            )
            example_bigrams_content = content_types.ContentDict(
                role="model",
                parts=[f"**Bigrams:**\n{ref_bigrams[i]}"]
            )
            history_messages.append(example_code_content)
            history_messages.append(example_korean_dialogue_content)
            history_messages.append(example_keywords_content)
            history_messages.append(example_bigrams_content)

    content_parts = []
    for idx, dialogue in enumerate(batch, start=1):
        content_parts.append(f"""
**Dialogue {idx}**
{dialogue}
"""
        )
    content_parts.append(f"주어진 대화에서 가장 중요한 내용을 반영하는 keywords와 bigrams을 추출하세요. 추출된 keywords와 bigrams은 대화에 대한 요약(Summary)을 생성하는데 사용될 것입니다.")
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
                cleaned_response = full_response.replace("**Dialogue**", "")
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
    intermediate_file = os.path.join(RESULT_PATH, f"test_kor_key_{START_IDX}_{END_IDX}_batch_{batch_num}.json")
    
    try:

        for result in results:
            if not result:
                logging.warning(f"결과가 비어있음: {result}")
                
        with open(intermediate_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        logging.info(f"중간 결과 저장: {intermediate_file}")
    except Exception as e:
        logging.error(f"중간 결과 저장 실패: {str(e)}")

async def extract_keywords(key_df: pd.DataFrame, reference_df: pd.DataFrame, batch_size: int = BATCH_SIZE, num_samples: int = -1) -> List[Dict]:
    num_samples = len(key_df) if num_samples == -1 else min(num_samples, len(key_df))

    dialogues = key_df['dialogue'][:num_samples].tolist()
    fnames = key_df['fname'][:num_samples].tolist()

    keywords_results_list = []
    start_time = time.time()
    
    model = gen_model
    
    pairs = list(zip(dialogues, fnames))
    batches = [pairs[i:i+batch_size] for i in range(0, len(pairs), batch_size)]
    
    logging.info(f"총 데이터 수: {len(pairs)}")
    logging.info(f"배치 크기: {batch_size}")
    logging.info(f"총 배치 수: {len(batches)}")

    ref_fname, ref_dialogue, ref_summary, ref_keywords, ref_bigrams = few_shot_sample(reference_df)
    formatted_examples = format_few_shot_examples(ref_fname, ref_dialogue, ref_summary, ref_keywords, ref_bigrams)

    async def process_batch(batch_idx: int, batch: List[Tuple[str, str]]) -> List[Dict]:
        batch_start_time = time.time()
        batch_results = []
        
        logging.info(f"배치 {batch_idx+1}/{len(batches)} 처리 시작 (크기: {len(batch)})")
        
        try:
            dialogues = []
            for item in batch:
                dialogue = item[0] if len(item) > 0 else ""
                dialogues.append(dialogue)
            
            results = await gemini_keywords_stream(
                dialogues, 
                model,
                formatted_examples
            )
            
            for idx, item in enumerate(batch):
                if idx < len(results):
                    dialogue = item[0] if len(item) > 0 else ""
                    fname = item[1] if len(item) > 1 else ""
                    
                    result_dict = {
                        'fname': fname,
                        'dialogue': dialogue,
                        'keywords': results[idx]["keywords"] if results[idx]["keywords"] else [],
                        'bigrams': results[idx]["bigrams"] if results[idx]["bigrams"] else []
                    }
                    batch_results.append(result_dict)

                    print(f"==GEMINI Keywords Batch {batch_idx+1}=="*5)
                    print(f"fname: {fname}")
                    print(f"\nDialogue: {dialogue}")
                    print(f"\nKeywords: {', '.join(results[idx]['keywords'])}")
                    print(f"\nBigrams: {', '.join(results[idx]['bigrams'])}")
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


    os.makedirs(RESULT_PATH, exist_ok=True)
    output_file = os.path.join(RESULT_PATH, FILE_NAME)
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
        reference_df = load_reference_data(BASELINE_PATH)
        await extract_keywords(key_df, reference_df, batch_size=BATCH_SIZE)

    asyncio.run(main())


