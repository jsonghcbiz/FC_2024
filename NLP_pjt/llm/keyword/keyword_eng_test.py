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
LOG_FILE = os.path.join(RESULT_PATH, 'keywords_eng_test.log')
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

### ✅ 저장할 파일 이름 세팅 ✅
DATA_NAME = 'test_eng.csv'
REFERENCE_NAME = 'train_dh_v3_eng_key.csv'
FILE_NAME = f'test_eng_keywords_{START_IDX}_{END_IDX}.json'

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

    key_df['english_dialogue'] = key_df['english_dialogue'].apply(lambda x: x.replace('\n', ' '))
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
1) Analyze the English Dialogue to identify the core topic and context, and prepare to extract keywords and bigrams that will be used for Summarization in the future.
2) Identify the most relevant and important keywords and bigrams that best represent the core topic and context of the Dialogue.

model: **Keywords**
{ref_keywords[i]}

model: **Bigrams**
{ref_bigrams[i]}"""
        formatted_examples.append(example)
    
    return "\n".join(formatted_examples)



###############################################
# 모델 호출 함수 (gemini_keywords_stream)
###############################################
async def gemini_keywords_stream(batch: List[Tuple[str, str]], model, formatted_examples:str, ref_fname=None, ref_dialogue=None, ref_summary=None, ref_keywords=None, ref_bigrams=None) -> List[Tuple[str, str]]:
    history_messages = []

    system_prompt = f"""
[Role]
You are a professional information extraction expert. You excel at identifying **keywords** and **bigrams** from a given Dialogue which would be used for Summarization of the given Dialogue later on. You are meticulous, precise, and format-conscious in your output.

[Goal]
- Analyze each provided Dialogue and understand the context.
- Ignore stop words such as particles, suffixes, and auxiliary verbs. If a particle is meaningful, it should be attached to the original word and treated as a single word.
- Identify up to 5 keywords that best reflects the core topic, significant content, and interactions within the Dialogue. The `keywords` value must be a list of at most 5 strings, extracted from the Dialogue.
- Identify up to 3 bigrams that best reflects the core topic, significant content, and interactions within the Dialogue. The `bigrams` value must be a list of at most 3 strings, extracted from the Dialogue.
- Output must be a JSON list of objects, with the following keys: `keywords` and `bigrams`.
- Do not add any text outside of the specified JSON format.
- Dialogue can be multiline starting with #Person1#, #Person2#, #Person3#, #Person4#, #Person5#, #Person6#, #Person7#. Consider all lines into a single string.

[Dataset Information]
- A dataset of dialogues will be provided.
- Each sample consists of a multiline Dialogue of conversation.
- The Dialogues will be used for Dialogue Summarization later on.

[Instructions]
- For each Dialogue, first identify the core topic, then select the keywords that reflect best the topic and context fo the Dialogue, and finally select the bigrams that describe the keywords.
- Keep in mind that the extracted keywords and bigrams will be used for Dialogue Summarization.
- Focus on identifying keywords and bigrams that are most relevant to the major topics and context of the conversation.
- Keywords are fewer than 5 or bigrams are fewer than 3, only return those that best represent the context of the conversation.
- When selecting keywords, focus on the most representative nouns or noun phrases that best represent the topic of the Dialogue.
- When selecting bigrams, focus on expressions that represent key actions, relationships, or main topics in the Dialogue.
- Focus on the major topics of the conversation, interactions, and connections between special tokens(#Person1#, #Person2#, #Person3#, #Person4#, #Person5#, #Person6#, #Person7#, #Address#, #CarNumber#, #CardNumber#, #DateOfBirth#, #Email#, #PassportNumber#, #PhoneNumber#, #SSN#).
- Do not include the special tokens in the keywords and bigrams.
- Keywords and bigrams will be used for topic modeling and mapped to corresponding dialogues and fnames(lable) for dialogue summarization.
- Strictly adhere to the specified JSON output format. The output should be a json list of objects in the specified format.

[Few-shot Examples]
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
            example_english_dialogue_content = content_types.ContentDict(
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
            history_messages.append(example_english_dialogue_content)
            history_messages.append(example_keywords_content)
            history_messages.append(example_bigrams_content)

    content_parts = []
    for idx, dialogue in enumerate(batch, start=1):
        content_parts.append(f"""
**Dialogue {idx}**
{dialogue}
"""
        )
    content_parts.append(f"Please extract keywords and bigrams that best reflects the most significant content of the dialogue.")
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
    intermediate_file = os.path.join(RESULT_PATH, f"test_eng_key_{START_IDX}_{END_IDX}_batch_{batch_num}.json")
    
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

    dialogues = key_df['english_dialogue'][:num_samples].tolist()
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
                formatted_examples)
            
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


