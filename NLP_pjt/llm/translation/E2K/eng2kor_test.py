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
RESULT_PATH = os.path.join(LLM_PATH, 'result', 'batch_translate')
LOG_FILE = os.path.join(RESULT_PATH, 'translation_batch.log')
ENV_PATH = os.path.join(LLM_PATH, '.env')
load_dotenv(ENV_PATH)



### ✅ 시간 세팅 ✅
KR_TIMEZONE = pytz.timezone('Asia/Seoul')
KR_NOW = datetime.now(KR_TIMEZONE)

### ✅ 로드할 데이터 범위 세팅 ✅    현재까지 0~3000까지 처리됨.
START_IDX = 180
END_IDX = 210 

### ✅ 배치 사이즈 세팅 ✅
BATCH_SIZE = 10    # ⚠️⚠️⚠️ output token 초과 시 데이터 추출이 제대로 안됨. 시간이 걸리더라도 배치 사이즈를 줄여야 함.

### ✅ API 리밋 초과 시 재시도 세팅 ✅
MAX_RETRIES = 5
BASE_DELAY = 15

### ✅ 파일 이름 세팅 ✅
DATA_NAME = 'bart_large_cnn_v2_eng.csv'         # inference 요약문 데이터
ORIGINAL_NAME = 'test_eng.csv'                      # 원본 한글 대화문 
TEST_KEYWORD_NAME = 'test_kor_key.csv'          # 테스트 한글 키워드 데이터
REF_DIASUM_NAME = 'train_dh_v3_eng.csv'            # 학습 대화문, 요약문 데이터
REF_KEY_NAME = 'train_dh_v3_kor_key.csv'      # 학습 대화문, 요약문, 키워드 데이터
FILE_NAME = f'test_output_{START_IDX}_{END_IDX}.json'          # 저장 파일 이름

### ✅ Few-shot 예시 샘플링 ✅
FS_NUM_SAMPLES = 100

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
def load_data(file_path=BASELINE_PATH, start_idx=0, end_idx=None, fs_num_samples=0):
    output_df = pd.read_csv(os.path.join(file_path, 'output', 'bart_large_cnn_v2_eng.csv'))
    test_original_df = pd.read_csv(os.path.join(file_path, 'original', 'test_eng.csv'))
    test_keyword_df = pd.read_csv(os.path.join(file_path, 'keywords', 'test_kor_key.csv'))
    if end_idx is not None:
        output_df = output_df.iloc[start_idx:end_idx]
        test_original_df = test_original_df.iloc[start_idx:end_idx]
        test_keyword_df = test_keyword_df.iloc[start_idx:end_idx]
    else:
        output_df = output_df.iloc[start_idx:]
        test_original_df = test_original_df.iloc[start_idx:]
        test_keyword_df = test_keyword_df.iloc[start_idx:]
    test_original_df['korean_dialogue'] = test_original_df['korean_dialogue'].apply(lambda x: x.replace('\n', ' '))
    test_original_df['english_dialogue'] = test_original_df['english_dialogue'].apply(lambda x: x.replace('\n', ' '))
    
    
    ref_diasum_df = pd.read_csv(os.path.join(file_path, 'original', 'train_dh_v3_eng.csv'))
    ref_key_df = pd.read_csv(os.path.join(file_path, 'keywords', 'train_dh_v3_kor_key.csv'))
    if FS_NUM_SAMPLES > 0:
        ref_diasum_df = ref_diasum_df.iloc[:FS_NUM_SAMPLES]
        ref_key_df = ref_key_df.iloc[:FS_NUM_SAMPLES]
    ref_diasum_df['korean_dialogue'] = ref_diasum_df['korean_dialogue'].apply(lambda x: x.replace('\n', ' '))
    ref_diasum_df['english_summary'] = ref_diasum_df['english_summary'].apply(lambda x: x.replace('\n', ' '))
    ref_diasum_df['korean_summary'] = ref_diasum_df['korean_summary'].apply(lambda x: x.replace('\n', ' '))

    # Debug print statements
    print("\nDataFrame shapes:")
    print(f"output_df: {output_df.shape}")
    print(f"test_original_df: {test_original_df.shape}")
    print(f"test_keyword_df: {test_keyword_df.shape}")
    print(f"ref_diasum_df: {ref_diasum_df.shape}")
    print(f"ref_key_df: {ref_key_df.shape}")

    print("\nDataFrame columns:")
    print(f"output_df: {output_df.columns.tolist()}")
    print(f"test_original_df: {test_original_df.columns.tolist()}")
    print(f"test_keyword_df: {test_keyword_df.columns.tolist()}")
    print(f"ref_diasum_df: {ref_diasum_df.columns.tolist()}")
    print(f"ref_key_df: {ref_key_df.columns.tolist()}")

    # Verify required columns exist
    required_columns = {
        'test_original_df': ['fname', 'korean_dialogue', 'english_dialogue'],
        'output_df': ['fname', 'summary'],
        'test_keyword_df': ['fname', 'keywords', 'bigrams'],
        'ref_diasum_df': ['fname', 'korean_dialogue', 'english_summary', 'korean_summary'],
        'ref_key_df': ['fname', 'keywords', 'bigrams']
    }

    for df_name, columns in required_columns.items():
        df = locals()[df_name]
        missing_columns = [col for col in columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns in {df_name}: {missing_columns}")

    return output_df, test_original_df, test_keyword_df, ref_diasum_df, ref_key_df


###############################################
# Few-shot 예시 샘플링
###############################################
def few_shot_sample(ref_diasum_df: pd.DataFrame, ref_key_df: pd.DataFrame) -> Tuple[List[str], List[str], List[str], List[str], List[str], List[str]]:
    # train dh v3 eng: korean_dialogue, english_summary, korean_summary
    ref_fname = ref_diasum_df['fname'].tolist()
    ref_kr_dialogue = ref_diasum_df['korean_dialogue'].tolist()
    ref_en_summary = ref_diasum_df['english_summary'].tolist()
    ref_kr_summary = ref_diasum_df['korean_summary'].tolist()
    
    # train dh v3 kor key: key, bigram
    ref_key = ref_key_df['keywords'].tolist()
    ref_bigram = ref_key_df['bigrams'].tolist()
    
    return ref_fname, ref_kr_dialogue, ref_en_summary, ref_kr_summary, ref_key, ref_bigram

def format_few_shot_examples(ref_fname: List[str], ref_kr_dialogue: List[str], ref_en_summary: List[str], ref_kr_summary: List[str], ref_key: List[str], ref_bigram: List[str]) -> str:
    formatted_examples = []
    for i in range(len(ref_fname)):
        example = f"""Example {ref_fname[i]}:
user: **Korean Dialogue**
{ref_kr_dialogue[i]}
user: **Keywords**
{ref_key[i]}
user: **Bigrams**
{ref_bigram[i]}
user: **English Summary**
{ref_en_summary[i]}
model: **Reasoning:**
1) I will analyse the English summary and the Korean dialogue to understand the context of both English summary and Korean dialogue.
2) I will translate the English summary to Korean, ensuring it matches the context of the Korean dialogue provided. 
3) I will use the Keywords and Bigrams to ensure the translation is accurate and natural. I will include the Keywords and Bigrams in the translation.
4) I will keep the original meaning and also preserve the special tokens.

model: **Korean Summary**
{ref_kr_summary[i]}"""
        formatted_examples.append(example)
    
    return "\n".join(formatted_examples)



###############################################
# 모델 호출 함수 (gemini_translation_stream)
###############################################
async def gemini_translation_stream(batch: List[Tuple[str, str, str, str, str]], model, formatted_examples:str, ref_fname=None, ref_kr_dialogue=None, ref_en_summary=None, ref_kr_summary=None, ref_key=None, ref_bigram=None) -> List[Tuple[str, str, str, str, str]]:
    history_messages = []

    system_prompt = f"""
[역할]
 당신은 전문적인 영한 번역가입니다. 한국어 대화문과 영어 요약문을 입력받아, 영어 요약문을 해당 한국어 대화문에 맞춰 한국어 요약문으로 번역합니다.

 [목표]
 - 주어진 영어 요약문을 한국어 요약문으로 번역하세요.
 - 한국어 요약문은 입력된 한국어 대화의 내용을 잘 반영해야 합니다.
 - 주어진 Keywords와 Bigrams를 참고하여 한국어 요약문 번역에 활용하세요.
 - 한국어 대화에서 사용된 단어와 문맥을 참고하여 한국어 요약문을 번역하십시오.
 - 모든 특수 토큰(#Person1# ~ #Person7#, #Address#, #CardNumber#, #DateOfBirth#, #Email#, #PassportNumber#, #PhoneNumber#, #SSN#)을 보존해야 합니다.
 - 출력은 반드시 JSON 객체로만 구성되어야 합니다. 다른 텍스트나 주석을 추가하지 마십시오.

 [데이터 세트 정보]
 - 각 샘플은 한국어 Dialogue, 한국어 Keywords, 한국어 Bigrams, 영어 Summary로 구성됩니다.
 - 영어 Summary는 주어진 한국어 Dialogue를 요약한 것입니다.
 - 한국어 Keywords와 Bigrams는 주어진 한국어 Dialogue에서 추출한 것으로, 한국어 Dialogue의 핵심 주제와 중요한 내용을 반영하여 추출되었습니다.

 [지침]
 - 영어 요약문을 한국어로 직접 번역하되, 한국어 대화문과의 맥락을 고려하여 한국어 요약문의 자연스러운 흐름을 유지하십시오.
 - 한국어 Keywords와 Bigrams를 참고하여 한국어 요약문 번역에 활용하세요.
 - 주어진 JSON 형식만을 준수해야 합니다.
 - 문장 구조와 사용된 어휘가 한국어에 자연스럽게 느껴지도록 번역하십시오.
 - 주어진 형식을 벗어나지 마십시오.
 
 """

    system_prompt_content = content_types.ContentDict(
        role="user", 
        parts=[system_prompt]
    )
    history_messages.append(system_prompt_content)

    if ref_fname and ref_kr_dialogue:
        for i in range(min(len(ref_fname), len(ref_kr_dialogue))):
            example_code_content = content_types.ContentDict(
                role="user",
                parts=[f"**Code:**\n{ref_fname[i]}"]
            )
            example_korean_dialogue_content = content_types.ContentDict(
                role="user",
                parts=[f"**Korean Dialogue:**\n{ref_kr_dialogue[i]}"]
            )
            example_korean_keywords_content = content_types.ContentDict(
                role="user",
                parts=[f"**Keywords:**\n{ref_key[i]}"]
            )
            example_korean_bigrams_content = content_types.ContentDict(
                role="user",
                parts=[f"**Bigrams:**\n{ref_bigram[i]}"]
            )
            example_english_summary_content = content_types.ContentDict(
                role="user",
                parts=[f"**English Summary:**\n{ref_en_summary[i]}"]
            )
            example_korean_summary_content = content_types.ContentDict(
                role="model",
                parts=[f"**Korean Summary:**\n{ref_kr_summary[i]}"]
            )
            history_messages.append(example_code_content)
            history_messages.append(example_korean_dialogue_content)
            history_messages.append(example_korean_keywords_content)
            history_messages.append(example_korean_bigrams_content)
            history_messages.append(example_english_summary_content)
            history_messages.append(example_korean_summary_content)

    content_parts = []
    for idx, (test_fname, test_kr_dialogue, test_key, test_bigram, summary) in enumerate(batch, start=1):
        content_parts.append(f"""
**Code{idx}**
{test_fname}
**Korean Dialogue{idx}**
{test_kr_dialogue}
**Keywords{idx}**
{test_key}
**Bigrams{idx}**
{test_bigram}
**English Summary{idx}**
{summary}
"""
        )
    content_parts.append(f"주어진 각 English Summary를 직접적인 번역을 하세요. 각 Korean Dialogue에서 활용된 단어, 문구, 어구 등을 잘 분석한 뒤 번역하세요. Keywords와 Bigrams는 최대한 포함하여 번역하세요.")
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
                # Clean up the response
                cleaned_response = full_response.strip()
                if cleaned_response.startswith("```json"):
                    cleaned_response = cleaned_response[7:-3]  # Remove ```json and ``` markers
                
                # Parse JSON response
                json_response = json.loads(cleaned_response)
                print("Debug - Raw JSON response:", json_response)
                
                # if isinstance(json_response, dict):
                #     print("디버깅 - 번역 요약문:", json_response.get('summary'))

                #     if not json_response.get('summary'):
                #         print("디버깅 - 번역 필드 비어있음")
                        
                results = []
                
                # Handle different response formats
                if isinstance(json_response, dict):
                    # Single response
                    if "korean_summary" in json_response:
                        results.append(json_response["korean_summary"])
                    elif "summary" in json_response:
                        results.append(json_response["summary"])
                    else:
                        # Try to find any key that might contain the summary
                        for key, value in json_response.items():
                            if isinstance(value, str) and len(value) > 10:  # Assume longer strings are summaries
                                results.append(value)
                                break
                elif isinstance(json_response, list):
                    # Multiple responses
                    for item in json_response:
                        if isinstance(item, dict):
                            if "korean_summary" in item:
                                results.append(item["korean_summary"])
                            elif "summary" in item:
                                results.append(item["summary"])
                            else:
                                # Try to find any key that might contain the summary
                                for key, value in item.items():
                                    if isinstance(value, str) and len(value) > 10:
                                        results.append(value)
                                        break
                        elif isinstance(item, str):
                            results.append(item)
                
                # Ensure we have the right number of results
                if len(results) != len(batch):
                    logging.warning(f"배치 크기 불일치: {len(results)} != {len(batch)}")
                    results.extend([""] * (len(batch) - len(results)))
                
                return results
                
            except json.JSONDecodeError as e:
                logging.error(f"JSON 파싱 오류: {str(e)}\n응답: {full_response}")
                return [""] * len(batch)
            
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
            logging.error(f"예기치 않은 오류: {str(e)}")
            if attempt < max_retries - 1:
                wait_time = base_delay * (2 ** attempt) + random.uniform(0, 1)
                print(f"재시도 중... {wait_time:.2f} 초 대기...")
                await asyncio.sleep(wait_time)
            else:
                raise

    return [{"summary": ""} for _ in range(len(batch))]

def save_intermediate_results(results: List[dict], batch_num: int) -> None:
    timestamp = KR_NOW.strftime("%m%d_%H%M%S")
    intermediate_file = os.path.join(LLM_PATH, 'result', 'batch_translate', f"test_eng_sum_batch_{batch_num}.json")
    
    try:

        for result in results:
            if not result.get("summary"):
                logging.warning(f"번역 결과 비어있음: {result}")
                
        with open(intermediate_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        logging.info(f"중간 결과 저장: {intermediate_file}")
    except Exception as e:
        logging.error(f"중간 결과 저장 실패: {str(e)}")

async def translate(output_df: pd.DataFrame, test_original_df: pd.DataFrame, test_keyword_df: pd.DataFrame, ref_diasum_df: pd.DataFrame, ref_key_df: pd.DataFrame, batch_size: int = BATCH_SIZE, num_samples: int = -1) -> List[Dict]:
    num_samples = len(output_df) if num_samples == -1 else min(num_samples, len(output_df))
    
    # Fix the column access based on the actual DataFrame contents
    test_kr_dialogues = test_original_df['korean_dialogue'][:num_samples].tolist()
    test_fnames = test_original_df['fname'][:num_samples].tolist()
    test_keys = test_keyword_df['keywords'][:num_samples].tolist()
    test_bigrams = test_keyword_df['bigrams'][:num_samples].tolist()
    test_summaries = output_df['summary'][:num_samples].tolist()

    summary_results_list = []
    start_time = time.time()
    
    model = gen_model
    
    # Fix the tuple structure to match what's being unpacked later
    pairs = list(zip(test_fnames, test_kr_dialogues, test_keys, test_bigrams, test_summaries))
    batches = [pairs[i:i+batch_size] for i in range(0, len(pairs), batch_size)]
    
    logging.info(f"총 데이터 수: {len(pairs)}")
    logging.info(f"배치 크기: {batch_size}")
    logging.info(f"총 배치 수: {len(batches)}")

    # Load reference data for few-shot examples
    ref_fname, ref_kr_dialogue, ref_en_summary, ref_kr_summary, ref_key, ref_bigram = few_shot_sample(ref_diasum_df, ref_key_df)
    formatted_examples = format_few_shot_examples(ref_fname, ref_kr_dialogue, ref_en_summary, ref_kr_summary, ref_key, ref_bigram)
    
    async def process_batch(batch_idx: int, batch: List[Tuple[str, str, str, str, str]]) -> List[Dict]:
        batch_start_time = time.time()
        batch_results = []
        logging.info(f"배치 {batch_idx+1}/{len(batches)} 처리 시작 (크기: {len(batch)})")
        
        try:
            summaries = []
            for fname, kr_dialogue, key, bigram, summary in batch:  # Properly unpack the 5 values
                summaries.append({
                    "fname": fname,
                    "dialogue": kr_dialogue,
                    "keywords": key,
                    "bigrams": bigram,
                    "summary": summary
                })
            
            results = await gemini_translation_stream(
                batch, 
                model=model,
                formatted_examples=formatted_examples
            )
            
            for idx, (fname, kr_dialogue, key, bigram, summary) in enumerate(batch):
                if idx < len(results):
                    result_dict = {
                        "fname": fname,
                        "english_summary": summary,
                        "dialogue": kr_dialogue,
                        "keywords": key,
                        "bigrams": bigram,
                        "korean_summary": results[idx] if isinstance(results[idx], str) else results[idx].get("summary", "")
                    }
                    batch_results.append(result_dict)
                    
                    print(f"==GEMINI Translation Batch {batch_idx+1}=="*5)
                    print(f"fname: {fname}")
                    print(f"\nEnglish Summary: {summary}")
                    print(f"\nKeywords: {key}")
                    print(f"\nBigrams: {bigram}")
                    print(f"\nKorean Summary: {results[idx]}")
                    print("=="*50)
                    
                    await asyncio.sleep(0.5)
                    
            return batch_results
            
        except Exception as e:
            logging.error(f"배치 {batch_idx+1} 처리 중 오류: {str(e)}")
            return []

    tasks = [
        process_batch(i, batch) for i, batch in enumerate(batches)
    ]
    
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

    summary_results_list = all_results  
    total_time = time.time() - start_time
    print(f"\n총 처리 시간: {total_time:.2f} seconds")

    # Fix the output file path creation
    output_dir = os.path.dirname(os.path.join(BASELINE_PATH, 'output', FILE_NAME))
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(BASELINE_PATH, 'output', FILE_NAME)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(summary_results_list, f, indent=4, ensure_ascii=False)
    print(f"번역 결과 저장: {output_file}")

    return summary_results_list

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
        output_df, test_original_df, test_keyword_df, ref_diasum_df, ref_key_df = load_data(BASELINE_PATH, start_idx=START_IDX, end_idx=END_IDX, fs_num_samples=FS_NUM_SAMPLES)
        logging.info(f"로드된 데이터 크기: {len(output_df)}") 
        await translate(output_df, test_original_df, test_keyword_df, ref_diasum_df, ref_key_df, batch_size=BATCH_SIZE)

    asyncio.run(main())


