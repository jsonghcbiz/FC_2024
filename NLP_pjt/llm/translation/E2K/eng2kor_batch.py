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
LOG_FILE = os.path.join(LLM_PATH, 'result', 'batch_translate', 'translation_batch.log')
ENV_PATH = os.path.join(LLM_PATH, '.env')
load_dotenv(ENV_PATH)



### ✅ 시간 세팅 ✅
KR_TIMEZONE = pytz.timezone('Asia/Seoul')
KR_NOW = datetime.now(KR_TIMEZONE)

### ✅ 로드할 데이터 범위 세팅 ✅    현재까지 0~3000까지 처리됨.
START_IDX = 0
END_IDX = 499

### ✅ 배치 사이즈 세팅 ✅
BATCH_SIZE = 25     # ⚠️⚠️⚠️ output token 초과 시 데이터 추출이 제대로 안됨. 시간이 걸리더라도 배치 사이즈를 줄여야 함.

### ✅ API 리밋 초과 시 재시도 세팅 ✅
MAX_RETRIES = 5
BASE_DELAY = 15

### ✅ 파일 이름 세팅 ✅
DATA_NAME = 'bart_large_cnn_v2_eng.csv'         # inference 요약문 데이터
ORIGINAL_NAME = 'test.csv'                      # 원본 한글 대화문 
REF_DIASUM_NAME = 'train_dh_v3_eng.csv'            # 학습 대화문, 요약문 데이터
REF_KEY_NAME = 'train_dh_v3_kor_key.csv'      # 학습 대화문, 요약문, 키워드 데이터
TEST_KEYWORD_NAME = 'test_kor_key.csv'          # 테스트 한글 키워드 데이터
FILE_NAME = f'test_output_key.json'          # 저장 파일 이름

### ✅ Few-shot 예시 샘플링 ✅
FS_NUM_SAMPLES = 50

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
def load_data(file_path=BASELINE_PATH):
    output_df = pd.read_csv(os.path.join(file_path, 'output', DATA_NAME))
    test_keyword_df = pd.read_csv(os.path.join(file_path, 'keywords', TEST_KEYWORD_NAME))
    ref_diasum_df = pd.read_csv(os.path.join(file_path, 'original', REF_DIASUM_NAME))
    ref_key_df = pd.read_csv(os.path.join(file_path, 'keywords', REF_KEY_NAME))
    test_original_df = pd.read_csv(os.path.join(file_path, 'original', ORIGINAL_NAME))
    return output_df, test_keyword_df, ref_diasum_df, ref_key_df, test_original_df


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
    ref_key = ref_key_df['key'].tolist()
    ref_bigram = ref_key_df['bigram'].tolist()
    
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
1) I will analyse the English summary and the Korean dialogue to identify common keywords and bigrams that aligns with the context of both English summary and Korean dialogue.
2) I will translate the English summary to Korean, ensuring it matches the context of the Korean dialogue provided. 
3) I will keep the original meaning and also preserve the special tokens.

model: **Korean Summary**
{ref_kr_summary[i]}"""
        formatted_examples.append(example)
    
    return "\n".join(formatted_examples)



###############################################
# 모델 호출 함수 (gemini_translation_stream)
###############################################
async def gemini_translation_stream(batch: List[Tuple[str, str, str]], model, formatted_examples:str, exp_fname=None, exp_kr_dialogue=None, exp_en_summary=None, exp_kr_summary=None) -> List[Tuple[str]]:
    history_messages = []

    system_prompt = f"""
[역할]
 당신은 전문적인 영한 번역가입니다. 한국어 대화문과 영어 요약문을 입력받아, 영어 요약문을 해당 한국어 대화문에 맞춰 한국어 요약문으로 번역합니다.

 [목표]
 - 주어진 영어 요약문을 한국어 요약문으로 번역하세요.
 - 한국어 요약문은 입력된 한국어 대화의 내용을 잘 반영해야 합니다.
 - 한국어 대화에서 사용된 단어와 문맥을 참고하여 한국어 요약문을 번역하십시오.
 - 모든 특수 토큰(#Person1# ~ #Person7#, #Address#, #CardNumber#, #DateOfBirth#, #Email#, #PassportNumber#, #PhoneNumber#, #SSN#)을 보존해야 합니다.
 - 출력은 반드시 JSON 객체로만 구성되어야 합니다. 다른 텍스트나 주석을 추가하지 마십시오.

 [데이터 세트 정보]
 - 각 샘플은 한국어 대화문과 영어 요약문으로 구성됩니다.
 - 영어 요약문은 주어진 한국어 대화문을 요약한 것입니다.

 [지침]
 - 영어 요약문을 한국어로 직접 번역하되, 한국어 대화문과의 맥락을 고려하여 한국어 요약문의 자연스러운 흐름을 유지하십시오.
 - 주어진 JSON 형식만을 준수해야 합니다.
 - 문장 구조와 사용된 어휘가 한국어에 자연스럽게 느껴지도록 번역하십시오.
 - 주어진 형식을 벗어나지 마십시오.
 
 [예시]
    {formatted_examples}
 """

    system_prompt_content = content_types.ContentDict(
        role="user", 
        parts=[system_prompt]
    )
    history_messages.append(system_prompt_content)

    if exp_fname and exp_kr_dialogue:
        for i in range(min(len(exp_fname), len(exp_kr_dialogue))):
            example_code_content = content_types.ContentDict(
                role="user",
                parts=[f"**Code:**\n{exp_fname[i]}"]
            )
            example_korean_dialogue_content = content_types.ContentDict(
                role="user",
                parts=[f"**Korean Dialogue:**\n{exp_kr_dialogue[i]}"]
            )
            example_english_summary_content = content_types.ContentDict(
                role="user",
                parts=[f"**English Summary:**\n{exp_en_summary[i]}"]
            )
            example_korean_summary_content = content_types.ContentDict(
                role="model",
                parts=[f"**Korean Summary:**\n{exp_kr_summary[i]}"]
            )
            history_messages.append(example_code_content)
            history_messages.append(example_korean_dialogue_content)
            history_messages.append(example_english_summary_content)
            history_messages.append(example_korean_summary_content)


    content_parts = []
    for idx, (original_fname, original_kr_dialogue, summary) in enumerate(batch, start=1):
        content_parts.append(f"""
**Code**
{original_fname}
**Korean Dialogue**
{original_kr_dialogue}
**English Summary**
{summary}
"""
        )
    content_parts.append(f"주어진 각 English Summary를 직접적인 번역을 하세요. 각 Korean Dialogue에서 활용된 단어, 문구, 어구 등을 잘 분석하여 번역에 활용하세요.")
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
                    print("디버깅 - 번역 요약문:", json_response.get('summary'))

                    if not json_response.get('summary'):
                        print("디버깅 - 번역 필드 비어있음")
                        
                results = []

                if isinstance(json_response, dict):
                    json_response = [json_response]
                
                for item  in json_response:
                    for key, value in item.items():
                        summary = value.get("Korean Summary", "").strip()
                        if not summary:
                            logging.warning(f"JSON 응답 {item}에서 'Korean Summary' 없음")
                            results.append("")
                            continue
                        results.append((summary))
                if len(results) != len(batch):
                    logging.warning(f"배치 크기 불일치: {len(results)} != {len(batch)}")
                    results.extend([""] * (len(batch) - len(results)))
                return results
            except json.JSONDecodeError:
                logging.error(f"json 응답 형식 불일치: {full_response}")
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
            print(f"예기치 않은 오류: {str(e)}")
            if attempt < max_retries - 1:
                wait_time = base_delay * (2 ** attempt) + random.uniform(0, 1)
                print(f"재시도 중... {wait_time:.2f} 초 대기...")
                await asyncio.sleep(wait_time)
            else:
                raise

    return [""] * len(batch)

def save_intermediate_results(results: List[dict], batch_num: int) -> None:
    timestamp = KR_NOW.strftime("%m%d_%H%M%S")
    intermediate_file = os.path.join(RESULT_PATH, f"test_eng_{START_IDX}_{END_IDX}_batch_{batch_num}.json")
    
    try:

        for result in results:
            if not result.get("summary"):
                logging.warning(f"번역 결과 비어있음: {result}")
                
        with open(intermediate_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        logging.info(f"중간 결과 저장: {intermediate_file}")
    except Exception as e:
        logging.error(f"중간 결과 저장 실패: {str(e)}")

async def translate(output_df: pd.DataFrame, original_df: pd.DataFrame, reference_df: pd.DataFrame, batch_size: int = BATCH_SIZE, num_samples: int = -1) -> List[Dict]:
    num_samples = len(output_df) if num_samples == -1 else min(num_samples, len(output_df))

    original_fnames = original_df['fname'][:num_samples].tolist()
    original_kr_dialogues = original_df['dialogue'][:num_samples].tolist()
    english_summaries = output_df['summary'][:num_samples].tolist()



    translate_results_list = []
    start_time = time.time()
    
    model = gen_model
    
    pairs = list(zip(original_fnames, original_kr_dialogues, english_summaries))
    batches = [pairs[i:i+batch_size] for i in range(0, len(pairs), batch_size)]
    
    logging.info(f"총 데이터 수: {len(pairs)}")
    logging.info(f"배치 크기: {batch_size}")
    logging.info(f"총 배치 수: {len(batches)}")

    # Load reference data for few-shot examples
    original_df, reference_df = load_reference_data(RESULT_PATH)
    original_fname, original_kr_dialogue, exp_fname, exp_kr_dialogue, exp_en_summary, exp_kr_summary = few_shot_sample(original_df, reference_df)
    formatted_examples = format_few_shot_examples(original_fname, original_kr_dialogue, exp_fname, exp_kr_dialogue, exp_en_summary, exp_kr_summary)
    
    async def process_batch(batch_idx: int, batch: List[Tuple[str, str, str]]) -> List[Dict]:
        batch_start_time = time.time()
        batch_results = []
        logging.info(f"배치 {batch_idx+1}/{len(batches)} 처리 시작 (크기: {len(batch)})")
        
        try:
            results = await gemini_translation_stream(
                batch, 
                model = model,
                formatted_examples = formatted_examples
            )
            
            for idx, (original_fname, original_kr_dialogue, english_summary) in tqdm(enumerate(batch), total=len(batch), desc=f"배치 {batch_idx+1} 처리 중...", leave=False):
                korean_summary = results[idx]
                if idx < len(results):
                    result_dict = {
                        "fname": original_fname,
                        "english_summary": english_summary,
                        "korean_summary": korean_summary
                    }
                    batch_results.append(result_dict)

                    print(f"==GEMINI Translation Batch {batch_idx+1}=="*5)
                    print(f"fname: {original_fname}")
                    print(f"\nEnglish Summary: {english_summary}")
                    print(f"\nKorean Summary: {korean_summary}")
                    print("=="*50)

                    await asyncio.sleep(0.5)
                    if (batch_idx + 1) % 10 == 0:  
                        await asyncio.sleep(1)

                    if (batch_idx+1) % 10 == 0 and idx % 10 == 0: 
                        logging.info(f"배치 {batch_idx+1} - {idx}/{len(batch)} 항목 처리 완료")
                else:
                    logging.error(f"배치 {batch_idx+1} - {idx}번째 항목 처리 중 오류 발생. 결과 없음")

        except Exception as e:
            logging.error(f"배치 {batch_idx+1} 처리 중 오류: {str(e)}")
        
        batch_end_time = time.time()
        batch_duration = batch_end_time - batch_start_time
        logging.info(f"배치 {batch_idx+1} 완료. 소요시간: {batch_duration:.2f}초")
        
        return batch_results

    tasks = [
        process_batch(
            i, 
            batch
        ) for i, batch in enumerate(batches)
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
        output_df = load_data(OUTPUT_PATH, start_idx=START_IDX, end_idx=END_IDX)
        original_df, reference_df = load_reference_data(RESULT_PATH)
        logging.info(f"로드된 데이터 크기: {len(output_df)}") 
        await translate(output_df, original_df, reference_df, batch_size=BATCH_SIZE)

    asyncio.run(main())


