import os
import requests
import pandas as pd
import time
import xml.etree.ElementTree as ET
import concurrent.futures
from tqdm import tqdm  # For progress bar


# API 크롤링 정보에 따라 교차해서 사용 
# BASE_URL = "http://apis.data.go.kr/1613000/AptBasisInfoServiceV2/getAphusDtlInfoV2"  ## 상세정보
BASE_URL = "http://apis.data.go.kr/1613000/AptBasisInfoServiceV2/getAphusBassInfoV2"  ## 기본정보
API_KEY = "CEMM2yAd83UCUH2bbZJbSayuEfm4GBBWTol1yugmgEpi95judA6fLkSY+h8uCWKSKBIBUKCbHFckZD7zG7f3xg=="

# 데이터 경로 설정
BASE_DIR = '/data/ephemeral/home'
KAPTCODE_FILE = os.path.join(BASE_DIR, '')
OUTPUT_PATH = os.path.join(BASE_DIR, '')

# 일전에 크롤링한 코드 리스트 읽기
def read_kaptcode():
    try:
        df = pd.read_csv(KAPTCODE_FILE)
        return df['kaptCode'].astype(str).str.strip().tolist()
    except Exception as e:
        print(f"코드 읽기 실패: {e}")
        return []

# 상세정보 크롤링
def fetch_apt_details(kaptCode):
   try:
        params = {
            'serviceKey': API_KEY,
            'kaptCode': kaptCode
        }
        
        response = requests.get(
            BASE_URL,
            params=params,
            timeout=10
        )
        
        if response.status_code == 200:
            root = ET.fromstring(response.text)
            item = root.find('.//item')

            if item is not None:
                return {
                    'kaptCode': kaptCode,
                    'kaptName': item.findtext('kaptName', ''),
                    'kaptPcnt': item.findtext('kaptPcnt', ''),
                    'kaptPcntu': item.findtext('kaptPcntu', ''),
                    'subwayLine': item.findtext('subwayLine', ''),
                    'subwayStation': item.findtext('subwayStation', ''),
                    'kaptdWtimesub': item.findtext('kaptdWtimesub', ''),
                    'kaptdWtimebus': item.findtext('kaptdWtimebus', ''),
                    'educationFacility': item.findtext('educationFacility', '')
                }   
        return None
        
   except Exception as e:
        print(f"데이터 조회 실패: {str(e)}")
        return None

# 결과 저장
def save_results(results):
    try:
        df = pd.DataFrame(results)
        df.to_csv(OUTPUT_PATH, index=False, encoding='utf-8-sig')
        print(f"{len(results)} 결과 {OUTPUT_PATH} 경로에 저장")
    except Exception as e:
        print(f"결과 저장 실패: {e}")

# 배치로 처리하면 더 빠르게 처리 가능
def process_batch(kaptcodes_batch):
    results = []
    for kaptcode in kaptcodes_batch:
        details = fetch_apt_details(kaptcode)
        if details:
            results.append(details)
        time.sleep(0.5)  # Reduced sleep time
    return results

def main():
    print("데이터 수집 시작")
    
    kaptcodes = read_kaptcode()
    if not kaptcodes:
        print("코드 못읽음")
        return

    batch_size = 100        # 배치 사이즈
    num_workers = 4         # 병렬 처리 스레드 수
    all_results = []
    
    # 배치 생성
    batches = [kaptcodes[i:i + batch_size] for i in range(0, len(kaptcodes), batch_size)]

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = list(tqdm(executor.map(process_batch, batches),
            total=len(batches),
            desc="Processing batches"
        ))
        
        for batch_result in futures:
            all_results.extend(batch_result)
    
    if all_results:
        save_results(all_results)
        print(f"처리 완료 {len(all_results)} 레코드")

if __name__ == "__main__":
    main()