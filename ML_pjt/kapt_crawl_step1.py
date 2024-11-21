import os
import requests
import pandas as pd
import time
import xml.etree.ElementTree as ET
from urllib.parse import quote
from utils import setup_logging, get_api_key, get_text

# API 크롤링 정보
BASE_URL = "http://apis.data.go.kr/1613000/AptListService2/getTotalAptList"
API_KEY = "CEMM2yAd83UCUH2bbZJbSayuEfm4GBBWTol1yugmgEpi95judA6fLkSY+h8uCWKSKBIBUKCbHFckZD7zG7f3xg=="

# 경로 설정
BASE_DIR = '/data/ephemeral/home'
DOWNLOAD_PATH = os.path.join(BASE_DIR, '')
FILENAME = "apt_kaptcode_list.csv"

# 크롤링 설정
NUM_OF_ROWS = 100              # 한번에 가져올 데이터 수
DELAY_BETWEEN_REQUESTS = 1     # 요청 사이 딜레이
MAX_RETRIES = 3                # 재시도 횟수

# 응답 필드 (api에 요청 할 필드)
response_fields = [
    'kaptCode',      # 단지코드
    'kaptName',      # 단지명
    'bjdCode'        # 법정동코드
]

# 총 데이터 수 조회
def get_total_count():
    params = {
        'numOfRows': NUM_OF_ROWS, 
        'pageNo': 1,
        'ServiceKey': API_KEY
    }
    
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(BASE_URL, params=params, verify=False)
            root = ET.fromstring(response.text)
            total_count = root.find('.//totalCount')
            
            if total_count is not None:
                return int(total_count.text)
            return 0
        except:
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)
                continue
            return 0

# 텍스트 추출
def get_text(item, tag):
    element = item.find(tag)
    return element.text if element is not None else ''

# 아파트 리스트 조회
def search_apt_list(page_no):
    params = {
        'numOfRows': NUM_OF_ROWS,
        'pageNo': page_no,
        'ServiceKey': API_KEY
    }

    try:
        response = requests.get(BASE_URL, params=params, timeout=10)
        root = ET.fromstring(response.text)
        items = root.findall('.//item')
        apt_list = []

        for item in items:
            apt_data = {
                'kaptCode': get_text(item, 'kaptCode'),
                'kaptName': get_text(item, 'kaptName'),
                'bjdCode': get_text(item, 'bjdCode'),
                'address': f"{get_text(item, 'as1')} {get_text(item, 'as2')} {get_text(item, 'as3')}".strip()
            }
            apt_list.append(apt_data)
            
        return apt_list
    except:
        return []

# 아파트 리스트 csv 파일로 저장
def save_to_csv(apt_list, filename=FILENAME, mode='w'):
    if not apt_list:
        return False
            
    output_path = os.path.join(DOWNLOAD_PATH, filename)
    df = pd.DataFrame(apt_list)
    
    columns = ['kaptCode', 'kaptName', 'bjdCode', 'address']
    df = df[columns]
    
    if mode == 'w':
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
    else:
        df.to_csv(output_path, mode='a', header=False, index=False, encoding='utf-8-sig')
    return True

# 메인 함수
def main():
    print("아파트 리스트 조회 시작")
    
    # 다운로드 폴더 생성
    os.makedirs(DOWNLOAD_PATH, exist_ok=True)
    
    # 총 데이터 수 조회
    total_count = get_total_count()
    if total_count == 0:
        print("총 데이터 수 조회 실패")
        return
    
    print(f"총 데이터 수: {total_count}")
    
    # 총 페이지 수 계산
    total_pages = (total_count + NUM_OF_ROWS - 1) // NUM_OF_ROWS
    print(f"총 페이지 수: {total_pages}")
    
    # 모든 페이지 처리
    for page in range(1, total_pages + 1):
        # 현재 페이지 처리
        apt_list = search_apt_list(page)
        
        if apt_list:
            # csv 파일 저장 (첫 페이지는 새로운 파일 생성, 이후 페이지는 append)
            mode = 'w' if page == 1 else 'a'
            save_to_csv(apt_list, mode=mode)
            
            # 요청 사이 딜레이
            if page < total_pages:
                time.sleep(DELAY_BETWEEN_REQUESTS)
        else:
            print(f"페이지 {page} 데이터 없음")
    
    print("크롤링 완료")

if __name__ == "__main__":
    main()