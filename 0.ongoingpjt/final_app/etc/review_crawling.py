from selenium import webdriver
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException
import time
import pandas as pd
import os

# 기본 설정 (CGV와 메가박스 각각 검색에 추가하고 싶은 영화 코드를 입력하세요)
CGV_MOVIE_CODES = {
    "moana2": "88381",
    "wicked": "88076",
    "1승": "89075",
    "gladiator2": "88459"
}
CGV_URL = "http://www.cgv.co.kr/movies/detail-view/?midx="
CGV_OBJECT_NAME = "cgv_reviews.csv"
CGV_DATA_PATH = os.path.join("data", CGV_OBJECT_NAME)

MEGABOX_MOVIE_CODES = {
    "moana2": "24036800",
    "wicked": "24010200",
    "1승": "24073500",
    "gladiator2": "24043900"
}
MEGABOX_URL = "https://www.megabox.co.kr/movie-detail/comment?rpstMovieNo="
MEGABOX_OBJECT_NAME = "megabox_reviews.csv"
MEGABOX_DATA_PATH = os.path.join("data", MEGABOX_OBJECT_NAME)


# CGV 리뷰 크롤링 함수
def get_movie_reviews_on_cgv(url, review_limit=60):
    # CGV는 한 페이지당 6개의 리뷰를 보여줍니다.
    # args: url, page_num, review_limit
    # review_limit는 크롤링할 수 있는 최대 리뷰 수. 맨 아래 main 함수에서 설정합니다.
    # data: pandas.DataFrame으로 리뷰를 반환합니다.

    wd = webdriver.Chrome()
    wd.get(url)
    review_list = []
    page_no = 1                 # 페이지 번호 초기화
    
    while len(review_list) < review_limit:                                        # 리뷰 리스트의 길이가 최대 리뷰 수보다 작으면 반복
        try:
            page_ul = wd.find_element(By.ID, 'paging_point')                      # 페이지 번호 요소 찾기. paging_point는 페이지 번호를 보여주는 요소
            page_a = page_ul.find_element(By.LINK_TEXT, str(page_no))             # 페이지 번호 요소 찾기. link_text는 텍스트를 찾아줌
            page_a.click()                                                        # 페이지 번호 클릭
            time.sleep(2)                                                         # 웹 페이지 로딩 대기

            reviews = wd.find_elements(By.CLASS_NAME, 'box-comment')              # 리뷰 요소 찾기. box-comment는 리뷰를 보여주는 요소
            new_reviews = [review.text for review in reviews]                     # 리뷰 텍스트 추출
            review_list += new_reviews[:review_limit - len(review_list)]          # 리뷰 리스트에 추가

            if len(new_reviews) == 0:                                               # 리뷰가 없으면 중단
                break

            if page_no % 10 == 0:                                                   # 페이지 번호가 10의 배수이면 다음 페이지로 이동        
                next_button = page_ul.find_element(By.XPATH, './/button[contains(@class, "btn-paging next")]')
                next_button.click()
                time.sleep(2)
            page_no += 1                                                            # 페이지 번호 증가

        except NoSuchElementException as e:
            print("더이상 로드할 페이지가 없습니다.")
            print(e)
            break
    
    movie_review_df = pd.DataFrame({"review": review_list})                      # 리뷰 리스트를 DataFrame으로 변환
    wd.close()                                                                   # 웹 드라이버 닫기
    return movie_review_df                                                       # 리뷰 리스트를 DataFrame으로 반환  

# 메가박스 리뷰 크롤링 함수
def get_movie_reviews_on_megabox(url, review_limit=30):
    # 메가박스는 한 페이지당 10개의 리뷰를 보여줍니다.
    # args: url, page_num, review_limit
    # review_limit는 크롤링할 수 있는 최대 리뷰 수. 맨 아래 main 함수에서 설정합니다.
    # data: pandas.DataFrame으로 리뷰를 반환합니다.

    wd = webdriver.Chrome()
    wd.get(url)
    review_list = []
    page_no = 1                                                                   # 페이지 번호 초기화

    while len(review_list) < review_limit:                                         # 리뷰 리스트의 길이가 최대 리뷰 수보다 작으면 반복
        try:
            if page_no % 10 != 1:                                                  # 페이지 번호가 1의 배수가 아니면 다음 페이지로 이동
                page_nav = wd.find_element(By.CLASS_NAME, 'pagination')            # 페이지 번호 요소 찾기. pagination은 페이지 번호를 보여주는 요소
                page_a = page_nav.find_element(By.LINK_TEXT, str(page_no))         # 페이지 번호 요소 찾기. link_text는 텍스트를 찾아줌
                page_a.click()                                                     # 페이지 번호 클릭
                time.sleep(2)                                                      # 웹 페이지 로딩 대기

            reviews = wd.find_elements(By.CLASS_NAME, 'story-txt')                  # 리뷰 요소 찾기. story-txt는 리뷰를 보여주는 요소
            new_reviews = [review.text for review in reviews]                       # 리뷰 텍스트 추출
            review_list += new_reviews[:review_limit - len(review_list)]            # 리뷰 리스트에 추가

            if len(new_reviews) == 0:                                               # 리뷰가 없으면 중단
                break

            if page_no % 10 == 0:                                                   # 페이지 번호가 10의 배수이면 다음 페이지로 이동
                page_nav = wd.find_element(By.CLASS_NAME, 'pagination')             # 페이지 번호 요소 찾기. pagination은 페이지 번호를 보여주는 요소
                next_button = page_nav.find_element(By.XPATH, './/a[contains(@class, "control next")]') # 다음 페이지 버튼 요소 찾기. control next는 다음 페이지 버튼을 보여주는 요소
                next_button.click()                                                                     # 다음 페이지 버튼 클릭
                time.sleep(2)                                                                           # 웹 페이지 로딩 대기
            page_no += 1                                                                                # 페이지 번호 증가

        except NoSuchElementException as e:
            print("더이상 로드할 페이지가 없습니다.")
            print(e)
            break
    
    movie_review_df = pd.DataFrame({"review": review_list[:review_limit]})
    wd.close()
    return movie_review_df

def ensure_data_directory():
    # 데이터 디렉토리가 없으면 생성합니다.
    os.makedirs("data", exist_ok=True)

def main():
    # 데이터 디렉토리 생성
    ensure_data_directory()
    
    # CGV 리뷰 크롤링
    cgv_url = CGV_URL + CGV_MOVIE_CODES["moana2"]                                           # CGV 영화 페이지 URL 설정
    cgv_reviews = get_movie_reviews_on_cgv(url=cgv_url, review_limit=60)                    # CGV 리뷰 크롤링. 리뷰 수는 60개로 설정
    cgv_reviews.to_csv(CGV_DATA_PATH, sep=",", index=False, encoding="utf-8")               # CGV 리뷰를 CSV 파일로 저장
    print(f"CGV 리뷰를 {CGV_DATA_PATH}에 저장했습니다.")
    
    # 메가박스 리뷰 크롤링
    megabox_url = MEGABOX_URL + MEGABOX_MOVIE_CODES["moana2"]                             # 메가박스 영화 페이지 URL 설정
    megabox_reviews = get_movie_reviews_on_megabox(url=megabox_url, review_limit=30)      # 메가박스 리뷰 크롤링. 리뷰 수는 30개로 설정
    megabox_reviews.to_csv(MEGABOX_DATA_PATH, sep=",", index=False, encoding="utf-8")     # 메가박스 리뷰를 CSV 파일로 저장
    print(f"메가박스 리뷰를 {MEGABOX_DATA_PATH}에 저장했습니다.")

if __name__ == "__main__":
    main() 