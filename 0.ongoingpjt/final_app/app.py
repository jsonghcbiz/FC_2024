import streamlit as st
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from library.s3 import s3_client

st.set_page_config(page_title="영화 리뷰 감성 분석", layout="wide")

from menu.home import home_view
from menu.text_analysis import text_analysis_view
from menu.movie_review_analysis import session_initialize, streamlit_movie_search, analyze_sentiment
from menu.image_generator import main
from menu.character_rec import main as character_rec_main

st.markdown(
    """
    <style>
        /* Center align content */
        [data-testid="stMainBlockContainer"] {
            width: 80%;
            margin: auto;
        }
        #fce8dc1b {
            text-align: center;
        }
        [data-testid="stMainBlockContainer"] {
            width: 80%;
            margin: auto;
        }
        .stHorizontalBlock st-emotion-cache-ocqkz7 e1f1d6gn5 {
            display: flex;
            width: 100%;
        }
        .stColumn st-emotion-cache-1r6slb0 e1f1d6gn3 {
            flex: 1;
        }
        /* 탭 레이블의 글씨 크기 조절 */
        [data-baseweb="tab"] [data-testid="stMarkdownContainer"] p {
            font-size: 15px !important;
        }
        /*header 크기 조절*/
        [data-testid="stHeader"] {
            text-align: center;
            font-size: 30px !important;
        }
        /*제목 크기 조절*/
        [data-testid="stTitle"] {
            font-size: 20px !important;
            text-align: center;
        }
        /* Example adjustment for columns */
        [data-testid="stHorizontalBlock"] {
            display: flex;
            width: 100%;
        }
        /* 컨테이너 스타일 조절 */
        .custom-container {
            background-color: lightblue;
            border: 4px solid black;
            padding: 10px;
            border-radius: 5px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)
# 데이터 준비
images = s3_client(st.secrets['aws']['access_key'], st.secrets['aws']['secret_access_key']).get_images()
detail_images = [
    "https://fastcampus-ml-p2-bucket.s3.ap-northeast-2.amazonaws.com/image/detail_image1.avif"
]
titles = [
    "Moana2",
    "wicked",
    "Gladiator2"
]
buttons = []


# 메뉴 옵션 정의
menu_options = ["🏠 홈", "📝 텍스트 분석", "🎞️ 영화 리뷰 분석", "🎨 AI 이미지", "👑 캐릭터 추천기"]
tab1, tab2, tab3, tab4, tab5 = st.tabs(menu_options)

# 현재 선택된 메뉴에 따라 콘텐츠 표시
with tab1:
    st.title("🎥 2조 오뚝이: 영화 리뷰 감성 분석")
    st.markdown("영화 리뷰를 크롤링하고 감성 분석을 수행하는 페이지입니다!")
    home_view(titles, images, buttons, detail_images)
with tab2:
    st.title("📝 텍스트 분석")
    st.markdown(
    """
    <hr style="border: none; height: 2px; background: linear-gradient(to right, #ff7e5f, #ffd700); margin: 20px 0;">
    """,
    unsafe_allow_html=True
)
    text_analysis_view()
with tab3:
    st.title("🎞️ 영화 리뷰 크롤링 및 감성 분석")
    st.markdown(
    """
    <hr style="border: none; height: 2px; background: linear-gradient(to right, #ff7e5f, #ffd700); margin: 20px 0;">
    """,
    unsafe_allow_html=True
)
#     st.markdown(
#     """
#     <hr style="border: none; height: 2px; background: #c0c0c0; margin: 20px 0;">
#     """,
#     unsafe_allow_html=True
# )
    streamlit_movie_search()
with tab4:
    st.title("🎨 AI 이미지")
    st.markdown(
    """
    <hr style="border: none; height: 2px; background: linear-gradient(to right, #ff7e5f, #ffd700); margin: 20px 0;">
    """,
    unsafe_allow_html=True
)
    main()
with tab5:
    st.title("Wicked 캐릭터 추천기")
    st.markdown(
    """
    <hr style="border: none; height: 2px; background: linear-gradient(to right, #ff7e5f, #ffd700); margin: 20px 0;">
    """,
    unsafe_allow_html=True
)
    character_rec_main()