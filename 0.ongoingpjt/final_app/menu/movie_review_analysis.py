import streamlit as st
import pandas as pd
import os
import time
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, DataReturnMode
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt

from library.kobert_classifier1 import predict_sentiment
from library.cgv_crawler1 import CGVTitleCrawler, CGVDetailCrawler
from library.megabox_crawler1 import MegaboxTitleCrawler, MegaboxDetailCrawler
from library.prompt_cgv import generate_cgv_prompt, cgv_summarize_reviews
from library.prompt_megabox import generate_megabox_prompt, megabox_summarize_reviews

FONT_PATH = 'AppleGothic' 

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
            font-size: 30px !important;
        }
        /*header 크기 조절*/
        [data-testid="stHeader"] {
            text-align: center;
            font-size: 30px !important;
        }
        /*제목 크기 조절*/
        [data-testid="stTitle"] {
            font-size: 30px !important;
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
            border: 2px solid black;
            padding: 10px;
            border-radius: 5px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

def session_initialize():
    def ensure_session_state_initialized():
        required_keys = {
            'cgv_movie_titles': [],
            'megabox_movie_titles': [],
            'cgv_selected_movie': None,
            'megabox_selected_movie': None,
            'cgv_movie_info': [],
            'megabox_movie_info': [],
            'cgv_reviews': None,
            'megabox_reviews': None,
            'show_megabox_sentiment': False,
            'show_cgv_sentiment': False,
            'selected_movie': None,
            'cgv_movie_data': None,
            'megabox_movie_data': None
        }
        for key, default_value in required_keys.items():
            if key not in st.session_state:
                st.session_state[key] = default_value

    ensure_session_state_initialized()

def analyze_sentiment(reviews_df, model_path="kobert_konply"):
    try:
        reviews_df["sentiment_analysis"] = reviews_df["review"].apply(
            lambda x: predict_sentiment(x, model_path)
        )
        reviews_df["sentiment"] = reviews_df["sentiment_analysis"].apply(lambda x: x["sentiment"])
        reviews_df["confidence"] = reviews_df["sentiment_analysis"].apply(lambda x: x["confidence"])
    except Exception as e:
        st.error(f"감성 분석 중 오류 발생: {e}")
    return reviews_df

def create_wordcloud(text_data):
    with st.container(border=True):
        try:
            wordcloud = WordCloud(
                font_path=FONT_PATH,
                width=800,
                height=400,
                background_color='white',
                max_words=100
            ).generate(' '.join(text_data))
            
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            st.pyplot(plt)
        except Exception as e:
            st.error(f"워드클라우드 생성 중 오류 발생: {str(e)}")

def streamlit_movie_search():
    session_initialize()
    st.markdown("""
#### 영화 리뷰 감성 분석 서비스 🎬
실시간으로 수집된 영화 리뷰들의 감성을 AI가 분석하여 보여드립니다.
긍정/부정 리뷰의 비율과 주요 키워드를 한눈에 확인하세요!
    """)
    with st.container(border=True, key="cgv_container"):
        st.header("🔍 CGV 리뷰 크롤링")
        with st.expander("크롤링 및 검색"):
            cgv_search_query = st.text_input("검색할 영화 제목을 입력하세요:", key="cgv_search_query")
            if st.button("CGV 검색", key="cgv_search_button"):
                if cgv_search_query:
                    with st.spinner('CGV 검색중...'):
                        cgv = CGVTitleCrawler()
                        cgv_movie_titles, cgv_movie_info = cgv.fetch_movie_titles(cgv_search_query)
                        if cgv_movie_titles:
                            st.session_state.cgv_movie_titles = cgv_movie_titles
                            st.session_state.cgv_movie_info = cgv_movie_info
                            st.session_state.selected_movie = cgv_movie_titles
                        else:
                            st.warning("CGV에서 영화를 찾을 수 없습니다.")
            st.session_state.cgv_selectbox = st.selectbox(
                "CGV 검색 결과:", 
                st.session_state.get("cgv_movie_titles", []),
                format_func=lambda x: f"{x}".replace("()", ""), 
                key="cgv_movie_selectbox",
            )
            st.session_state.cgv_selected_movie = st.session_state.cgv_selectbox
            cgv_review_limit = st.number_input("크롤링할 CGV 리뷰 수", min_value=1, value=5, key="cgv_review_limit")
            if st.button("영화 정보 가져오기", key="cgv_info_button"):
                with st.spinner("영화 정보 가져오는 중... 잠시만 기다려주세요."):
                    cgv_detail = CGVDetailCrawler()
                    cgv_movie_data = cgv_detail.crawl_movie_details(st.session_state.cgv_selected_movie, review_limit=cgv_review_limit) 
                    if cgv_movie_data:
                        st.session_state.cgv_movie_data = cgv_movie_data

                        st.write('### 영화 정보')
                        st.write(f'**제목:** {cgv_movie_data["Title"]}')
                        st.write(f'**감독:** {cgv_movie_data["Directors"]}')
                        st.write(f'**출연:** {cgv_movie_data["Cast"]}')
                        st.write(f'**줄거리:** {cgv_movie_data["Plot"]}')

                        st.dataframe(cgv_movie_data["Reviews"])
                        st.session_state.cgv_reviews = cgv_movie_data["Reviews"]
                        st.session_state.show_cgv_sentiment = True
                    else:
                        st.warning("영화 정보를 찾을 수 없습니다.") 

            if st.session_state.cgv_reviews is not None and st.session_state.show_cgv_sentiment:
                st.markdown("### CGV 감성 분석")    
                if st.button("CGV 리뷰 감성 분석", key="cgv_sentiment_button"):
                    with st.spinner("감성 분석 중... 잠시만 기다려주세요."):
                        progress_bar = st.progress(0)
                        total_reviews = len(st.session_state.cgv_reviews)
                        
                        analyzed_reviews = []
                        for idx, review in st.session_state.cgv_reviews.iterrows():
                            sentiment_result = predict_sentiment(review['review'], "kobert_konply")
                            analyzed_reviews.append(sentiment_result)
                            progress_bar.progress((idx + 1) / total_reviews)
                        
                        st.session_state.cgv_reviews["sentiment_analysis"] = analyzed_reviews
                        st.session_state.cgv_reviews["sentiment"] = st.session_state.cgv_reviews["sentiment_analysis"].apply(lambda x: x["sentiment"])
                        st.session_state.cgv_reviews["confidence"] = st.session_state.cgv_reviews["sentiment_analysis"].apply(
                            lambda x: float(x["confidence"].strip('%')) / 100
                        )
                        
                        st.success("감성 분석 완료!")
                        st.subheader("감성 분석 결과")
                        st.balloons()
                        
                        st.dataframe(st.session_state.cgv_reviews[["review", "sentiment", "confidence"]], use_container_width=True)
                        
                        # Add sentiment analysis details
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("### 분석 결과")
                            positive_count = len(st.session_state.cgv_reviews[st.session_state.cgv_reviews['sentiment'] == '긍정'])
                            negative_count = len(st.session_state.cgv_reviews[st.session_state.cgv_reviews['sentiment'] == '부정'])
                            neutral_count = len(st.session_state.cgv_reviews[st.session_state.cgv_reviews['sentiment'] == '중립'])
                            
                            st.markdown(f"**전체 리뷰 수:** {len(st.session_state.cgv_reviews)}")
                            st.markdown(f"**긍정 리뷰:** {positive_count}")
                            st.markdown(f"**부정 리뷰:** {negative_count}")
                            st.markdown(f"**중립 리뷰:** {neutral_count}")
                        
                        with col2:
                            st.markdown("### 상세 정보")
                            avg_confidence = st.session_state.cgv_reviews['confidence'].mean()
                            st.markdown(f"**평균 신뢰도:** {avg_confidence:.2f}")
                            sentiment_ratio = f"긍정: {positive_count/len(st.session_state.cgv_reviews)*100:.1f}% / 부정: {negative_count/len(st.session_state.cgv_reviews)*100:.1f}% / 중립: {neutral_count/len(st.session_state.cgv_reviews)*100:.1f}%"
                            st.markdown(f"**감성 비율:** {sentiment_ratio}")

                        col1, col2 = st.columns(2)
                        with col1:
                            st.subheader("📊 시각화 결과")
                            fig = px.histogram(
                                st.session_state.cgv_reviews,
                                x="confidence",
                                color="sentiment",
                                title="감성 신뢰도 분포",
                                labels={"confidence": "Confidence", "sentiment": "Sentiment"},
                                color_discrete_map={"긍정": "#28A745", "부정": "#FF073A", "중립": "#FF073A"}
                            )
                            st.plotly_chart(fig, use_container_width=True)

                        with col2:
                            sentiment_counts = st.session_state.cgv_reviews['sentiment'].value_counts()
                            fig_pie = px.pie(
                                values=sentiment_counts.values,
                                names=sentiment_counts.index,
                                title="감성 분석 분포",
                                color=sentiment_counts.index,
                                color_discrete_map={"긍정": "#28A745", "부정": "#FF073A", "중립": "#FF073A"}
                            )
                            st.plotly_chart(fig_pie, use_container_width=True)

                        st.subheader("🔤 워드클라우드")
                        create_wordcloud(st.session_state.cgv_reviews['review'])
        
                        st.subheader("📊 데이터 테이블")
                        gb = GridOptionsBuilder.from_dataframe(st.session_state.cgv_reviews[["review", "sentiment", "confidence"]])
                        gb.configure_pagination(paginationAutoPageSize=True)
                        gb.configure_side_bar()
                        gb.configure_default_column(
                            groupable=True,
                            value=True,
                            enableRowGroup=True,
                            editable=False,
                            filterable=True,
                        )
                        grid_options = gb.build()

                        AgGrid(
                            st.session_state.cgv_reviews[["review", "sentiment", "confidence"]],
                            gridOptions=grid_options,
                            height=300,
                            theme="balham",
                            data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
                            update_mode=GridUpdateMode.MODEL_CHANGED,
                            fit_columns_on_grid_load=True,
                            allow_unsafe_jscode=True
                        )
                if st.button("프롬프트 생성", key="cgv_prompt_button"):
                    with st.spinner("프롬프트 생성 중..."):
                        summarized_content = cgv_summarize_reviews(
                            st.session_state.cgv_reviews,
                            st.session_state.cgv_movie_data
                        )
                        if summarized_content:
                            generated_prompt = generate_cgv_prompt(
                                st.session_state.cgv_reviews, 
                                st.session_state.cgv_movie_data, 
                                summarized_content
                            )
                            st.success("프롬프트 생성 완료!")

                            st.subheader("생성된 이미지 프롬프트")
                            st.text_area("프롬프트:", value=generated_prompt, height=400)
                        else:
                            st.error("내용 요약 중 오류가 발생했습니다.")



########################################################################################


    


    
    st.markdown(
    """
    <hr style="border: none; border-top: 2px dotted #c0c0c0; margin: 20px 0;">
    """,
    unsafe_allow_html=True
)

    with st.container(border=True, key="megabox_container"):
        st.header("🔍 Megabox 리뷰 크롤링")
        with st.expander("크롤링 및 검색"):
            megabox_search_query = st.text_input("검색할 영화 제목을 입력하세요:", key="megabox_search_query")
            if st.button("Megabox 검색", key="megabox_search_button"):
                if megabox_search_query:
                    with st.spinner('Megabox 검색중...'):
                        megabox = MegaboxTitleCrawler()
                        megabox_movie_titles, megabox_movie_info = megabox.fetch_movie_titles(megabox_search_query)
                        if megabox_movie_titles:
                            st.session_state.megabox_movie_titles = megabox_movie_titles
                            st.session_state.megabox_movie_info = megabox_movie_info
                            st.session_state.selected_movie = megabox_movie_titles
                        else:
                            st.warning("메가박스에서 영화를 찾을 수 없습니다.")
            st.session_state.megabox_selectbox = st.selectbox(
                "Megabox 검색 결과:", 
                st.session_state.get("megabox_movie_titles"),
                format_func=lambda x: f"{x}".replace("()", ""), 
                # st.session_state.get("megabox_movie_titles", []),
                # format_func=lambda x: f"{x}".replace("()", ""), 
                key="megabox_movie_selectbox",
            )
            st.session_state.megabox_selected_movie = st.session_state.megabox_selectbox
            megabox_review_limit = st.number_input("크롤링할 Megabox 리뷰 수", min_value=1, value=5, key="megabox_review_limit")
            if st.button("영화 정보 가져오기", key="megabox_info_button"):
                with st.spinner("영화 정보 가져오는 중... 잠시만 기다려주세요."):
                    megabox_detail = MegaboxDetailCrawler()
                    megabox_movie_data = megabox_detail.crawl_movie_details(st.session_state.megabox_selected_movie, review_limit=megabox_review_limit) 
                    if megabox_movie_data:
                        st.session_state.megabox_movie_data = megabox_movie_data

                        st.write('### 영화 정보')
                        st.write(f'**제목:** {megabox_movie_data["Title"]}')
                        st.write(f'**감독:** {megabox_movie_data["Directors"]}')
                        st.write(f'**출연:** {megabox_movie_data["Casts"]}')
                        st.write(f'**줄거리:** {megabox_movie_data["Plot"]}')

                        st.dataframe(megabox_movie_data["Reviews"])
                        st.session_state.megabox_reviews = megabox_movie_data["Reviews"]
                        st.session_state.show_megabox_sentiment = True
                    else:
                        st.warning("영화 정보를 찾을 수 없습니다.") 

            if st.session_state.megabox_reviews is not None and st.session_state.show_megabox_sentiment:
                st.markdown("### Megabox 감성 분석")    
                if st.button("Megabox 리뷰 감성 분석", key="megabox_sentiment_button"):
                    with st.spinner("감성 분석 중... 잠시만 기다려주세요."):
                        progress_bar = st.progress(0)
                        total_reviews = len(st.session_state.megabox_reviews)
                        
                        analyzed_reviews = []
                        for idx, review in st.session_state.megabox_reviews.iterrows():
                            sentiment_result = predict_sentiment(review['review'], "kobert_konply")
                            analyzed_reviews.append(sentiment_result)
                            progress_bar.progress((idx + 1) / total_reviews)
                        
                        st.session_state.megabox_reviews["sentiment_analysis"] = analyzed_reviews
                        st.session_state.megabox_reviews["sentiment"] = st.session_state.megabox_reviews["sentiment_analysis"].apply(lambda x: x["sentiment"])
                        st.session_state.megabox_reviews["confidence"] = st.session_state.megabox_reviews["sentiment_analysis"].apply(
                            lambda x: float(x["confidence"].strip('%')) / 100
                        )
                        
                        st.success("감성 분석 완료!")
                        st.subheader("감성 분석 결과")
                        st.balloons()
                        
                        st.dataframe(st.session_state.megabox_reviews[["review", "sentiment", "confidence"]], use_container_width=True)
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("### 분석 결과")
                            positive_count = len(st.session_state.megabox_reviews[st.session_state.megabox_reviews['sentiment'] == '긍정'])
                            negative_count = len(st.session_state.megabox_reviews[st.session_state.megabox_reviews['sentiment'] == '부정'])
                            neutral_count = len(st.session_state.megabox_reviews[st.session_state.megabox_reviews['sentiment'] == '중립'])
                            
                            st.markdown(f"**전체 리뷰 수:** {len(st.session_state.megabox_reviews)}")
                            st.markdown(f"**긍정 리뷰:** {positive_count}")
                            st.markdown(f"**부정 리뷰:** {negative_count}")
                            st.markdown(f"**중립 리뷰:** {neutral_count}")
                        
                        with col2:
                            st.markdown("### 상세 정보")
                            avg_confidence = st.session_state.megabox_reviews['confidence'].mean()
                            st.markdown(f"**평균 신뢰도:** {avg_confidence:.2f}")
                            sentiment_ratio = f"긍정: {positive_count/len(st.session_state.megabox_reviews)*100:.1f}% / 부정: {negative_count/len(st.session_state.megabox_reviews)*100:.1f}% / 중립: {neutral_count/len(st.session_state.megabox_reviews)*100:.1f}%"
                            st.markdown(f"**감성 비율:** {sentiment_ratio}")

                        col1, col2 = st.columns(2)
                        with col1:
                            st.subheader("📊 시각화 결과")
                            fig = px.histogram(
                                st.session_state.megabox_reviews,
                                x="confidence",
                                color="sentiment",
                                title="감성 신뢰도 분포",
                                labels={"confidence": "Confidence", "sentiment": "Sentiment"},
                                color_discrete_map={"긍정": "#28A745", "부정": "#FF073A", "중립": "#FF073A"}
                            )
                            st.plotly_chart(fig, use_container_width=True)

                        with col2:
                            sentiment_counts = st.session_state.megabox_reviews['sentiment'].value_counts()
                            fig_pie = px.pie(
                                values=sentiment_counts.values,
                                names=sentiment_counts.index,
                                title="감성 분석 분포",
                                color=sentiment_counts.index,
                                color_discrete_map={"긍정": "#28A745", "부정": "#FF073A", "중립": "#FF073A"}
                            )
                            st.plotly_chart(fig_pie, use_container_width=True)

                        st.subheader("🔤 워드클라우드")
                        create_wordcloud(st.session_state.megabox_reviews['review'])


                        gb = GridOptionsBuilder.from_dataframe(st.session_state.megabox_reviews[["review", "sentiment", "confidence"]])
                        gb.configure_pagination(paginationAutoPageSize=True)
                        gb.configure_side_bar()
                        gb.configure_default_column(
                            groupable=True,
                            value=True,
                            enableRowGroup=True,
                            editable=False,
                            filterable=True,
                        )
                        grid_options = gb.build()

                        AgGrid(
                            st.session_state.cgv_reviews[["review", "sentiment", "confidence"]],
                            gridOptions=grid_options,
                            height=300,
                            theme="balham",
                            data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
                            update_mode=GridUpdateMode.MODEL_CHANGED,
                            fit_columns_on_grid_load=True,
                            allow_unsafe_jscode=True
                        )
                if st.button("프롬프트 생성", key="megabox_prompt_button"):
                    with st.spinner("프롬프트 생성 중..."):
                        summarized_content = megabox_summarize_reviews(
                            st.session_state.megabox_reviews,
                            st.session_state.megabox_movie_data
                        )
                        
                        if summarized_content:
                            generated_prompt = generate_megabox_prompt(
                                st.session_state.megabox_reviews,
                                st.session_state.megabox_movie_data,
                                summarized_content
                            )
                            
                            st.success("프롬프트 생성 완료!")
                            # st.subheader("요약된 내용")
                            # st.write(summarized_content)
                            
                            st.subheader("생성된 이미지 프롬프트")
                            st.text_area("프롬프트:", value=generated_prompt, height=400)
                        else:
                            st.error("내용 요약 중 오류가 발생했습니다.")

            
            


