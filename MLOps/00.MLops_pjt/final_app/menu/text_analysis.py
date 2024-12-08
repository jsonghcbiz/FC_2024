import streamlit as st
import pandas as pd
import os

import plotly.express as px
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, DataReturnMode
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from library.kobert_classifier1 import predict_sentiment

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "../kobert_konply")
FONT_PATH = 'AppleGothic'  # Define font path as a constant

def analyze_sentiment_text(input_text, model_path):
    try:
        sentiment = predict_sentiment(input_text, model_path)
        confidence_str = sentiment['confidence'].replace('%', '')
        confidence = float(confidence_str)
        probabilities = sentiment['probabilities']
        formatted_probabilities = ', '.join([f"{key}: {value}" for key, value in probabilities.items()])
        

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 분석 결과")
            st.markdown(f"**입력 텍스트:**\n{input_text}")
            st.markdown(f"**감성 결과:** {sentiment['sentiment']}")
        with col2:
            st.markdown("### 상세 정보")
            st.markdown(f"**신뢰도:** {confidence:.2f}")
            st.markdown(f"**확률 분포:** {formatted_probabilities}")
        st.balloons()
    except Exception as e:
        st.error(f"분석 중 오류가 발생했습니다: {str(e)}")

def analyze_sentiment_file(uploaded_file, model_path):
    try:
        df = pd.read_csv(uploaded_file)
        
        progress_bar = st.progress(0)
        total_rows = len(df)
        
        results = []
        for index, row in df.iterrows():
            sentiment_result = predict_sentiment(row['review'], model_path)
            results.append(sentiment_result)
            progress_bar.progress((index + 1) / total_rows)
        
        df["sentiment_analysis"] = results
        df["sentiment"] = df["sentiment_analysis"].apply(lambda x: x["sentiment"])
        df["confidence"] = df["sentiment_analysis"].apply(lambda x: x["confidence"])
        
        st.subheader("📊 데이터 테이블")
        gb = GridOptionsBuilder.from_dataframe(df)
        gb.configure_pagination(paginationAutoPageSize=True)
        gb.configure_side_bar()
        gb.configure_default_column(
            groupable=True,
            value=True,
            enableRowGroup=True,
            editable=False,
            filterable=True
        )
        gb.configure_column("confidence", type=["numericColumn"], precision=2)
        
        grid_response = AgGrid(
            df,
            gridOptions=gb.build(),
            height=300,
            theme="balham",
            data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
            update_mode=GridUpdateMode.MODEL_CHANGED,
            fit_columns_on_grid_load=True,
            allow_unsafe_jscode=True
        )
        

        st.subheader("📈 감성 분석 분포")
        with st.container(border=True):
            sentiment_counts = df["sentiment"].value_counts()
            fig = px.pie(values=sentiment_counts.values, 
                        names=sentiment_counts.index, 
                        title='감성 분석 결과 분포')
            st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("🔤 워드클라우드")
        create_wordcloud(df['review'])
            
        return df
    except Exception as e:
        st.error(f"감성 분석 중 오류 발생: {str(e)}")
        return None

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

def text_analysis_view():
    st.markdown("""
    # 텍스트 분석 도구 ✒️
    입력된 텍스트의 감정, 주제, 키워드를 AI가 자동으로 분석합니다.
    워드클라우드와 차트로 텍스트의 특징을 시각화하여 보여드립니다.
    """)

    tab1, tab2 = st.tabs(["텍스트 입력", "파일 업로드"])
    
    with tab1:
        user_input = st.text_area("분석할 텍스트를 입력하세요:", height=150)
        if st.button("텍스트 분석", key="analyze_text"):
            if user_input.strip():
                with st.spinner("분석 중..."):
                    analyze_sentiment_text(user_input, MODEL_PATH)
            else:
                st.warning("텍스트를 입력해주세요.")
    
    with tab2:
        uploaded_file = st.file_uploader("CSV 파일 업로드 (리뷰가 포함된 파일)", type="csv")
        if st.button("파일 분석", key="analyze_file"):
            if uploaded_file is not None:
                with st.spinner("파일 분석 중..."):
                    df = analyze_sentiment_file(uploaded_file, MODEL_PATH)
            else:
                st.warning("CSV 파일을 업로드해주세요.")

    print("text_analysis_view")