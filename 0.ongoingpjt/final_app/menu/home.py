import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from wordcloud import WordCloud
from st_aggrid import AgGrid, GridOptionsBuilder
from library.kobert_classifier1 import predict_sentiment

def analyze_sentiment(df, model_path="kobert_konply"):
    try: 
        df["sentiment_analysis"] = df["review"].apply(
            lambda x: predict_sentiment(x, model_path)
        )
        df["sentiment"] = df["sentiment_analysis"].apply(lambda x: x["sentiment"])
        df["confidence"] = df["sentiment_analysis"].apply(lambda x: x["confidence"])
    except Exception as e:
        st.error(f"감성 분석 중 오류 발생: {e}")
    return df

def home_view(titles, images, buttons, detail_images):
    
    # 상영중인 영화 카드 3개 
    cols = st.columns(3)
    for idx, col in enumerate(cols):
        with col.container(border=True, key=f"{titles[idx]}_container"):
            st.image(image=images[idx], use_container_width=True)
            buttons.append(st.button(label=titles[idx], key=f"{titles[idx]}_btn", use_container_width=True))

    def view(idx):
        with st.container(border=True):
            st.progress(80, text="⭐️⭐️ 별점 ⭐️⭐️")

        exp_cols = st.columns(2)
        with exp_cols[0].container(border=True):
            st.image(image=detail_images[0], use_container_width=True)
        
        with exp_cols[1].container(border=True):
            # 데이터 설정
            labels = ['유쾌', '상쾌', '통쾌']
            sizes = [50, 30, 20]
            colors = ['lightskyblue', 'lightcoral', 'lightgreen']

            # 파이 차트 그리기
            plt.figure(figsize=(7, 7))
            plt.axis('equal')  # 원형으로 그리기 위해 비율 설정
            plt.rcParams['font.family'] = 'AppleGothic'  # 맥의 경우
            fig, ax = plt.subplots()
            ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)

            st.pyplot(fig)

        text = "2조 오뚝이 최진호 김현진 김남섭 박지은 송주은 영화 리뷰 감성 분석 모아나2 위키드 글래디에이터2 좋아요 싫어요 CGV Megabox"
        wordcloud = WordCloud(font_path='AppleGothic', width=400, height=400, background_color='white').generate(text)
        plt.figure(figsize=(5, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        st.pyplot(plt)




        df = pd.read_csv("library/home_team2.csv")
        df = analyze_sentiment(df, model_path="kobert_konply")
        gb = GridOptionsBuilder.from_dataframe(df)
        gb.configure_pagination(paginationAutoPageSize=True)
        gb.configure_default_column(
            groupable=True,
            value=True,
            enableRowGroup=True,
            editable=False,
            filterable=True,
        )
        gb.configure_column("confidence", type=["numericColumn"], precision=2)
        grid_options = gb.build()

        AgGrid(
            df,
            gridOptions=grid_options,
            height=300,
            theme="balham",
            update_mode="MODEL_CHANGED",
            fit_columns_on_grid_load=True,
        )
        # ------------------------------------------------------------

    with st.expander("영화 리뷰 감성 보기",expanded=True):
        if buttons[0]: 
            st.title(titles[0])
            view(0)
        elif buttons[1]:
            st.title(titles[1])
        elif buttons[2]:
            st.title(titles[2])

    print("home_view")