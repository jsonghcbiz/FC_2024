import streamlit as st
import pandas as pd
import plotly.express as px
from senti_classifier_kobert import predict_sentiment
from st_aggrid import AgGrid, GridOptionsBuilder
from nav import inject_custom_navbar

st.set_page_config(page_title="Sentiment Analysis", 
                   layout="wide",
                   initial_sidebar_state="collapsed"   # 사이드바 확장
                   )
inject_custom_navbar()
MODEL_PATH = 'kobert_konply'

# Custom CSS for styling
st.markdown(
    """
    <style>
        /* Background and font */
        html, body, [data-testid="stAppViewContainer"] {
            background-color: #1C1C1E;
            color: #F0F0F0;
            font-family: 'Arial', sans-serif;
        }

        /* Title and subheader styles */
        h1 {
            color: #FF416C;
            text-align: center;
            margin-bottom: 10px;
        }
        h2, h3 {
            color: #FFFFFF;
            margin-top: 20px;
        }

        /* Input box styling */
        textarea {
            background-color: #2C2C2E;
            color: #FFFFFF;
            border: 1px solid #555;
            border-radius: 10px;
        }

        /* Buttons */
        div.stButton > button {
            background: linear-gradient(90deg, #ff4b2b, #ff416c);
            color: white;
            font-size: 16px;
            padding: 10px 20px;
            border: none;
            border-radius: 20px;
            cursor: pointer;
            transition: all 0.3s ease-in-out;
            box-shadow: 0 4px 10px rgba(255, 65, 108, 0.3);
        }
        div.stButton > button:hover {
            background: linear-gradient(90deg, #ff416c, #ff4b2b);
            transform: scale(1.05);
        }

        /* Uploaded file box */
        div[data-testid="stFileUploader"] {
            background-color: #2C2C2E;
            border: 1px dashed #FF416C;
            border-radius: 10px;
        }

        /* Dataframe styling */
        div[data-testid="stDataFrameContainer"] {
            background-color: #2C2C2E;
            border-radius: 10px;
            color: #F0F0F0;
        }

        /* Chart title */
        .plotly-container .main-svg {
            fill: #FFFFFF;
        }
        .metric-box {
            background-color: #2C2C2E;
            padding: 10px;
            border-radius: 10px;
            text-align: center;
        }
    </style>
    """,
    unsafe_allow_html=True,
)



# Page Title
st.title("🎉 감성 분석 페이지")
st.subheader("입력하신 리뷰의 감성을 분석해 드립니다.")

# Input Section
st.markdown("---")
st.subheader("📋 분석할 리뷰 입력")
user_input = st.text_area("분석할 텍스트를 입력하세요:", height=150)

if st.button("감성 분석"):
    if user_input.strip():
        with st.spinner("분석 중..."):
            try:
                sentiment = predict_sentiment(user_input, MODEL_PATH)
                st.markdown(f"**입력 텍스트:** {user_input}")
                st.markdown(f"**감성 결과:** {sentiment['sentiment']}")
                st.markdown(f"**신뢰도:** {sentiment['confidence']}")
                st.markdown(f"**확률 분포:** {sentiment['probabilities']}")
            except Exception as e:
                st.error(f"분석 중 오류가 발생했습니다: {e}")
    else:
        st.warning("분석할 텍스트를 입력하세요.")


# File Upload Section
st.markdown("---")
st.subheader("📂 첨부한 리뷰 파일 분석")
uploaded_file = st.file_uploader("CSV 파일 업로드 (리뷰가 포함된 파일)", type="csv")

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.success("파일 업로드 성공!")
        st.dataframe(df.head(), use_container_width=True)
        st.write("사용 가능한 열:", df.columns.tolist())

        # Select column for sentiment analysis
        selected_column = st.selectbox("분석할 열 선택", df.columns)
    
        # Perform Sentiment Analysis
        # df["sentiment_score"] = df[selected_column].apply(lambda x: predict_sentiment(x, MODEL_PATH))

        if st.button("분석 시작"):
            with st.spinner("분석 중..."):
                df[selected_column] = df[selected_column].fillna("Unknown").astype(str)
                df["analysis"] = df[selected_column].apply(lambda x: predict_sentiment(x, MODEL_PATH))
                df["sentiment"] = df["analysis"].apply(lambda x: x['sentiment'])
                df["confidence"] = df["analysis"].apply(lambda x: x['confidence'])

                # Display summary
                st.success("분석 완료!")
                sentiment_counts = df["sentiment"].value_counts()
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("긍정 리뷰", f"{sentiment_counts.get('긍정', 0)}개")
                with col2:
                    st.metric("부정 리뷰", f"{sentiment_counts.get('부정', 0)}개")

                # Display histogram of confidence scores
                st.subheader("📊 감성 신뢰도 분포")
                fig = px.histogram(
                    df,
                    x="confidence",
                    color="sentiment",
                    title="감성 신뢰도 분포",
                    labels={"confidence": "Confidence", "sentiment": "Sentiment"},
                    color_discrete_map={"긍정": "#28A745", "부정": "#FF073A"}
                )
                st.plotly_chart(fig, use_container_width=True)

                st.subheader("📋 분석 결과")
                gb = GridOptionsBuilder.from_dataframe(df)
                gb.configure_pagination(paginationAutoPageSize=True)  # Enable pagination
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
                    theme="balham",  # "light", "dark", "blue", "fresh", "material"
                    update_mode="MODEL_CHANGED",
                    fit_columns_on_grid_load=True,
                )

    except Exception as e:
        st.error(f"파일 처리 중 오류가 발생했습니다: {e}")
