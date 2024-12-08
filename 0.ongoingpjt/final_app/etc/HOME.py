import streamlit as st
from s3 import s3_client
from nav import inject_custom_navbar
# Set page config
st.set_page_config(page_title="영화 리뷰 감성 분석", layout="wide", initial_sidebar_state="collapsed")
inject_custom_navbar()

# Custom CSS for styling
st.markdown(
    """
    <style>

        html, body, [data-testid="stAppViewContainer"] {
            background-color: #1C1C1E;
            color: white;
        }
        [data-testid="stMainBlockContainer"] {
            width: 80%;
            margin: auto;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Homepage content
st.markdown(
    """
    <div style="text-align: center;">
        <h1 style="color: #ff416c;">2조 오뚝이</h1>
        <h2>영화 리뷰 감성 분석</h2>
    </div>
    """,
    unsafe_allow_html=True
)

# Display images
images = s3_client(st.secrets['access_key'], st.secrets['secret_access_key']).get_images()
titles = ["Moana2", "Wicked", "Gladiator2"]

cols = st.columns(3)
for idx, col in enumerate(cols):
    with col:
        st.image(images[idx], use_container_width=True)
        st.markdown(f"<h4 style='text-align: center;'>{titles[idx]}</h4>", unsafe_allow_html=True)

# Navigation Button
st.markdown(
    """
    <style>
    div.stButton > button {
        background: linear-gradient(90deg, #ff4b2b, #ff416c);
        color: white;
        padding: 12px 24px;
        font-size: 18px;
        font-weight: bold;
        border: none;
        border-radius: 25px;
        cursor: pointer;
        transition: all 0.3s ease-in-out;
        box-shadow: 0 4px 10px rgba(255, 65, 108, 0.3);
    }
    div.stButton > button:hover {
        background: linear-gradient(90deg, #ff416c, #ff4b2b);
        transform: scale(1.05);
        box-shadow: 0 6px 15px rgba(255, 65, 108, 0.5);
    }
    div.stButton {
        display: flex;
        justify-content: center;
        margin-top: 30px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Redirect to main page using st.query_params
if st.button("메인 페이지로 이동"):
    st.switch_page("pages/1. MAIN.py")

    