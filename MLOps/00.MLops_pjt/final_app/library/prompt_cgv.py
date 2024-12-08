import streamlit as st
from openai import OpenAI
import requests
from PIL import Image
from io import BytesIO


try:
    api_key = st.secrets["openai"]["api_key"]
    if not api_key or not api_key.startswith("sk-"):
        st.error("올바른 API 키가 설정되지 않았습니다. .streamlit/secrets.toml 파일을 확인해주세요.")
        st.stop()
    client = OpenAI(api_key=api_key)
except Exception as e:
    st.error(f"API 키 설정 중 오류가 발생했습니다: {e}")
    st.stop()

def cgv_summarize_reviews(reviews_df, cgv_movie_data):
    # 감성 분석 결과 가져오기
    sentiments = {
        '긍정': '',
        '부정': '',
        '중립': ''
    }
    if 'sentiment' in reviews_df.columns:
        for sentiment in sentiments.keys():
            reviews = reviews_df[reviews_df['sentiment'] == sentiment]['review'].tolist()
            sentiments[sentiment] = ' '.join(reviews[:3]) if reviews else ''
    plot = cgv_movie_data.get('Plot', '')
    summary_prompt = f"""
영화 줄거리와 리뷰를 간단히 요약해주세요:

줄거리: {plot}

관객 리뷰:
긍정적 의견: {sentiments['긍정'][:200]}
부정적 의견: {sentiments['부정'][:200]}
중립적 의견: {sentiments['중립'][:200]}

다음 형식으로 요약해주세요:
1. 줄거리 요약 (3문장)
2. 긍정적 평가 요약 (3문장)
3. 부정적 평가 요약 (3문장) 

600자 이내로 영어로 요약해주세요.
"""
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a professional movie critic who provides concise summaries in Korean."},
                {"role": "user", "content": summary_prompt}
            ],
            max_tokens=500,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"요약 생성 중 오류가 발생했습니다: {e}")
        return ""
    
def get_tone(positive_ratio):
    if positive_ratio >= 0.7:
        return "밝고 희망적인"
    elif positive_ratio <= 0.3:
        return "어둡고 긴장감 있는"
    return "대비되는 명암이 강한"

def generate_cgv_prompt(reviews_df, cgv_movie_data, summarized_content):
    # 영화 정보 가져오기
    total_reviews = len(reviews_df)
    if 'sentiment' in reviews_df.columns and total_reviews > 0:
        positive_reviews = reviews_df[reviews_df['sentiment'] == '긍정']['review'].tolist()
        positive_ratio = len(positive_reviews) / total_reviews
    else:
        positive_ratio = 0.5


    movie_info = {
        'title': cgv_movie_data.get('Title', ''),
        'director': cgv_movie_data.get('Directors', ''),
        'cast': cgv_movie_data.get('Cast', '')
    }

    tone = get_tone(positive_ratio)

    image_prompt = f"""
A professional artwork in the style of contemporary gallery art:

Subject: An artistic interpretation of the movie '{movie_info['title']}'
Atmosphere: {tone}

Style specifications:
- Modern art composition which depicts the {summarized_content}
- Elegant watercolor
- Clean and balanced composition
- Elegant and symbolic representation of emotions
- Color palette: {positive_ratio:.0%} warm/positive tones

Additional notes:
- Viewers should be able to understand the movie from the artwork
- High-quality rendering
- Contemporary art style suitable for public display and potential for high value at Art Basel
- Suitable for general audiences
- No text or explicit elements
- Focus on artistic interpretation
- Make it under 1000 characters
"""
    return image_prompt

def main():
    if 'cgv_summarize_prompt' in st.session_state:
        cgv_summarize_prompt = st.session_state.get('cgv_summarize_prompt')
        if st.button("프롬프트 생성"):
            if 'reviews_df' in st.session_state and 'cgv_movie_data' in st.session_state:
                summarized_content = cgv_summarize_reviews(
                    st.session_state.reviews_df,
                    st.session_state.cgv_movie_data
                )
                refined_prompt = generate_cgv_prompt(
                    st.session_state.reviews_df,
                    st.session_state.cgv_movie_data,
                    summarized_content
                )
                st.subheader("프롬프트 생성 결과")
                st.write(refined_prompt)
                st.session_state['refined_prompt'] = refined_prompt
            else:
                st.error("필요한 데이터가 없습니다. 영화 리뷰를 먼저 분석해주세요.")

if __name__ == "__main__":
    main()