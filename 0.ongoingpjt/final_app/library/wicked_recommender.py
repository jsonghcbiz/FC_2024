import streamlit as st
from openai import OpenAI
import time

def setup_openai_client():
    """OpenAI 클라이언트 설정 및 에러 처리"""
    try:
        api_key = st.secrets["openai"]["api_key"]
        if not api_key or not api_key.startswith("sk-"):
            st.error("⚠️ 올바른 API 키가 설정되지 않았습니다. .streamlit/secrets.toml 파일을 확인해주세요.")
            st.stop()
        return OpenAI(api_key=api_key)
    except Exception as e:
        st.error(f"⚠️ API 키 설정 중 오류가 발생했습니다: {e}")
        st.stop()

def get_mbti_options():
    """MBTI 옵션 반환"""
    return [
        "ENFJ", "ENFP", "ENTJ", "ENTP",
        "ESFJ", "ESFP", "ESTJ", "ESTP",
        "INFJ", "INFP", "INTJ", "INTP",
        "ISFJ", "ISFP", "ISTJ", "ISTP",
    ]

def generate_character_recommendation(client, description: str, mbti: str) -> str:
    """캐릭터 추천 생성"""
    prompt = f"""Based on the following information about a person, recommend a character from the musical 'Wicked' that best matches their personality:

User's Description: {description}
User's MBTI: {mbti}

Consider the following main characters from Wicked:
- Elphaba (The Wicked Witch of the West)
- Glinda (The Good Witch)
- Fiyero (The Scarecrow Prince)
- Madame Morrible
- Doctor Dillamond
- Nessarose (The Wicked Witch of the East)
- The Wonderful Wizard of Oz
- Boq (The Tin Man)

Please do recommend the character that best matches the user's personality. Do not recommend the same character twice for same MBTI.



Please provide your response in Korean with:
1. 추천 캐릭터
2. 추천 이유
3. 공통된 성격 특징
"""

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert on the musical 'Wicked' and personality analysis. Please respond in Korean."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"⚠️ 추천 생성 중 오류가 발생했습니다: {e}")
        return None

def main():
    
    client = setup_openai_client()
    
    st.markdown("""
    #### 뮤지컬 'Wicked'의 등장인물 중 당신과 가장 잘 맞는 캐릭터를 찾아보세요! ✨
    자신의 성격과 특징을 입력하면 AI가 가장 잘 맞는 캐릭터를 추천해드립니다.
    """)
    
    with st.container():
        user_description = st.text_area(
            "🤔 자신을 설명해주세요:",
            placeholder="성격, 가치관, 좋아하는 것들을 자유롭게 적어주세요...",
            height=150
        )
        
        mbti = st.selectbox(
            "📊 MBTI를 선택하세요:",
            get_mbti_options()
        )
    
    if st.button("✨ 캐릭터 추천받기", type="primary") and user_description:
        with st.spinner("🔮 당신과 잘 맞는 캐릭터를 찾는 중..."):
            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.01)
                progress_bar.progress(i + 1)
            
            recommendation = generate_character_recommendation(client, user_description, mbti)
            if recommendation:
                st.success("캐릭터 추천이 완료되었습니다! 🎉")
                st.markdown("### ✨ 추천 결과")
                st.write(recommendation)
    else:
        st.info("💡 캐릭터 추천을 받으려면 자기소개를 입력하고 MBTI를 선택해주세요.")

if __name__ == "__main__":
    main() 