import streamlit as st
from openai import OpenAI
import requests
from PIL import Image
from io import BytesIO
import time

try:
    api_key = st.secrets["openai"]["api_key"]
    if not api_key or not api_key.startswith("sk-"):
        st.error("올바른 API 키가 설정되지 않았습니다. .streamlit/secrets.toml 파일을 확인해주세요.")
        st.stop()
    client = OpenAI(api_key=api_key)
except Exception as e:
    st.error(f"API 키 설정 중 오류가 발생했습니다: {e}")
    st.stop()

st.markdown(
    """
    <style>
        /* Your existing styles... */
    </style>
    """,
    unsafe_allow_html=True,
)

def sanitize_input(user_input: str) -> str:
    forbidden_words = ["violence", "explicit", "hate", "illegal", "attack"]
    sanitized_input = user_input
    for word in forbidden_words:
        sanitized_input = sanitized_input.replace(word, "[redacted]")
    return sanitized_input

def generate_prompt(user_input: str) -> str:
    # First sanitize the input
    sanitized_input = sanitize_input(user_input)
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a creative professional artist that generates safe, detailed, and artistic prompts for image creation."},
            {"role": "user", "content": f"Refine the following text into one paragraph with a creative and artistic prompt for generating an image:\n\n{sanitized_input}"}
        ],
        max_tokens=300,
        temperature=0.7
    )
    return response.choices[0].message.content.strip()

def generate_image(refined_prompt: str) -> Image.Image:
    try:
        # Show a simple progress message
        progress_placeholder = st.empty()
        progress_placeholder.text("이미지 생성 중...")
        
        # Call OpenAI API
        response = client.images.generate(
            prompt=refined_prompt,
            n=1,
            size="256x256"
        )
        image_url = response.data[0].url
        image_response = requests.get(image_url)
        
        # Clear the progress message
        progress_placeholder.empty()
        
        return Image.open(BytesIO(image_response.content))
    except Exception as e:
        if "content_policy_violation" in str(e):
            st.error("The image prompt violates content policy. Please revise the prompt.")
        else:
            st.error(f"An error occurred: {e}")
        return None

# Main UI flow
def main():
    st.markdown("""
    #### AI 이미지 생성기 🎨
    당신이 상상하는 장면을 텍스트로 입력해보세요.
    AI가 당신의 상상을 아름다운 이미지로 만들어드립니다!
    """)
    
    # User input
    user_input = st.text_input("프롬프트 생성:", placeholder="앞에서 생성한 프롬프트를 활용하거나, 생성하고 싶은 이미지를 구체적으로 입력해주세요.")
    
    if user_input:
        # Generate prompt button
        if st.button("프롬프트 생성"):
            refined_prompt = generate_prompt(user_input.strip())
            
            st.subheader("프롬프트 생성 결과")
            st.write(refined_prompt)
            
            # Store the refined prompt in session state
            st.session_state['refined_prompt'] = refined_prompt
            
        # Generate image button
        if 'refined_prompt' in st.session_state and st.button("이미지 생성"):
            with st.spinner("이미지 생성 중..."):
                image = generate_image(st.session_state['refined_prompt'])
                if image:
                    st.image(image, caption="생성된 이미지")
    else:
        st.info("이미지를 생성하려면 프롬프트를 입력해주세요.")

if __name__ == "__main__":
    main()