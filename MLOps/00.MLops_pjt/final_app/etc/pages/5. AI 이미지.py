import streamlit as st
from nav import inject_custom_navbar
from openai import OpenAI
import requests
from PIL import Image
from io import BytesIO

# Initialize OpenAI client
try:
    api_key = st.secrets["openai"]["api_key"]
    if not api_key or not api_key.startswith("sk-"):
        st.error("올바른 API 키가 설정되지 않았습니다. .streamlit/secrets.toml 파일을 확인해주세요.")
        st.stop()
    client = OpenAI(api_key=api_key)
except Exception as e:
    st.error(f"API 키 설정 중 오류가 발생했습니다: {e}")
    st.stop()

# Configure Streamlit page
st.set_page_config(
    page_title="AI 이미지 생성",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Inject custom navigation bar
inject_custom_navbar()

# Page Title
st.title("AI 이미지 생성")
st.write("AI 이미지 생성 페이지 내용을 여기에 추가하세요.")

# Sanitize user input
def sanitize_input(user_input: str) -> str:
    """
    Cleans and validates user input to avoid triggering OpenAI's safety filters.
    """
    forbidden_words = ["violence", "explicit", "hate", "illegal", "attack"]
    sanitized_input = user_input
    for word in forbidden_words:
        sanitized_input = sanitized_input.replace(word, "[redacted]")
    return sanitized_input

# Function to ensure the refined prompt fits within 999 characters
def enforce_character_limit(text: str, limit: int = 999) -> str:
    """
    Truncates the text to ensure it fits within the character limit.

    Args:
        text (str): The input text to truncate.
        limit (int): The character limit (default is 999).

    Returns:
        str: Truncated text.
    """
    return text[:limit]

st.write("Input your text to generate a refined and descriptive prompt.")

# User Input
user_input = st.text_area("Enter your text:", placeholder="Describe what you want to create...", height=150)

# Button to Generate Prompt
if st.button("Generate Prompt"):
    if user_input.strip():
        with st.spinner("Generating a refined prompt..."):
            try:
                sanitized_input = sanitize_input(user_input.strip())
                # Call OpenAI GPT to refine the input
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a creative assistant that generates safe, detailed, and artistic prompts for image creation."},
                        {"role": "user", "content": f"Refine the following text into one paragraph with a creative and artistic prompt for generating an image. Keep it concise and limit the description:\n\n{sanitized_input}"}
                    ],
                    max_tokens=300,  # Approximation to ensure the result is within 999 characters
                    temperature=0.7
                )
                refined_prompt = response.choices[0].message.content.strip()
                # Enforce 999-character limit
                refined_prompt = enforce_character_limit(refined_prompt, limit=999)
                st.text_area("Refined Prompt", value=refined_prompt, height=200)
            except Exception as e:
                if "content_policy_violation" in str(e):
                    st.error("Your input or generated prompt contains sensitive content. Please revise your text to comply with content guidelines.")
                else:
                    st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter some text to generate a prompt.")

# Prompt input
prompt_input = st.text_input("이미지 생성 프롬프트를 입력하세요:")

# Function to generate an AI image
def generate_ai_image(prompt: str) -> Image:
    try:
        response = client.images.generate(
            prompt=prompt,
            n=1,
            size="512x512"
        )
        image_url = response.data[0].url
        image_response = requests.get(image_url)
        image = Image.open(BytesIO(image_response.content))
        return image
    except Exception as e:
        if "content_policy_violation" in str(e):
            st.error("The image prompt violates content policy. Please revise the prompt.")
        else:
            st.error(f"An error occurred: {e}")
        return None

# Generate image on button click
if st.button("이미지 생성"):
    if prompt_input.strip():
        with st.spinner("이미지를 생성하는 중..."):
            image = generate_ai_image(prompt_input.strip())
            if image:
                st.image(image, caption="생성된 이미지", use_container_width=True)
            else:
                st.error("이미지 생성에 실패했습니다. 다시 시도해주세요.")
    else:
        st.warning("이미지 생성 프롬프트를 입력하세요.")