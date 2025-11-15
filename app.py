import streamlit as st
from google import genai

# Load your Gemini API key securely
client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])

# Function to call Gemini 2.5 Flash
def call_gemini(prompt: str, model: str = "gemini-2.5-flash"):
    try:
        # Generate content using Gemini 2.5 Flash model
        response = client.models.generate_content(
            model=model,
            contents=prompt
        )
        return response.text
    except Exception as e:
        return f"⚠️ Error: {str(e)}"

# Streamlit UI
st.set_page_config(page_title="Gemini 2.5 Flash Chatbot 💬")
st.title("🤖 Chatbot using Gemini 2.5 Flash")

user_input = st.text_input("You:", placeholder="Type your message here...")

if st.button("Send"):
    if user_input.strip():
        st.chat_message("user").write(user_input)
        bot_response = call_gemini(user_input)
        st.chat_message("assistant").write(bot_response)
    else:
        st.warning("Please enter a message.")
