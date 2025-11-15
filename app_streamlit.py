# app_streamlit.py
import streamlit as st
import time
from rag_utils import generate_response

NAME = "Sahaara — Mental Health Companion"

st.set_page_config(page_title=NAME, layout="wide", page_icon="💬")
st.markdown("""
<style>
.chat-column { max-width:800px; margin:auto; }
.msg-user { background:#DCF8C6; padding:10px; border-radius:10px; margin:6px 0; align-self:flex-end; max-width:75%; word-wrap:break-word; }
.msg-assistant { background:#FFFFFF; padding:10px; border-radius:10px; margin:6px 0; align-self:flex-start; max-width:75%; word-wrap:break-word; }
.chat-box { display:flex; flex-direction:column; }
.header { display:flex; align-items:center; gap:12px; }
.header img { border-radius:50%; }
.small-muted { color:#888; font-size:12px; }
</style>
""", unsafe_allow_html=True)

col1, col2 = st.columns([3,1])
with col1:
    st.markdown(f"<div class='header'><img src='https://img.icons8.com/color/48/000000/mental-health.png'/> <h2>{NAME}</h2></div>", unsafe_allow_html=True)
    st.markdown("**Sahaara** is a supportive chatbot. It can provide information, empathetic replies, and immediate helpline guidance if you appear in crisis. Not a replacement for professional help.")
with col2:
    st.sidebar.header("Settings")
    if "country_code" not in st.session_state:
        st.session_state.country_code = "DEFAULT"
    st.session_state.country_code = st.sidebar.text_input("Country ISO2 code (e.g. IN, US)", value=st.session_state.country_code).strip().upper() or "DEFAULT"
    st.sidebar.markdown("---")
    st.sidebar.markdown("Data & model settings (set via env or build step).")
    if st.sidebar.button("Clear chat"):
        st.session_state.history = []
        st.experimental_rerun()

if "history" not in st.session_state:
    st.session_state.history = []

def render_history():
    st.markdown('<div class="chat-column">', unsafe_allow_html=True)
    for turn in st.session_state.history:
        if turn["role"] == "user":
            st.markdown(f'<div class="chat-box" style="align-items:flex-end;"><div class="msg-user">{turn["text"]}</div></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-box" style="align-items:flex-start;"><div class="msg-assistant">{turn["text"]}</div></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

render_history()

# input area
with st.form(key="input_form", clear_on_submit=True):
    user_text = st.text_input("Type a message", key="input_text")
    submit = st.form_submit_button("Send")

if submit and user_text:
    st.session_state.history.append({"role": "user", "text": user_text})
    # re-render immediately to show user's message
    st.experimental_rerun()

# if last role is user -> generate
if st.session_state.history and st.session_state.history[-1]["role"] == "user":
    user_q = st.session_state.history[-1]["text"]
    placeholder = st.empty()
    try:
        out = generate_response(user_q, history=st.session_state.history, user_meta={"country_code": st.session_state.country_code})
    except Exception as e:
        st.error(f"Pipeline failed: {e}")
        out = {"answer": "[pipeline error]", "source": "error", "retrieved": [], "llm_called": False}

    # crisis case
    if out.get("crisis"):
        st.markdown(f"<div style='border:2px solid #ff4d4d; background:#fff4f4; padding:12px; border-radius:8px;'><b>Immediate support suggested (auto-detected):</b><br>{out['answer']}</div>", unsafe_allow_html=True)
        st.session_state.history.append({"role":"assistant","text": out["answer"]})
        st.experimental_rerun()

    # streaming emulation
    ans = out.get("answer", "")
    disp = ""
    for i in range(0, len(ans), 80):
        disp += ans[i:i+80]
        placeholder.markdown(f'<div class="chat-box" style="align-items:flex-start;"><div class="msg-assistant">{disp}</div></div>', unsafe_allow_html=True)
        time.sleep(0.03)
    st.session_state.history.append({"role":"assistant","text": ans})
    placeholder.empty()
    st.experimental_rerun()
