import streamlit as st
import google.generativeai as genai
from datetime import datetime

# ----------------------------
# Page + Client
# ----------------------------
st.set_page_config(page_title="Mental Health Chat • WhatsApp Style", layout="centered")
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
# ----------------------------
# Constants
st.set_page_config(page_title="Mental Health Chat • WhatsApp Style", layout="centered")

# WhatsApp-style bubble layout (no avatars)
st.markdown("""
<style>
body {background:#ece5dd;}
.wa-bg {background-color:#ece5dd; padding:16px; border-radius:12px;}
.msg-row {display:flex; align-items:flex-end; margin-bottom:10px;}
.msg-row.right {justify-content:flex-end;}   /* user on right */
.msg-row.left {justify-content:flex-start;}  /* bot on left */
.bubble {
    display:inline-block;
    padding:10px 14px;
    border-radius:12px;
    max-width:75%;
    word-wrap:break-word;
    box-shadow:0 1px 1px rgba(0,0,0,0.15);
    color:#000000;
}
.from-user {background:#dcf8c6;}  /* light green bubble */
.from-bot  {background:#ffffff;}  /* white bubble */
.time {
    font-size:0.75rem;
    color:gray;
    margin-left:8px;
}
</style>
""", unsafe_allow_html=True)



# ----------------------------
SYSTEM_TONE = (
    "You are a supportive, non-judgmental mental health companion. "
    "Be empathetic, brief, and practical. Do NOT diagnose or prescribe. "
    "Offer gentle questions, simple coping tips (breathing, grounding, journaling), "
    "and encourage seeking professional help when appropriate."
)
CONCISE_BIAS = "Keep replies short (3–6 sentences max), use calm, simple language."

# Crisis keywords (non-exhaustive, conservative)
CRISIS_KEYWORDS = [
    "suicide","kill myself","end my life","can't go on","want to die","self harm",
    "self-harm","cut myself","overdose","jump off","hang myself","no reason to live",
    "suicidal","take my life","die by suicide"
]

# India helplines (verified widely; may change—show as guidance)
def crisis_card():
    st.error("If you’re in immediate danger, call your local emergency number (India: **112**) or a 24×7 helpline below.", icon="🚨")
    with st.expander("Trusted India helplines (tap to view)"):
        st.markdown(
            """
- **Tele-MANAS (Govt. of India)**: **14416** or **1800-891-4416** — 24×7, multilingual  
- **KIRAN (MoSJE)**: **1800-599-0019** — 24×7  
- **Vandrevala Foundation**: **+91 9999 666 555** — 24×7 (call/WhatsApp)  
- **iCALL (TISS)**: **022-2552-1111** — Mon–Sat hours; email **icall@tiss.edu**  
- **AASRA** (directory of helplines across India): https://www.aasra.info/helpline.html
            """
        )

def looks_crisis(text: str) -> bool:
    t = text.lower()
    return any(k in t for k in CRISIS_KEYWORDS)

# ----------------------------
# Model call
# ----------------------------
def call_gemini(user_msg: str, max_tokens: int = 300, compact: bool = True) -> str:
    prefix = f"{SYSTEM_TONE}\n{CONCISE_BIAS if compact else ''}\n\nUser: {user_msg}\nAssistant:"
    try:
        import google.generativeai as genai
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(
            prefix,
            generation_config={"max_output_tokens": max_tokens}
        )
        return (response.text or "").strip()
    except Exception as e:
        return f"⚠️ Error: {e}"


# ----------------------------
# Session state
# ----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []  # {"role": "user"/"assistant","content": "...","time": "HH:MM"}

def now_time():
    return datetime.now().strftime("%H:%M")

# ----------------------------
# Sidebar
# ----------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    max_tokens = st.slider("Max output tokens", 50, 1000, 250, step=50)
    compact = st.toggle("Extra compact replies", value=True)
    st.markdown("---")
    if st.button("🧹 Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ----------------------------
# Disclaimer banner
# ----------------------------
# ----------------------------
# Logo + Disclaimer (side-by-side)
# ----------------------------
# st.markdown(
#     """
#     <div style="display: flex; align-items: center; background-color: #f0f8ff;
#                 padding: 10px 16px; border-radius: 8px; margin-bottom: 10px;">
#         <img src="logo.png" width="60" style="margin-right: 12px; border-radius: 8px;">
#         <div style="color: #000000; font-size: 15px;">
#             💙 This chatbot offers supportive conversation and general coping tips.<br>
#             It is <b>not</b> a medical professional and doesn’t diagnose or provide emergency care.
#         </div>
#     </div>
#     """,
#     unsafe_allow_html=True
# )


# st.info(
#     "💙 This chatbot offers supportive conversation and general coping tips. "
#     "It is **not** a medical professional and doesn’t diagnose or provide emergency care.",
# )

import base64

# --- Load and encode logo image ---
def get_logo_base64(path="logo.png"):
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except FileNotFoundError:
        return ""

logo_base64 = get_logo_base64()

# --- Display disclaimer with embedded logo ---
st.markdown(
    f"""
    <div style="display:flex;align-items:center;background-color:#eaf6f6;
                padding:10px 14px;border-radius:10px;margin:10px auto 15px auto;
                max-width:780px;box-shadow:0 1px 3px rgba(0,0,0,0.08);">
        {'<img src="data:image/png;base64,' + logo_base64 + '" width="45" style="margin-right:10px;border-radius:8px;">' if logo_base64 else ''}
        <div style="color:#000;font-size:15px;line-height:1.4;">
            💙 <b>This chatbot offers supportive conversation and general coping tips.</b><br>
            It is <b>not</b> a medical professional and doesn’t diagnose or provide emergency care.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("<div style='margin-top:-10px;'></div>", unsafe_allow_html=True)





# ----------------------------
# Chat pane
# ----------------------------
st.markdown('<div class="wa-bg">', unsafe_allow_html=True)

for m in st.session_state.messages:
    role = m["role"]
    content = m["content"]
    time = m["time"]
    if role == "user":
        st.markdown(
            f'''
            <div class="msg-row right">
                <div class="bubble from-user">
                    {content}
                    <span class="time">{time}</span>
                </div>
            </div>
            ''',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'''
            <div class="msg-row left">
                <div class="bubble from-bot">
                    {content}
                    <span class="time">{time}</span>
                </div>
            </div>
            ''',
            unsafe_allow_html=True
        )



# for m in st.session_state.messages:
#     if m["role"] == "user":
#         st.markdown(
#             f"""
#             <div class="msg-row right">
#             <div class="avatar avatar-user">👤</div>
#                 <div class="bubble from-user">{m['content']}<span class="time">{m['time']}</span></div>
#             </div>
#             """,
#             unsafe_allow_html=True,
#         )
#     else:
#         st.markdown(
#             f"""
#             <div class="msg-row left">
#                 <div class="avatar avatar-bot">🤖</div>
#                 <div class="bubble from-bot">{m['content']}<span class="time">{m['time']}</span></div>
#             </div>
#             """,
#             unsafe_allow_html=True,
#         )

st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------
# Composer (Enter-to-send)
# ----------------------------
with st.form("composer", clear_on_submit=True):
    user_text = st.text_input("Type a message", "", label_visibility="collapsed",
                              placeholder="Share what’s on your mind…")
    send = st.form_submit_button("Send ➤", use_container_width=True)

if send and user_text.strip():
    message = user_text.strip()
    st.session_state.messages.append({"role": "user", "content": message, "time": now_time()})

    # Safety: crisis detection → show helplines before/alongside reply
    crisis_flag = looks_crisis(message)
    if crisis_flag:
        crisis_card()

    # Compose prompt with safety context (soft guidance)
    if crisis_flag:
        message = (
            "The user may be in crisis. Respond with high empathy, keep it very short, "
            "encourage reaching out to emergency services/helplines, avoid platitudes. "
            "Offer one immediate grounding step. "
            f"User message: {message}"
        )

    bot_reply = call_gemini(message, max_tokens=max_tokens, compact=compact)
    st.session_state.messages.append({"role": "assistant", "content": bot_reply, "time": now_time()})
    st.rerun()
# ----------------------------
# Fixed Emergency Helpline Footer (India)
# ----------------------------
st.markdown("""
<div class="helpline-footer">
    🚨 <b>Need immediate help?</b> Call <b>112</b> (India emergency) or 24×7 helplines:<br>
    ☎️ <b>Tele-MANAS:</b> 14416 or 1800-891-4416 &nbsp; | &nbsp;
    <b>KIRAN:</b> 1800-599-0019 &nbsp; | &nbsp;
    <b>Vandrevala Foundation:</b> +91 9999 666 555 &nbsp; | &nbsp;
    <a href="https://www.aasra.info/helpline.html" target="_blank">More helplines</a>
</div>
""", unsafe_allow_html=True)
