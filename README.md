# 🧠 RAG Chatbot (Gemini + SBERT + FAISS + Streamlit)

This project implements a **Retrieval-Augmented Generation (RAG)** chatbot with:
- SBERT sentence embeddings  
- FAISS vector database  
- Gemini API for LLM responses  
- Crisis message detection  
- WhatsApp-style UI (Streamlit)

This guide explains **how to set up, build, and run the project from start** using **VS Code**.

---

## 🚀 1. Clone or Open the Project in VS Code

Open VS Code → **File → Open Folder** → select the project folder.

---

## 🐍 2. Create and Activate Virtual Environment (Recommended)

### Windows
```bash
python -m venv venv
venv\Scripts\activate
```
### Install Dependencies
## Install PyTorch CPU
```bash
python -m pip install --no-cache-dir "torch==2.8.0" -f https://download.pytorch.org/whl/cpu/torch_stable.html
```
### Install requirements
```bash
pip install -r requirements.txt
```

###Add Gemini API Key
```bash
GEMINI_API_KEY=your_api_key_here
GEMINI_MODEL=models/gemini-2.0-pro
SBERT_MODEL=all-mpnet-base-v2

```

###Build the FAISS Vector Index (Required First Time)
```bash
python build_index.py
```
###Run the Streamlit Chat App
```bash
streamlit run app_streamlit.py

```

###streamlit link

https://mental-health-chatbot-bishnu.streamlit.app/
