# README_RUN.md

1. Create & activate a virtual environment (Python 3.10+ recommended).

2. Install torch (CPU) then requirements:
   - Example (CPU wheel):
     python -m pip install --no-cache-dir "torch==2.8.0" -f https://download.pytorch.org/whl/cpu/torch_stable.html
   - Then:
     python -m pip install -r requirements.txt

3. Prepare data:
   - Put domain data into `data/` folder. Examples: `intents.json`, `combined_dataset.json`, CSVs with counseling transcripts.
   - Edit `.env` from `.env.example` and set GEMINI_API_KEY.

4. Build retrieval artifacts:
   python build_index.py

5. Run Streamlit app:
   streamlit run app_streamlit.py

Notes:
- Make sure `GEMINI_API_KEY` is set in environment or `.env`.
- If you don't have access to Gemini, you can mock `call_gemini()` in `rag_utils.py` to return a canned response for development.
