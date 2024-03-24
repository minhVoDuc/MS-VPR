# MS-VPR

## How to run local (on Windows)
1. Create virtual env (venv)  
`python -m venv .venv`
2. Activate venv  
`.venv\Scripts\activate`
3. Install required libraries  
`pip install -r requirements.txt`
4. Prepage query images and place in `/static/query_map/`
5. Run app  
`streamlit run streamlit_app.py`