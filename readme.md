# MS-VPR

## How to run local (on Windows)

1. Create virtual env (venv)  
   `python -m venv .venv`
2. Activate venv  
   `.venv\Scripts\activate`
3. Install required libraries  
   `pip install -r requirements.txt`
4. Prepare dataset:

- Query images: `static/query/`
- Reference images: `./static/ref/`
- Reference descriptors, ground truth, model weights: `./static/`

5. Run app  
   `streamlit run streamlit_app.py`
