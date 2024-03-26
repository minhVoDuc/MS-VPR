import streamlit as st
import pandas as pd

def print_content(url):  
  content = open(url, 'r').read()
  st.markdown(f'<div style="text-align: justify;">{content}</div><br>', unsafe_allow_html=True)
  
@st.cache_data
def load_data_from_csv(url):
  df = pd.read_csv(url)
  return df