import streamlit as st
import numpy as np
import pandas as pd
from modules.interactive_map import create_map

# Title
st.title('MS-VPR')
# st.divider()

# Setting tabs
st.markdown("""
<style>
	.stTabs [data-baseweb="tab"] {
		border-radius: 4px 4px 0px 0px;
		padding: 10px;
    font-size: 2em;
  }
</style>""", unsafe_allow_html=True)
tab_abstract, tab_eval, tab_interact, tab_demo = st.tabs(["ABSTRACT", "EVALUATION", "INTERACT", "DEMO"])

# Sec 1. Abstract
with tab_abstract:
  content = open('./static/abstract.txt', 'r').read()
  # st.write(f"""{content}""")
  st.markdown(f'<div style="text-align: justify;">{content}</div><br>', unsafe_allow_html=True)
  st.image("./static/MultiMixVPR.jpg", caption="Model's Architecture")

# Sec 2. Eval


# Sec 3. Interact
df = pd.read_csv('./static/gps.csv')
r = create_map(df)

with tab_interact:
  # df
  # st.map(data=df, size=1)
  st.pydeck_chart(r)