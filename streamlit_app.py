import streamlit as st
import numpy as np
import pandas as pd
from modules.interactive_map import show_map

st.set_page_config(layout="wide")

# Title
st.title('MS-VPR')
st.markdown("""> **Made by:** Duc Quach-Minh, Minh Vo-Duc  
            **Instructed by:** Mr. Anh Pham-Hoang""")

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

with tab_interact:
  col_1, col_2 = st.columns(2)  
  with col_1:
    mapbox_events = show_map(df)
  
  with col_2:
    # Display the captured events
    plot_name_holder_clicked = st.empty()
    plot_name_holder_clicked.write("Please click a point on map")
    if len(mapbox_events[0]) != 0:
      img_name = str(df.loc[mapbox_events[0][0]['pointIndex'], 'timestamp']) + '.png'
      plot_name_holder_clicked.write(f"Clicked Point: {img_name}")
      # get image
      st.image(f"./static/query_map/{img_name}", caption=img_name)