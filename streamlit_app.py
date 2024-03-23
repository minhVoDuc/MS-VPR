import streamlit as st
import numpy as np
import pandas as pd
import time

# Title
st.title('MS-VPR')
st.divider()

# Sec 1. Abstract
st.header('Abstract')
content = open('./static/abstract.txt', 'r').read()
# st.write(f"""{content}""")
st.markdown(f'<div style="text-align: justify;">{content}</div>', unsafe_allow_html=True)
st.image("./static/MultiMixVPR.jpg", caption="Model's Architecture")
