import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
from streamlit_plotly_mapbox_events import plotly_mapbox_events
import plotly.express as px
from model_utils import calculate_desc, load_model_musc
import cv2

MODEL_PATH = './static/resnet50musc_epoch(72)_step(35697)_R1[0.5717]_R5[0.7312].ckpt'
QUERY_PATH = './static/query/'
REF_PATH = './static/ref/'
DESC_PATH = './static/descriptors.pt'
GT_PATH = './static/gt_names.py'

st.set_page_config(layout="wide")

@st.cache_resource
def load_model():
  return load_model_musc(MODEL_PATH)

ground_truth_names = np.load(GT_PATH)

model = load_model()

def show_map(df):
  # Create a Plotly Mapbox figure
  mapbox = px.scatter_mapbox(df, lat="latitude", lon="longitude", 
                             hover_name="timestamp", 
                             color_discrete_sequence=["#FF4B4B"], 
                             zoom=15.5, height=500)
  mapbox.update_layout(mapbox_style="carto-positron")
  mapbox.update_layout(margin={"r":0, "t":0, "l":0, "b":0})
  mapbox.update_layout(
    hoverlabel=dict(
      bgcolor="#FF4B4B",
      font_color="#FAFAFA",
      font_family="sans-serif"
    )
)

  # Create an instance of the plotly_mapbox_events component
  mapbox_events = plotly_mapbox_events(
      mapbox,
      click_event=True,
      # hover_event=True,
      override_height=500
  )
  
  return mapbox_events

# Title
st.title('MS-MixVPR')
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
  candidate = None
  col_1, col_2 = st.columns([3, 1])

  with col_1:
    mapbox_events = show_map(df)
    if st.button("Find match", type="primary"):
      if len(mapbox_events[0]) != 0:
        img_name = str(df.loc[mapbox_events[0][0]['pointIndex'], 'timestamp']) + '.png'
        # get image
        img_path = f"{QUERY_PATH}{img_name}"
        query_img = cv2.imread(img_path, 1)
        candidate = calculate_desc(model, query_img, DESC_PATH)

  with col_2:
    # Display the captured events
    st.text("Query:")
    plot_name_holder_clicked = st.empty()
    if len(mapbox_events[0]) != 0:
      img_name = str(df.loc[mapbox_events[0][0]['pointIndex'], 'timestamp']) + '.png'
      # plot_name_holder_clicked.write(f"Clicked Point: {img_name}")
      # get image
      img_path = f"{QUERY_PATH}{img_name}"
      query_img = Image.open(img_path)
      resized_query = query_img.resize((int(640/2), int(480/2)))
      st.image(resized_query)
    else:
      plot_name_holder_clicked.write("Please click a point on map")
    if candidate == None:
      st.text("Result:")
    else:
      st.text("Result:")
      ref_path = f"{REF_PATH}{ground_truth_names[candidate.item()]}.png"
      ref_img = Image.open(ref_path)
      resized_ref = ref_img.resize((int(640/2), int(480/2)))
      st.image(resized_ref)
      
with tab_demo:
  col_1, col_2 = st.columns(2)
  with col_1:
    st.text("Normal loop closure detection:")
    st.video('https://youtu.be/jzUvipfSQOw')

  with col_2:
    st.text("Long-term loop closure detection:")
    st.video("https://youtu.be/pJdo2Fkpw04")