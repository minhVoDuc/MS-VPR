import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
from modules.interactive_map import show_map
from modules.content_page import print_content, load_data_from_csv
from model_utils import calculate_desc, load_model_musc
import cv2
import re

MODEL_PATH = './static/resnet50musc_epoch(72)_step(35697)_R1[0.5717]_R5[0.7312].ckpt'
QUERY_PATH = './static/query/'
REF_PATH = './static/ref/'
DESC_PATH = './static/descriptors.pt'
GT_PATH = './static/gt_names.npy'

st.set_page_config(  
  page_title="MS-MixVPR",
  page_icon="🔎",
  layout="wide"
)

@st.cache_resource
def load_model():
  return load_model_musc(MODEL_PATH)

ground_truth_names = np.load(GT_PATH)

model = load_model()

# Title
st.title('MS-MixVPR')
st.markdown("""> **Made by:** Duc Quach-Minh, Minh Vo-Duc  
            **Instructed by:** Mr. Anh Pham-Hoang""")
# if st.button("Refresh"):
#   # Clear values from *all* all in-memory and on-disk data caches:
#   # i.e. clear values from both square and cube
#   st.cache_data.clear()

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
  print_content('./static/content/abstract/abstract.txt')
  st.image("./static/content/abstract/MultiMixVPR.jpg", caption="Model's Architecture")

# Sec 2. Eval
with tab_eval:
  eval_types = ["Comparison", "Ablation", "Qualitative"]
  sub_tab = [None] * len(eval_types)
  sub_tab = st.tabs(eval_types)
  # Comparison
  with sub_tab[0]:
    # Display metric recallRate@N
    with st.expander("RecallRate@N"):
      print_content('./static/content/eval/comparison_recall.txt')
      # Load table R@1, R@5 and R@10
      df_recall = load_data_from_csv('./static/content/eval/compare_recall.csv')
      df_recall.set_index('Method', inplace=True)
      dataset = st.selectbox('Please choose the dataset',
                            ('Nordland', 'SPEDTest', 'MSLS-val', 'Pittsburgh'))
      # Load specific dataset
      df_recall_show = df_recall[[f"{dataset}_R@1", f"{dataset}_R@5", f"{dataset}_R@10"]]
      df_recall_show.rename(columns = lambda x: x[x.find('_')+1:], inplace = True) 
      
      col_table, col_chart = st.columns(2)
      with col_table:
        st.dataframe(data=df_recall_show.style.highlight_max(color="#FF4B4B", axis=0) \
                                              .format(precision=2), 
                    width=600)
      with col_chart:
        chart_df = load_data_from_csv(f'./static/content/eval/chart_{dataset}.csv')
        chart_df.set_index('Method', inplace=True)
        chart_df = chart_df.T
        chart_df.index = chart_df.index.map(int)
        st.line_chart(data=chart_df)
        
    # Display metric Retrieval time
    with st.expander("Time retrieval"):
      print_content('./static/content/eval/comparison_time.txt')
      df_time = load_data_from_csv('./static/content/eval/compare_time.csv')
      df_time.set_index('Method', inplace=True)
      df_time.rename(columns={"Retrieval time": "Retrieval time (ms)"})
      col_table, col_chart = st.columns(2)
      with col_table:
        st.dataframe(data=df_time.style.highlight_min(color="#FF4B4B", axis=0) \
                                              .format(precision=2), 
                    width=600)
      with col_chart:
        chart_df = df_time.reset_index()
        st.scatter_chart(data=chart_df, color="Method", x="Dimension", y="Retrieval time")
    print_content('./static/content/eval/comparison.txt')
    
  # Ablation
  with sub_tab[1]:
    with st.expander("The number of Feature-Mixer blocks and MLP ratio"):
      print_content('./static/content/eval/comparison_ratio-depth.txt')
      df_tmp = load_data_from_csv('./static/content/eval/compare_ratio-depth.csv')
      datasets = ['Nordland', 'SPEDTest']
      df_ratio = dict.fromkeys(datasets)
      cols = dict.fromkeys(datasets)
      cols['Nordland'], cols['SPEDTest'] = st.columns(2)
      for dataset in datasets:
        with cols[dataset]:
          st.text(dataset)
          df_ratio[dataset] = df_tmp[["Ratio-Depth","Parameters",f"{dataset}_R@1", f"{dataset}_R@5", f"{dataset}_R@10"]]
          df_ratio[dataset].rename(columns = lambda x: x[x.find('_')+1:] if x.find('_')>=0 else x, inplace = True)
          st.dataframe(data=df_ratio[dataset].style.highlight_max(subset=["R@1", "R@5", "R@10"], color="#FF4B4B", axis=0) \
                                              .format(precision=2), 
                      width=650, hide_index=True)
                                 
    with st.expander("Output descriptor dimension"):
      print_content('./static/content/eval/comparison_dimension.txt')
      df_tmp = load_data_from_csv('./static/content/eval/compare_dimension.csv')
      datasets = ['Nordland', 'SPEDTest']
      df_dim = dict.fromkeys(datasets)
      cols = dict.fromkeys(datasets)
      cols['Nordland'], cols['SPEDTest'] = st.columns(2)
      for dataset in datasets:
        with cols[dataset]:
          st.text(dataset)
          df_dim[dataset] = df_tmp[["Dimension","Parameters",f"{dataset}_R@1", f"{dataset}_R@5", f"{dataset}_R@10"]]
          df_dim[dataset].rename(columns = lambda x: x[x.find('_')+1:] if x.find('_')>=0 else x, inplace = True)
          st.dataframe(data=df_dim[dataset].style.highlight_max(subset=["R@1", "R@5", "R@10"], color="#FF4B4B", axis=0) \
                                              .format(precision=2), 
                      width=650, hide_index=True)
          
    with st.expander("Compare retrieval time and R@1 on Nordland dataset"):
      metrics = ['Ratio-Depth', 'Dimension']
      cols = dict.fromkeys(metrics)
      cols['Ratio-Depth'], cols['Dimension'] = st.columns(2)
      for metric in metrics:
        with cols[metric]:
          st.text(metric)
          df = load_data_from_csv(f'./static/content/eval/chart_{metric}.csv')
          df = df[[metric, "Retrieval time (ms)", "R@1"]]
          # df
          st.scatter_chart(data=df, color=metric, x="R@1", y="Retrieval time (ms)", width=300)
    
  # Qualitative
  with sub_tab[2]:
    print_content('./static/content/eval/quanlitative.txt')
    st.image("./static/content/eval/QualitativeResult.png", width=600)

# Sec 3. Interact
df = load_data_from_csv('./static/gps.csv')

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
      plot_name_holder_clicked.write("Please click on a point on the map")
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