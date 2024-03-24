from streamlit_plotly_mapbox_events import plotly_mapbox_events
import plotly.express as px
import pandas as pd

def show_map(df):
  # Create a Plotly Mapbox figure
  mapbox = px.scatter_mapbox(df, lat="latitude", lon="longitude", 
                             hover_name="timestamp", 
                             color_discrete_sequence=["#FF4B4B"], 
                             zoom=15.5, height=600)
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
      override_height=600
  )
  
  return mapbox_events
