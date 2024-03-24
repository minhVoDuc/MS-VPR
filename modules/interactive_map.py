import pydeck as pdk
import pandas as pd
import os

def convert(x):
  return str(x)

def create_map(df):
  # QUERY_MAP = (
  #   data_url
  # )
  dirs = df['timestamp'].apply(convert)
  df_transform = pd.DataFrame().assign(dir=dirs, lat=df['latitude'], lng=df['longitude'])
  layer = pdk.Layer(
      "ScatterplotLayer",
      df_transform,
      get_position=['lng', 'lat'],
      auto_highlight=True,
      pickable=True,
      get_radius=2.5,
      get_fill_color=[180, 0, 200, 140]
  )
  # Set the viewport location
  view = pdk.ViewState(
    longitude=-1.2585, latitude=51.759, zoom=15.75, min_zoom=14, max_zoom=20, pitch=40.5, bearing=-27.36
  )
  # view = pdk.data_utils.compute_view(df_transform[["lng", "lat"]])
  # view.zoom = 15.75
  # view.min_zoom = 14
  # view.max_zoom = 20
  # view.pitch = 40.5
  # view.bearing = -27.36
  
  # Combined all of it and render a viewport
  r = pdk.Deck(
      map_style="mapbox://styles/mapbox/light-v9",
      layers=[layer],
      initial_view_state=view,
      # tooltip={"html": "<img src='../static/query/{dir}.png' width='320' height='240'>"},
      tooltip={"html": "<b>Query:</b> {dir}.png", "style": {"color": "white"}},
  )
  return r

