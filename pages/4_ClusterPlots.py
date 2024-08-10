import time

import numpy as np
import streamlit as st
from scipy import interpolate

st.set_page_config(page_title="Cluster Density Plot", page_icon="📈", layout="wide")

st.markdown("# Cluster Density Plot")
st.sidebar.header("Cluster Density Plot")
st.write(
    """This demo illustrates a combination of plotting and animation with
Streamlit. We're generating a bunch of random numbers in a loop for around
5 seconds. Enjoy!"""
)
import numpy as np
import pandas as pd
import plotly.express as px
import scipy.optimize as opt
import streamlit as st


# Load your CSV files (replace with actual file paths)
# @st.cache
abundance_df = st.session_state.df1
abundance_df = abundance_df.T
#st.dataframe(abundance_df)

meta_df = st.session_state.df_enrich
#st.dataframe(meta_df)

# User input widgets
aggregation_method = st.selectbox("Select aggregation method", ["Sum", "Mean"])
# cluster_id = st.text_input("Enter cluster ID")

clusters = meta_df.copy()
# print(clusters.head())
if "Nodes" in clusters.columns:
    clusters.set_index("Nodes",inplace=True)
clusters = clusters["LouvainLabelD"]
clu_dict = clusters.to_dict()
print(clu_dict)
# Data processing
if aggregation_method == "Sum":
    abundance_df["LouvainLabelD"] = abundance_df.index.map(clu_dict)
    aggregated_values = abundance_df.groupby("LouvainLabelD").agg(np.sum)
    st.dataframe(aggregated_values)
else:
    abundance_df["LouvainLabelD"] = abundance_df.index.map(clu_dict)
    aggregated_values = abundance_df.groupby("LouvainLabelD").agg(np.mean)
    st.dataframe(aggregated_values)

aggregated_values["LouvainLabelD"] = aggregated_values.index
unique_labels = meta_df["LouvainLabelD"].unique()
fig = px.line(title="Cluster Abundance with Sinusoidal Fit")

print(aggregated_values.head())
aggregated_values.drop("LouvainLabelD", axis=1, inplace=True)
fig = px.area(aggregated_values.T)
# Save plot
save_path = "./cache/cluster_plot.jpeg"
fig.write_image(save_path, format="jpeg")

# Display download button for the saved image
with open(save_path, "rb") as file:
    btn = st.download_button(
        label="Download image",
        data=file,
        file_name="Cluster_Plot.jpeg",
        mime="image/png",
    )
st.plotly_chart(fig,use_container_width = True)




