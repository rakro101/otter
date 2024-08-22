import numpy as np
import plotly.express as px
import scipy.optimize as opt
import streamlit as st
from scipy import interpolate

st.set_page_config(page_title="Visualisation", page_icon="📈", layout="wide")

st.markdown("# Visualisation")
#st.sidebar.header("Visualisation")
st.write(
    """This demo illustrates a combination of plotting and animation with
Streamlit. We're generating a bunch of random numbers in a loop for around
5 seconds. Enjoy!"""
)

abundance_df = st.session_state.df1
meta_df = st.session_state.df_enrich



TimeLen = abundance_df.shape[0]
st.dataframe(abundance_df)
TimeCols = [ent for ent in abundance_df.T.columns.tolist() if ent != "Unnamed: 0"]

abundance_df = abundance_df.T
st.dataframe(meta_df)

# User input widgets
aggregation_method = st.selectbox("Select aggregation method", ["Sum", "Mean"])
# cluster_id = st.text_input("Enter cluster ID")

clusters = meta_df.copy()
if "Nodes" in clusters.columns:
    clusters.set_index("Nodes",inplace=True)
clusters = clusters["LouvainLabelD"]
clu_dict = clusters.to_dict()

# Data processing
if aggregation_method == "Sum":
    print(abundance_df.head())
    abundance_df["LouvainLabelD"] = abundance_df.index.map(clu_dict)
    # print(abundance_df[KEY].tolist())
    print(abundance_df["LouvainLabelD"].head())
    aggregated_values = abundance_df.groupby("LouvainLabelD").agg(np.sum)
    print("Aggregated values:+####################################")
    print(aggregated_values.head())
else:
    abundance_df["LouvainLabelD"] = abundance_df.index.map(clu_dict)
    aggregated_values = abundance_df.groupby("LouvainLabelD").agg(np.mean)
    # print(aggregated_values)

aggregated_values["LouvainLabelD"] = aggregated_values.index

# Sinusoidal fitting
def sinusoidal(x, A=1, B=2, C=2, D=2):
    return A * np.sin(B * x + C) + D

# Sinusoidal fitting for each unique Louvain label
unique_labels = meta_df["LouvainLabelD"].unique()
unique_labels = [label for label in unique_labels if str(label)!="nan"]
print("unique_labels=", unique_labels)
fig = px.line(title="Cluster Abundance with Sinusoidal Fit")
print("ooooooooooooooooooooo")
print(aggregated_values)

for label in unique_labels:
    label_values = np.array(
        aggregated_values[aggregated_values["LouvainLabelD"] == label].values
    )
    print("label values: ", label_values)
    print(label_values)
    label_values = np.reshape(label_values, (TimeLen+1,))
    x_values = np.arange(TimeLen+1)  # More data points
    x_interp = np.linspace(0, TimeLen, 10000)
    y_interp = interpolate.interp1d(x_values, label_values, kind="slinear")(x_interp)
    # x_values = np.arange(23)

    print("########++++++++++++")
    print(label_values)
    # popt, _ = opt.curve_fit(sinusoidal, x_values, label_values)
    popt, _ = opt.curve_fit(sinusoidal, x_values, label_values, p0=[1, 2, 2, 2], maxfev=10000)
    fig.add_scatter(
        x=x_values,
        y=sinusoidal(x_values, *popt),
        mode="lines",
        name=f"Cluster {label}",
    )

st.plotly_chart(fig, use_container_width = True)
# Save plot
save_path = "./cache/season_fit.jpeg"
fig.write_image(save_path, format="jpeg")

# Display download button for the saved image
with open(save_path, "rb") as file:
    btn = st.download_button(
        label="Download image",
        data=file,
        file_name="season_fit.jpeg",
        mime="image/jpeg",
    )


