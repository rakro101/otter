import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Enviromental Data", page_icon="📈", layout="wide")

st.markdown("# Enviromental Data")
st.sidebar.header("Enviromental Data")
st.write(
    """This demo illustrates a combination of plotting and animation with
Streamlit. We're generating a bunch of random numbers in a loop for around
5 seconds. Enjoy!"""
)

# Upload metadata CSV
df = st.session_state.df2
st.dataframe(df)
df.index = pd.to_datetime(df.index)
df.sort_index(inplace=True)

# Checkbox selection for columns
selected_columns = []
for col in df.columns:
    if st.checkbox(col):
        selected_columns.append(col)

if st.button("Plot Environmental Parameters",help="Calculate Cluster Centroids"):
    if len(selected_columns) > 1:
        # Plot selected columns
        fig, axs = plt.subplots(
            len(selected_columns), 1, figsize=(16, 6 * len(selected_columns)), sharex="col"
        )
        for i, col in enumerate(selected_columns):
            axs[i].plot(df.index, df[col], color="darkblue", linewidth=2)
            axs[i].set_ylabel(col, fontweight="bold")
            axs[i].grid(True)
            axs[i].xaxis.set_major_locator(mdates.MonthLocator(interval=4))
            axs[i].xaxis.set_major_formatter(mdates.DateFormatter("%m.%y"))
            axs[i].set_facecolor("white")
            axs[i].grid(color="black", linestyle="-", linewidth=0.5)
            axs[i].locator_params(axis="y", nbins=5)

        plt.tight_layout(pad=1.1, h_pad=0.85, w_pad=0.85)
        st.pyplot(fig)

        # Save plot
        save_path = "./cache/environment.png"
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

        # Display download button for the saved image
        with open("./cache/environment.png", "rb") as file:
            btn = st.download_button(
                label="Download image",
                data=file,
                file_name="environment2.png",
                mime="image/png",
            )
    else:
        st.warning("Select at least 2 Env parameters.")
