import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from scipy.cluster import hierarchy
import seaborn as sns
from matplotlib import pyplot as plt

st.set_page_config(
    page_title="Enviromental Heatmap Data", page_icon="📈", layout="wide"
)
import numpy as np

st.markdown("# Enviromental Heat Map Data")
st.sidebar.header("Enviromental Heat Map Data")
st.write(
    """This demo illustrates a combination of plotting and animation with
Streamlit. We're generating a bunch of random numbers in a loop for around
5 seconds. Enjoy!"""
)

# Create sub-containers for file uploads
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.title("Abundance")
    df1 = st.session_state.df1
    st.dataframe(df1)


with col2:
    st.title("Environment")
    df2= st.session_state.df2
    st.dataframe(df2)

    with col4:
        st.title("Columns")
        # Checkbox selection for columns
        selected_columns = []
        for col in df2.columns:
            if st.checkbox(col):
                selected_columns.append(col)



with col3:
    st.title("Enrich")
    df3= st.session_state.df_enrich
    if "Nodes" in df3.columns:
        df3.set_index("Nodes", inplace=True)
    st.dataframe(df3)

df = df2
df_taxa = df3
df_abundance = df1

df_env_F4 = df
selected_asv = df_taxa.index.tolist()

def get_dict(col="Class"):
    df_taxa_temp = df_taxa[[col]]
    #df_taxa_temp.set_index("Nodes", inplace=True)
    d_ = df_taxa_temp.to_dict()[col]
    return d_

dff = []
df_taxa["cluster_names"] = df_taxa["LouvainLabelD"]
l_cus = df_taxa["cluster_names"].unique().tolist()
l_cus.sort()
for clu in l_cus:
    if f"{clu}" != "nan":
        temp = df_taxa[df_taxa["cluster_names"] == clu]
        temp_asv = temp.index.tolist()
        print(temp_asv)
        temp_ab = df_abundance[temp_asv]
        temp_ab = temp_ab.sum(axis=1).reset_index()
        # print(temp_ab.columns)
        temp_ab.rename(columns={"Unnamed: 0": "date", 0: f"{clu}"}, inplace=True)
        print("+++++++++")
        print(temp_ab.columns)
        try:
            temp_ab.set_index("index", inplace=True)
        except:
            temp_ab.set_index("date", inplace=True)
            pass
        # print(temp_ab.head())
        dff.append(temp_ab)
df_clus = pd.concat(dff, axis=1)
clu_cols = df_clus.columns.tolist()

cols = [
    "MLD",
    "PAR",
    "temp",
    "sal",
    "PW_frac",
    "O2_conc",
    "depth",
]  # df_env_F4.columns#
cols = selected_columns
# corr with only one mooring ASV
#df_env_F4.rename(columns={"PAR_satellite": "PAR"}, inplace=True)
df_env = df_env_F4[cols]

window_size = 8  # You can adjust this as needed
df_env = df_env.reset_index()
try:
    for i in range(len(df_env)):
        if pd.isna(df_env.at[i, "O2_conc"]):
            # Calculate the rolling mean for the neighborhood around the NaN value
            start = max(0, i - (window_size // 2))
            end = min(len(df_env), i + (window_size // 2) + 1)
            neighborhood_mean = df_env["O2_conc"][start:end].mean()

            # Replace NaN with the neighborhood mean
            df_env.at[i, "O2_conc"] = neighborhood_mean
except Exception as e:
    print("no Column named O2_conc")

try:
    if "date" in df_env.columns:
        df_env.set_index("date", inplace=True)
    df_env.index = pd.to_datetime(df_env.index)
except Exception as e:
    if "date_" in df_env.columns:
        df_env.set_index("date_", inplace=True)
    df_env.index = pd.to_datetime(df_env.index)
    print(e)

#st.dataframe(df_env)
#st.dataframe(df_clus)
df_clus.index = pd.to_datetime(df_clus.index)
df_corr_gen = df_clus.join(df_env, how="inner")  # only the 94 events (here 93!)
#st.dataframe(df_corr_gen)

correlation_gen = df_corr_gen.corr(method="pearson")
target_df_gen = (
    correlation_gen[cols]
    .T[[cll for cll in df_clus.columns if cll not in ["index"]]]
    .T
)
a = target_df_gen.index.tolist()
a.sort()
try:
    df_corr = target_df_gen.T.drop("nan", axis=1).T
except Exception as e:
    df_corr = target_df_gen.T.T
    print(e)

SMALL_SIZE = 20
MEDIUM_SIZE = 20
BIGGER_SIZE = 20
MEDIUM = 20
TICK = 20
I_SIZE = 8
linewidth = 4
plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc("axes", labelsize=MEDIUM)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=TICK)  # fontsize of the tick labels
plt.rc("ytick", labelsize=TICK)  # fontsize of the tick labels
plt.rc("legend", fontsize=MEDIUM)  # legend fontsize
plt.rc("figure", titlesize=BIGGER_SIZE)


# Reshape the data for heatmap
heatmap_data = df_corr.fillna(0)

# Logarithmic normalization
heatmap_data_normalized = heatmap_data

# Create the heatmap
plt.figure(figsize=(16, 16))
sns.heatmap(
    heatmap_data_normalized, cmap="coolwarm", cbar=True, annot=True, linewidths=1.5
)

plt.xlabel("Environment parameter")
plt.ylabel("Cluster")
save_path_temp = f"./cache/000-Env_heatmap_Sec_Filter.png"
plt.savefig(save_path_temp, dpi=200, bbox_inches='tight')

col_plot1 , col_plot2  = st.columns(2)

with col_plot1:
    st.pyplot(plt)

# Display download button for the saved image
with open(save_path_temp, "rb") as file:
    btn = st.download_button(
        label="Download image 2",
        data=file,
        file_name="000-Env_heatmap_Sec_Filter.jpeg",
        mime="image/png",
    )


import scipy.stats as stats

gen_col = [clu for clu in clu_cols if clu != "nan"]
print(clu_cols)
env_col = cols #['MLD', 'PAR', 'temp', "sal", "PW_frac", "O2_conc", "depth"]
env_no = len(env_col)
gen_no = len(gen_col)
m_1 = np.zeros((gen_no, env_no))
m = np.zeros((gen_no, env_no))
df_corr_gen.fillna(0,inplace=True)


for i in range(0, gen_no):
    for j in range(0, env_no):
        cors, p_values = stats.pearsonr(df_corr_gen[gen_col[i]], df_corr_gen[env_col[j]])
        m[i, j] = p_values
        m_1[i, j] = cors
p_values = m
cor_values = m_1
mask = (p_values > 0.05)
df_p_values = pd.DataFrame(p_values)
index_val = gen_col
import pandas as pd

SMALL_SIZE = 20
MEDIUM_SIZE = 20
BIGGER_SIZE = 20
MEDIUM = 20
TICK = 20
I_SIZE = 8
linewidth = 4
plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=TICK)  # fontsize of the tick labels
plt.rc('ytick', labelsize=TICK)  # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM)  # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)
import seaborn as sns
import matplotlib.pyplot as plt
# sns.set_theme(style='white')
from scipy.cluster import hierarchy

# Reshape the data for heatmap
plt.figure(figsize=(20, 30))
heatmap_data = df_corr.fillna(0)
sns.set_style('white')
# Logarithmic normalization

clustermap = sns.clustermap(heatmap_data, cmap='coolwarm', annot=True, linewidths=2.5, col_cluster=False,
                            row_cluster=True, mask=mask, rasterized=False, cbar_kws={'drawedges': False})

# print(clustermap.dendrogram_row.reordered_ind)
# print(clustermap.dendrogram_col.reordered_ind)
colorbar_ax = clustermap.cax
colorbar_ax.grid(False)
clustermap.ax_heatmap.set(ylabel="Cluster", xlabel="Environment Parameter")
plt.grid(False)

# Reorder the rows based on the desired order
# mask = mask[clustermap.dendrogram_row.reordered_ind]

save_path_temp = f'./cache/000-Significants_Env_Dendogram.png'
plt.savefig(save_path_temp, dpi=600, bbox_inches='tight')
#fig.write_image(save_path_temp, format="jpeg")


with col_plot2:
    st.pyplot(plt)


# Display download button for the saved image
with open(save_path_temp, "rb") as file:
    btn = st.download_button(
        label="Download image",
        data=file,
        file_name="000-Significants_Env_Dendogram.jpeg",
        mime="image/png",
    )

