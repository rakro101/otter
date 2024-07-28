import pandas as pd
from sklearn.decomposition import PCA
import seaborn as sns
import plotly.express as px
import umap
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import pdist, squareform
import json

SEED = 42
np.random.seed(SEED)

def save_dict_to_json(dictionary, file_path):
    """
    Save a Python dictionary as a JSON file.

    Parameters:
    dictionary (dict): The dictionary to save.
    file_path (str): The path to the file where the dictionary should be saved.
    """
    with open(file_path, 'w') as json_file:
        json.dump(dictionary, json_file, indent=4)

def pad_ccm_matrix(ccm_matrix, original_nodes, all_nodes):
    """
    Pad a smaller M x M ccm_matrix into a larger N x N matrix, keeping entries at the correct positions.

    Parameters:
    ccm_matrix (numpy.ndarray): The smaller M x M matrix to be padded.
    original_nodes (list): The list of M nodes corresponding to the ccm_matrix.
    all_nodes (list): The list of N nodes corresponding to the desired larger matrix.

    Returns:
    numpy.ndarray: The padded N x N matrix.
    """
    # Create a mapping from node to its index in the all_nodes list
    node_index_map = {node: idx for idx, node in enumerate(all_nodes)}

    # Initialize a larger N x N matrix with zeros
    N = len(all_nodes)
    padded_matrix = np.zeros((N, N))

    # Place the values of the ccm_matrix into the correct positions in the padded_matrix
    for i, node_i in enumerate(original_nodes):
        for j, node_j in enumerate(original_nodes):
            padded_matrix[node_index_map[node_i], node_index_map[node_j]] = ccm_matrix[i, j]

    return padded_matrix

def calculate_fourier_coefficients(series, num_coefficients):
    """ Calculate Fourier Coefficients from the time series"""
    print(series.shape)
    fourier_transform = np.fft.fft(series)
    coefficients = fourier_transform[1:num_coefficients]  # Select the desired number of coefficients
    ret = np.concatenate([np.real(coefficients), np.imag(coefficients)], axis=0)
    return ret

def normalize(df_spec, hellinger):
    """ Normalize abundance table """
    if hellinger == True:
        df_sqrt = np.sqrt(df_spec)
        row_norms = np.linalg.norm(df_sqrt, axis=1)
        df_normalized = df_sqrt.div(row_norms, axis=0)
        df_spec = df_normalized
    return df_spec

def plot_distance_matrix_unpruned(dist_map,save_path_temp = f'Tutorial/figures/Sup_Figure_S5_Distance_ALL_Connections_heatmap_ALL.png'):
    plt.figure(figsize=(8, 6))
    pivot_table_mask = (dist_map == 0.00)
    sns.clustermap(dist_map, annot=True, cmap='coolwarm', fmt='.0f', cbar=True, square=True, linewidths=1,
                   mask=pivot_table_mask, col_cluster=True, row_cluster=True, )

    sns.set_style('white')
    plt.savefig(save_path_temp, dpi=200, bbox_inches='tight')
    plt.show()
    return plt.gcf()

def plot_distance_matrix_pruned(result, save_path_temp = f'Tutorial/figures/Sup_Figure_S6_Distance_Connections_heatmap_ALL.png'):
    plt.figure(figsize=(8, 6))
    pivot_table_mask = (result == 0.00)
    sns.clustermap(result, annot=True, cmap='coolwarm', fmt='.0f', cbar=True, square=True, linewidths=1,
                   mask=pivot_table_mask, col_cluster=False, row_cluster=False)

    sns.set_style('white')
    plt.savefig(save_path_temp, dpi=200, bbox_inches='tight')
    plt.show()
    return plt.gcf()

def plot_2d_umap_embedding(U_embedding, meta, save_path="Tutorial/figures/PCA_Umap2_seaborn.png"):
    sns.set(style="whitegrid")
    # Create a scatter plot using seaborn
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=U_embedding[:, 0], y=U_embedding[:, 1], hue=meta["clu"].values,
                    palette=meta["colors"].unique().tolist(), s=50)

    # Customize the plot further if needed
    plt.xlabel('Umap x')
    plt.ylabel('Umap y')
    plt.legend(title='Colors', loc='upper right')

    # Save the plot
    plt.savefig(save_path)

    # Show the plot
    plt.show()

def plot_3d_umap_embedding(V_embedding, meta, centroids, save_path = "Tutorial/figures/Sup_Figure_S4_PCA_Umap3d_seaborn.png"):
    fig = px.scatter_3d(x=V_embedding[:, 0], y=V_embedding[:, 1], z=V_embedding[:, 2],
                        color_discrete_sequence=meta["colors"].unique(), color=meta["clu"].values, opacity=0.5)
    centroids_trace = px.scatter_3d(x=centroids["x_3d"], y=centroids["y_3d"], z=centroids["z_3d"])
    fig.add_trace(centroids_trace.update_traces(
        marker=dict(color=meta["colors"].unique().tolist(), symbol="x", line=dict(color='black', width=1000, )),
        name='Centroids').data[0])
    fig.update_layout(scene=dict(aspectmode="data"), height=1000, width=1000)
    fig.write_image(save_path)
    #fig.show()
    return fig
    
def apply_pca(X, n_components=10):
    """
    Apply PCA to
    :param X: 
    :param n_components: 
    :return: 
    """
    pca = PCA(n_components=n_components, random_state=SEED)
    pca_data = pca.fit_transform(X)
    return pca_data

def apply_umap(pca_data, n_components=2, scaled=False):
    """
    Apply UMAP to PCA abundance data
    :param pca_data:
    :param n_components:
    :param scaled:
    :return:
    """
    reducer = umap.UMAP(random_state=SEED, n_components=n_components)
    if scaled:
        scaled_data= StandardScaler().fit_transform(pca_data)
    else:
        scaled_data = pca_data
    U_embedding = reducer.fit_transform(scaled_data)
    return U_embedding

def calculate_distance_matrix(df_fft_spec,df_ccm, mfd, save_path ="Tutorial/tables/Main_Figure_S3_D_table.csv"):
    """
    Calculate distance matrix, prunded (by ccm connections) and prunded
    :param df_fft_spec:
    :param df_ccm:
    :param mfd:
    :return:
    """
    # centroids on 3 d umap embedding
    umap3d_matrix = df_fft_spec[["x_3d", "y_3d", "z_3d", "clu"]]
    centroids = umap3d_matrix.groupby('clu').mean()
    print(centroids)
    # Calculate the pairwise distances
    distance_vector = pdist(centroids)
    # Convert the distance vector to a square distance matrix
    distance_matrix = squareform(distance_vector)
    # Display the distance matrix
    distance_matrix_rounded = np.round(distance_matrix, 0)
    print("??????" * 10)
    print(centroids.index.tolist())
    print("?????" * 10)
    df_ccm["from_clu"] = df_ccm["from"].map(mfd)
    df_ccm["to_clu"] = df_ccm["to"].map(mfd)
    matrix_3b = df_ccm[["from_clu", "to_clu", "corr"]].groupby(["to_clu", "from_clu"]).agg(np.mean).reset_index()
    # Create a matrix using pivot
    matrix_pivot = matrix_3b.pivot(index='to_clu', columns='from_clu', values='corr')
    print("+++++++" * 10)
    print(matrix_pivot)
    print("+++++++" * 10)
    matrix_pivot.to_csv(save_path, sep=";")

    ccm_matrix = matrix_pivot
    ccm_matrix = ccm_matrix.fillna(0)
    ccm_matrix = pd.DataFrame(ccm_matrix)
    print("ppp"*17)
    print(ccm_matrix)
    ccm_matrix.rename(columns={x: str(x) for x in ccm_matrix.columns.tolist()}, inplace=True)
    ccm_matrix["new_index"] = ccm_matrix.index
    ccm_matrix["new_index"] = ccm_matrix["new_index"].apply(lambda x: str(x))
    ccm_matrix["new_index"].astype("str")
    ccm_matrix.set_index("new_index", inplace=True)
    print("ppp" * 17)
    mask_for_connections = ccm_matrix[ccm_matrix == 0]
    mask_for_connections = mask_for_connections.fillna(1)

    df_distance_matrix_rounded = pd.DataFrame(distance_matrix_rounded)
    no_masking = df_distance_matrix_rounded[df_distance_matrix_rounded != df_distance_matrix_rounded]
    print(no_masking)
    no_masking = no_masking.fillna(1)

    # All distances
    print(no_masking.shape)
    print(no_masking)
    print(distance_matrix_rounded.shape)
    print(distance_matrix_rounded)
    dist_map = np.multiply(no_masking, distance_matrix_rounded)

    # distance masked with CCMNs
    # ToDo: Add padding for CCMN matrix to replace missing with 0.
    orignal_nodes =ccm_matrix.columns.tolist()
    #
    all_nodes = centroids.index.tolist()
    print(orignal_nodes)
    print(all_nodes)

    mask_for_connections = pad_ccm_matrix(mask_for_connections.values, original_nodes=orignal_nodes, all_nodes=all_nodes)
    print("+++++++"*10)
    print(mask_for_connections)
    print("+++++++" * 10)

    masked_dist_map = np.multiply(mask_for_connections, distance_matrix_rounded)
    return dist_map, masked_dist_map, centroids

def fourier_transformed_spec(df_spec, list_off_con, clu_dict, pca_n_components=10, save_path = "Tutorial/tables/Tutorial_latentspacelatent_space.csv",num_coefficients=14):
    df_spec_T = df_spec.T
    df_fft_spec = df_spec_T.apply(lambda row: calculate_fourier_coefficients(row, num_coefficients=num_coefficients), axis=1, result_type='expand')
    # Create column names for the FFT components
    fft_column_names = [f'fft_component_{i + 1}' for i in range(26)]
    # Rename the columns in the new DataFrame
    df_fft_spec.columns = fft_column_names
    df_fft_spec = df_fft_spec.T[list_off_con].T
    #### PCA
    X = df_fft_spec.copy()
    pca_data = apply_pca(X, n_components=pca_n_components)
    # UMAP
    U_embedding = apply_umap(pca_data, n_components=2, scaled=False)
    df_fft_spec["x_2d"] = U_embedding[:, 0]
    df_fft_spec["y_2d"] = U_embedding[:, 1]
    V_embedding = apply_umap(pca_data, n_components=3, scaled=False)
    df_fft_spec["x_3d"] = V_embedding[:, 0]
    df_fft_spec["y_3d"] = V_embedding[:, 1]
    df_fft_spec["z_3d"] = V_embedding[:, 2]
    df_fft_spec.to_csv(save_path, sep=";")
    df_fft_spec["Nodes"] = df_fft_spec.index
    df_fft_spec["clu"] = df_fft_spec["Nodes"].apply(lambda x: clu_dict[x])
    df_fft_spec = df_fft_spec[df_fft_spec["clu"] != "nan"]
    return df_fft_spec, U_embedding, V_embedding

def main_embeddings(df_spec,meta, df_ccm, hellinger=True, num_coefficients=14, save_pre_fig ="Tutorial/figures", save_pre_tab ="Tutorial/tables"):
    df_spec = normalize(df_spec, hellinger=hellinger)
    meta = meta[~meta["LouvainLabelD"].isna()]
    meta["clu"] = meta["LouvainLabelD"]
    meta["clu"] = meta["clu"].apply(lambda x: str(x))
    meta["colors"] = meta["ColorLabel"]
    meta = meta.sort_values(by='clu')

    node_label = meta[["Nodes", "clu"]]
    node_label.set_index("Nodes", inplace=True)
    clu_dict = node_label.to_dict()["clu"]

    list_off_con = meta["Nodes"]
    meta_file_dict = meta[["Nodes", "LouvainLabelD"]]
    meta_file_dict.set_index("Nodes", inplace=True)
    meta_file_dict = meta_file_dict.to_dict()["LouvainLabelD"]

    df_fft_spec, U_embedding, V_embedding = fourier_transformed_spec(df_spec, list_off_con, clu_dict,
                                                                     pca_n_components=10,
                                                                     num_coefficients = num_coefficients,
                                                                     save_path=f"{save_pre_tab}/Tutorial_latentspacelatent_space.csv")

    dist_map, masked_dist_map, centroids = calculate_distance_matrix(df_fft_spec, df_ccm, meta_file_dict)

    plot_2d_umap_embedding(U_embedding, meta, save_path=f"{save_pre_fig}/PCA_Umap2_seaborn.png")

    plot_3d_umap = plot_3d_umap_embedding(V_embedding, meta, centroids,
                           save_path=f"{save_pre_fig}/Sup_Figure_S4_PCA_Umap3d_seaborn.png")

    distance_matrix_pruned = plot_distance_matrix_pruned(masked_dist_map,
                                                         save_path_temp=f"{save_pre_fig}/Pruned_Distance_Connections_heatmap_ALL.png")

    distance_matrix_unpruned =plot_distance_matrix_unpruned(dist_map,
                                  save_path_temp=f"{save_pre_fig}/Unpruned_Distance_ALL_Connections_heatmap_ALL.png")

    print("Cluster distribution:")
    print(df_fft_spec["clu"].value_counts())
    print(centroids)
    return plot_3d_umap, distance_matrix_unpruned, distance_matrix_pruned

if __name__ == "__main__":
    ABUNDANCES_FILE = "BeyondBlooms2024/data/F4_euk_abundance_table.csv"
    ENRICH = "BeyondBlooms2024/tables/Enriched_Paper_Meta.csv"
    PRUNED_PVAL_CCMN_PATH = "BeyondBlooms2024/tables/BeyondBlooms2024_Hellinger_True_14_Pruned_CCM_CON_MAP_Network.csv"
    num_coefficients = 14
    df_spec = pd.read_csv(ABUNDANCES_FILE, sep=";", index_col=0)
    df_ccm = pd.read_csv(PRUNED_PVAL_CCMN_PATH, sep=";")
    meta = pd.read_csv(ENRICH, sep=",")
    main_embeddings(df_spec,meta, df_ccm, hellinger=True, num_coefficients = num_coefficients)









