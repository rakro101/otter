import pandas as pd
import networkx as nx
import numpy as np

np.random.seed(42)
from lutra.config import color_month_dict, color_label_dict


def calc_centrality(df_search_all: pd.DataFrame):
    """

    :param df_search_all:
    :return:
    """
    df_temp_clu = df_search_all
    df_temp_clu["1-corr"] = 1 - df_temp_clu["corr"]
    G = nx.from_pandas_edgelist(
        df_temp_clu, "from", "to", ["1-corr"], create_using=nx.DiGraph()
    )
    # Compute betweenness centrality for each node
    betweenness = nx.betweenness_centrality(G, weight="1-corr")

    # Compute closeness centrality for each node
    closeness = nx.closeness_centrality(G, distance="1-corr")

    # Display the betweenness centrality and closeness centrality for each node
    list_of_nodes = []
    list_of_BC = []
    list_of_CC = []
    for node in G.nodes():
        # print(f"Node {node}:")
        # print(f"Betweenness Centrality: {betweenness[node]}")
        # print(f"Closeness Centrality: {closeness[node]}")
        # print("------------------------")
        list_of_nodes.append(node)
        list_of_BC.append(betweenness[node])
        list_of_CC.append(closeness[node])
    df_03 = pd.DataFrame(list_of_nodes)
    df_03["Betweenness_Centrality"] = list_of_BC
    df_03["Closeness Centrality"] = list_of_CC
    bc_dict = dict(zip(list_of_nodes, list_of_BC))
    cc_dict = dict(zip(list_of_nodes, list_of_CC))
    return bc_dict, cc_dict


def calc_cluster_centrality(net: pd.DataFrame, meta: pd.DataFrame):
    list_of_all_nodes = list(set(net["from"].unique()).union(set(net["from"].unique())))

    cluster_dict = {
        row["Nodes"]: row["LouvainLabelD"]
        for _, row in meta.iterrows()
        if row["Nodes"] in list_of_all_nodes
    }

    net["from_clu"] = net["from"].apply(lambda x: d_apply(x, cluster_dict))
    net["to_clu"] = net["to"].apply(lambda x: d_apply(x, cluster_dict))

    node_centrality_dict = {}
    node_between_dict = {}
    for clu_num in meta.LouvainLabelD.unique():
        print("########### Cluster", clu_num)
        temp_net = net[(net["to_clu"] == clu_num) & (net["from_clu"] == clu_num)]
        temp_net["1-corr"] = 1 - temp_net["corr"]
        temp_G = nx.from_pandas_edgelist(temp_net, "from", "to", ["1-corr"])
        temp_closeness_values = nx.closeness_centrality(temp_G, distance="Weight")
        temp_betweeness_values = nx.betweenness_centrality(temp_G, weight="Weight")
        node_centrality_dict.update(temp_closeness_values)
        node_between_dict.update(temp_betweeness_values)

    print(len(node_centrality_dict))
    return node_centrality_dict, node_between_dict


def d_apply(x, d):
    try:
        ret = d[x]
    except Exception as err:
        print(f"err {err}")
        ret = np.nan
    return ret


def enriched_meta_table(
    df_abund: pd.DataFrame,
    df_meta: pd.DataFrame,
    df_env: pd.DataFrame,
    network: pd.DataFrame,
    ccm_network: pd.DataFrame,
    save_meta_path: str,
    calc_centralities: bool = True,
) -> pd.DataFrame:
    """

    :param df_abund:
    :param df_meta:
    :return:
    """
    df_abundance = df_abund.copy()
    # print(df_abundance.head())
    try:
        df_abundance.set_index("Unnamed: 0", inplace=True)
    except Exception as err:
        print("No Unnamed Col")
    print("Warning: Currently only for for years implemented.")
    abu_dict_ = df_abund.sum().fillna(0)
    abu_dict = abu_dict_.to_dict()
    print(abu_dict_)
    print(abu_dict)
    df_meta["Abundance4y"] = df_meta["Nodes"].apply(lambda x: abu_dict[x])
    df_meta["ColorLabel"] = df_meta["LouvainLabelD"].apply(
        lambda x: color_label_dict[x]
    )
    df_meta.to_csv(save_meta_path, sep=";")
    print("Before idxmac error")
    max_abundance_samples = df_abundance.idxmax()
    print("After idxmac error")
    print(max_abundance_samples)
    print("create max abundance table")
    max_abundance_table = pd.DataFrame(
        {
            "Nodes": max_abundance_samples.index,
            "Sample_with_Max_Abundance": max_abundance_samples.values,
        }
    )

    print("merge  max abundance table with env table ")
    meta_data_enhanced = pd.merge(
        max_abundance_table,
        df_env,
        left_on=max_abundance_samples.values,
        right_on=df_env.index,
    )

    # Drop redundant 'Sample' columns
    print("Drop redundant 'Sample' columns ")
    meta_data_enhanced.drop(columns=["Sample_with_Max_Abundance"], inplace=True)
    try:
        meta_data_enhanced["max_abundance_month"] = pd.to_datetime(
            meta_data_enhanced["key_0"]
        ).dt.month
    except Exception as exp_:
        meta_data_enhanced["max_abundance_month"] = meta_data_enhanced["key_0"]
        print(exp_)
    print("max_abundance_month")
    print(
        "The maximal abundance month is defined as the month of sample with highest abundance (of all samples)"
    )
    print(meta_data_enhanced["max_abundance_month"])
    meta_data_enhanced.drop(columns=["key_0"], inplace=True)
    print("apply colors for monthes ")
    try:
        meta_data_enhanced["max_abundance_month_color"] = meta_data_enhanced[
            "max_abundance_month"
        ].apply(lambda x: color_month_dict[str(x)])
    except Exception as exp__:
        print(exp__)
        meta_data_enhanced["max_abundance_month_color"] = "#FFFFFF"
        # Display or use the 'merged_data' as needed

    print(meta_data_enhanced.head())

    df_taxa_louvain = pd.read_csv(save_meta_path, index_col=0, sep=";")
    meta_data_combined = df_taxa_louvain.merge(meta_data_enhanced, how="outer")
    if calc_centralities:
        ### CCM Centrality

        print("Start calculating centrality")
        bc_dict, cc_dict = calc_centrality(ccm_network)
        print("Calc CCM_Betweenness_Centrality ")
        print(meta_data_combined.columns)
        meta_data_combined["CCM_Betweenness_Centrality"] = meta_data_combined[
            "Nodes"
        ].apply(lambda x: d_apply(x, bc_dict))
        print("Calc CCM_Closeness Centrality ")
        meta_data_combined["CCM_Closeness Centrality"] = meta_data_combined["Nodes"].apply(
            lambda x: d_apply(x, cc_dict)
        )

        ccm_node_centrality_dict, ccm_node_between_dict = calc_cluster_centrality(
            ccm_network, df_meta
        )
        meta_data_combined["ccm_cluster_closseness_centrality"] = meta_data_combined[
            "Nodes"
        ].map(ccm_node_centrality_dict)
        meta_data_combined["ccm_cluster_betweeness_centrality"] = meta_data_combined[
            "Nodes"
        ].map(ccm_node_between_dict)

        ### CON Centrality
        print("Calc CON Centrality")
        con_bc_dict, con_cc_dict = calc_centrality(network)

        print(meta_data_combined.columns)
        meta_data_combined["CON_Betweenness_Centrality"] = meta_data_combined[
            "Nodes"
        ].apply(lambda x: d_apply(x, con_bc_dict))
        meta_data_combined["CON_Closeness Centrality"] = meta_data_combined["Nodes"].apply(
            lambda x: d_apply(x, con_cc_dict)
        )

        con_node_centrality_dict, con_node_between_dict = calc_cluster_centrality(
            network, df_meta
        )
        meta_data_combined["con_cluster_closseness_centrality"] = meta_data_combined[
            "Nodes"
        ].map(con_node_centrality_dict)
        meta_data_combined["con_cluster_betweeness_centrality"] = meta_data_combined[
            "Nodes"
        ].map(con_node_between_dict)

    print(meta_data_combined.columns)
    print("Apply Louvain Color")
    meta_data_combined["louvain_label_color"] = meta_data_combined[
        "LouvainLabelD"
    ].apply(lambda x: d_apply(x, color_label_dict))
    try:
        shape_dict = {"Eukaryota": "Diamond", "Bacteria": "Ellipse", "Phages": "V", "Archea": "Octagon",
                      "Zooplankton": "Triangle", }
        meta_data_combined["Shape"] = meta_data_combined["Kingdom"].map(shape_dict)
    except KeyError as ke:
        print(ke)
        print("No Kingdom found in DF")
    meta_data_combined.to_csv(save_meta_path, index=False)
    print(meta_data_combined.head())

    return df_meta
