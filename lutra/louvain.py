from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
import pandas as pd

from lutra.config import color_label_dict

np.random.seed(42)


def create_edge_list(df: pd.DataFrame) -> nx.Graph:
    """
    Create a graph from a pandas DataFrame containing edge list data.

    Parameters:
        df (pd.DataFrame): DataFrame containing the edge list data with 'from', 'to', 'corr', and 'p-value' columns.

    Returns:
        G (nx.Graph): The graph created from the edge list.
    """
    G = nx.from_pandas_edgelist(df, "from", "to", ["corr", "p-value"])
    return G


def find_fewest_cluster_number(
    df: pd.DataFrame, resolutions: List[float]
) -> Tuple[Dict[str, int], float]:
    """
    Find the fewest cluster number using Louvain community detection with different resolutions.

    Parameters:
        df (pd.DataFrame): DataFrame containing the edge list data with 'from', 'to', 'corr', and 'p-value' columns.
        resolutions (List[float]): List of resolution values to be used in Louvain clustering.

    Returns:
        ret_dict (Dict[str, int]): A dictionary mapping nodes to their cluster numbers.
        res (float): The resolution value that resulted in the fewest clusters.
    """
    G = create_edge_list(df)
    best_partition = None
    best_modularity_value = -1  # Initialize with a large value

    for resolution in resolutions:
        partition = nx.community.louvain_communities(G, resolution=resolution, seed=42)
        modularity_value = nx.algorithms.community.modularity(G, partition)

        if modularity_value > best_modularity_value:
            best_partition = partition
            best_modularity_value = modularity_value
            res = resolution

    partition = best_partition
    modularity_value = nx.algorithms.community.modularity(G, partition)
    print("Modularity:", modularity_value)
    ret_dict = {
        node: cluster for cluster, nodes in enumerate(partition) for node in nodes
    }
    print("number of classes:", len(best_partition))
    return ret_dict, res


def apply_dict(x: str, di: Dict[str, int]) -> int:
    """
    Apply a dictionary to assign cluster numbers to nodes.

    Parameters:
        x (str): The node name.
        di (Dict[str, int]): A dictionary mapping nodes to their cluster numbers.

    Returns:
        int: The cluster number assigned to the node.
    """
    try:
        ret = di[x]
    except KeyError:
        ret = -1
    return ret


def create_meta_file(
    ret_dict: Dict[str, int], df_meta: pd.DataFrame, res: float, save_meta: str
) -> pd.DataFrame:
    """
    Create a meta file with Louvain cluster information.

    Parameters:
        ret_dict (Dict[str, int]): A dictionary mapping nodes to their cluster numbers.
        df_meta (pd.DataFrame): DataFrame containing the metadata with 'Nodes' column representing node names.
        res (float): The resolution value that resulted in the fewest clusters.

    Returns:
        pd.DataFrame: DataFrame with added Louvain cluster information.
    """
    df_meta["LouvainLabelD"] = df_meta["Nodes"].apply(lambda x: apply_dict(x, ret_dict))
    df_meta["LouvainLabelD_res"] = res
    df_meta["LouvainLabelD"].apply(lambda x: color_label_dict[x])
    df_meta = df_meta[df_meta["LouvainLabelD"] != -1]
    df_meta.to_csv(save_meta, sep=";")
    return df_meta


def filter_threshold(
    df: pd.DataFrame, threshold: float, save_table: str
) -> pd.DataFrame:
    """
    Filter the DataFrame based on a threshold value and save the result to a CSV file.

    Parameters:
        df (pd.DataFrame): DataFrame containing the edge list data with 'corr' column.
        threshold (float): The threshold value for filtering.
        save_table (str): The file path to save the filtered DataFrame.

    Returns:
        pd.DataFrame: The filtered DataFrame.
    """
    df = df[df["corr"] >= threshold]
    df.to_csv(save_table, sep=";")
    return df
