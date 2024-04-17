""" Mutual Information Attractor Network - miat """

from itertools import combinations, combinations_with_replacement
from lutra.ccmn import ConvergentCrossMapping as ccmn
from lutra.con import CoOccurrence as con
import pandas as pd
from tqdm import tqdm
from typing import Union, Any, List
import numpy as np
import networkx as nx
import warnings
import statsmodels.stats.multitest as smm
from multiprocessing import Pool
from functools import partial

np.random.seed(42)

warnings.filterwarnings("ignore")


class Networkz:
    def __init__(
        self,
        data_spec: pd.DataFrame,
        data_env: Union[pd.DataFrame, Any] = None,
        data_taxa_info: Union[pd.DataFrame, Any] = None,
        tau: int = 1,
        d: int = 2,
        identity: bool = False,
        num_cores=8,
        method: str = "AMI",
        num_coefficients=14,
    ):
        """
        Initializes the ConvergentCrossMappingNetwork instance.

        Parameters:
            data_spec: data frame with timeseries with abundance of species
            data_env: data frame with timeseries with abundance of env (optional)
            data_taxa_info: Taxa info (optional)
            num_cores: Number of cores for parallel
            method:  AMI: Adjusted Mutual Info Score, NMI: Normalized Mutual Info Score, Pearson Pearson: Correlation,
            tau (int): The time lag.
            d (int): The Shadow manifold embedding dimension.
            identity (bool): If True, calculate cross mapping correlation between the same ASV

        """
        self.tau = tau
        self.d = d
        self.identity = identity
        self.data_spec = data_spec
        self.data_env = data_env
        if type(data_taxa_info) != type(None):
            self.data_taxa_info = data_taxa_info
        self.data = data_spec
        if type(self.data_env) != type(None):
            self.join_spec_env()
        self.network_table = self.data
        self.network_table_copy = self.data
        self.network_table_filter_corr = self.data
        self.network_table_filter_p = self.data
        self.network_table_no_na = self.data
        self.network_table_meta = self.data
        self.num_cores = num_cores
        self.method = method
        self.num_coefficients = num_coefficients

    def join_spec_env(self):
        """
        Join Enviroment and Species Data

        :return: data (pd.DataFrame) DataFrame with Species and Enviroment data

        """
        self.data = self.data_spec.join(self.data_env)

    def calculate_relation_pairwise(self, pair, data):
        asv1, asv2 = pair
        L = len(data)
        if self.method == "CCM" in ["Pearson", "Pearson_FFT", "Pearson_FFT_MI"]:
            from_to = con(data[asv1], data[asv2], L, self.num_coefficients)
            to_from = con(data[asv2], data[asv1], L, self.num_coefficients)
        else:
            from_to = ccmn(
                data[asv1], data[asv2], self.tau, self.d, L, self.num_coefficients
            )
            to_from = ccmn(
                data[asv2], data[asv1], self.tau, self.d, L, self.num_coefficients
            )
        if self.method == "CCM":
            # Using CrossConvMapping and measure correlation with Pearson R
            corr_, p, pear = from_to.causality()
            corr_2, p2, pear2 = to_from.causality()
        elif self.method == "NMI":
            # Using CrossConvMapping and measure correlation with normalized MI
            corr_, p, pear = from_to.nmi_causality()
            corr_2, p2, pear2 = to_from.nmi_causality()
        elif self.method == "Pearson":
            #  Using Calculate pairwise Co Occurence
            corr_, p, pear = from_to.occurence()
            corr_2, p2, pear2 = -666, -666, -666  # non sym
        elif self.method == "Pearson_FFT":
            #  Using Calculate pairwise Co Occurence using FFT
            corr_, p, pear = from_to.occurence_fft()
            corr_2, p2, pear2 = -666, -666, -666  # non sym
        elif self.method == "Pearson_FFT_MI":
            #  Using Calculate pairwise Co Occurence using FFT
            corr_, p, pear = from_to.occurence_fft_mi()
            corr_2, p2, pear2 = to_from.occurence_fft_mi()
        else:
            print("Warning - no method selected.")
            return []
        return [(asv1, asv2, corr_, p, pear), (asv2, asv1, corr_2, p2, pear2)]

    def calculate_relation_networkz(self) -> pd.DataFrame:
        """
        Calculates the causality correlations between column pairs in the given DataFrame.

        Parameters:
            data (pd.DataFrame): The input DataFrame.

        Returns:
            pd.DataFrame: A new DataFrame containing the from, to, correlation, and p-value columns.

        """
        print(f"As corr the {self.method} score is selected.")
        print("The p-value is calculated for the pearson correlation - not the MI")
        self.data = self.data.fillna(0)
        identity = False
        if identity:
            column_pairs = list(
                combinations_with_replacement(list(self.data.columns), 2)
            )
        else:
            column_pairs = list(combinations(list(self.data.columns), 2))
        num_cores = self.num_cores
        num_pairs = len(column_pairs)
        print(num_pairs)
        chunksize = num_pairs // (num_cores * 10)
        # Create a multiprocessing pool with 72 processes (one for each CPU core)
        with Pool(processes=num_cores) as pool:
            calculate_causality_partial = partial(
                self.calculate_relation_pairwise, data=self.data
            )
            # Map pairs to processes and get the results
            results = list(
                tqdm(
                    pool.imap(
                        calculate_causality_partial, column_pairs, chunksize=chunksize
                    ),
                    total=len(column_pairs),
                    desc="Calculating correlations",
                )
            )
            correlations = [result for sublist in results for result in sublist]

        result_df = pd.DataFrame(
            correlations, columns=["from", "to", "corr", "p-value", "pearson"]
        )
        print("Before droping na:", result_df.shape)
        result_df = result_df.dropna()
        print("After droping na:", result_df.shape)
        # result_df = result_df[result_df['pearson']!=-666]#to make it symetric
        result_df["p-value_adj_fdr_bh"] = smm.multipletests(
            result_df["p-value"], method="fdr_bh"
        )[1]
        result_df = result_df.drop_duplicates()
        self.network_table = result_df
        self.network_table_copy = result_df
        print("Networktable:", self.network_table.shape)
        return self.network_table

    def reset_filtering(self):
        self.network_table = self.network_table_copy
        print(f"The full network table without filtering was restored")

    def filter_correlations(
        self, threshold: float, replace: bool = True
    ) -> pd.DataFrame:
        """
        Filters the correlations DataFrame based on a given threshold.

        Parameters:
            data (pd.DataFrame): The input DataFrame containing correlations.
            threshold (float): The correlation threshold value.
            replace (bool): if you want overwrite the networktable

        Returns:
            pd.DataFrame: A new DataFrame containing correlations above the threshold.

        """

        self.network_table_filter_corr = self.network_table[
            self.network_table["corr"] >= threshold
        ]
        if replace:
            self.network_table = self.network_table_filter_corr
        print("Print after filter corr:", self.network_table_filter_corr.shape)
        return self.network_table_filter_corr

    def remove_high_p_values(
        self, alpha: float = 0.99, replace: bool = True
    ) -> pd.DataFrame:
        """
        Removes rows from the DataFrame where the p-value is greater than a given threshold.

        Parameters:
            data (pd.DataFrame): The input DataFrame containing correlations.
            alpha (float): The significance level threshold.
            replace (bool): if you want overwrite the networktable

        Returns:
            pd.DataFrame: A new DataFrame with rows removed where the p-value is greater than the threshold.

        """
        self.network_table_filter_p = self.network_table[
            self.network_table["p-value_adj_fdr_bh"] <= alpha
        ]
        if replace:
            self.network_table = self.network_table_filter_p
        print("Print after filter p-val:", self.network_table_filter_p.shape)
        return self.network_table_filter_p

    def drop_na_columns(self, replace: bool = True) -> pd.DataFrame:
        """
        Drops columns with missing values from the DataFrame.
        Parameters:
            replace (bool): if you want overwrite the networktable
        Returns:
            (pd.Dataframe): containing the modified DataFrame
        """
        df_shape_before = self.network_table.shape[0]
        self.network_table_no_na = self.network_table.dropna(axis=0)
        df_shape_after = self.network_table.shape[0]
        print(f"{df_shape_before-df_shape_after} rows was removed.")
        if replace:
            self.network_table = self.network_table_no_na
        print("Print after drop_na_cols", self.network_table_no_na.shape)
        return self.network_table_no_na

    def save_to_csv(
        self, filename: str, mod: Union[str, Any] = None, sym: bool = False
    ):
        """
        Saves the DataFrame to a CSV file.

        Parameters:
            data (pd.DataFrame): The input DataFrame containing correlations.
            filename (str): The filename for the CSV file.
        """
        if mod == "meta":
            self.network_table_meta.to_csv(filename, sep=";", decimal=".")
            print(f"Meta DataFrame saved to {filename}.")
        else:
            if sym:
                self.sym_relation_ship()
            self.network_table.to_csv(filename, index=False, sep=";", decimal=".")
            print(f"Network Table DataFrame saved to {filename}.")

    def sym_relation_ship(self):
        # Create a new column 'to_from' which contains sorted pairs of 'from' and 'to' columns
        print("Before sym", self.network_table.shape)
        self.network_table = self.network_table[self.network_table["pearson"] != -666]
        print("After Sym ", self.network_table.shape)
        return self.network_table

    def create_meta_data(
        self, with_clusters: bool = True, thresholds: List[float] = [1.0]
    ) -> pd.DataFrame:
        """Create metadata for the network table.
        Parameters:
            with_clusters (bool): If True add a column called cluster to the meta data
        Returns:
            DataFrame: The metadata DataFrame with unique entries and joined taxa information.
        """
        unique_entries = np.unique(self.network_table[["from", "to"]].values)
        unique_entries = sorted(unique_entries)
        df_unique_entries = pd.DataFrame({"Nodes": unique_entries})
        df_unique_entries = df_unique_entries.set_index("Nodes", drop=True)
        self.network_table_meta = df_unique_entries.join(self.data_taxa_info).fillna(
            "Environment_Condition"
        )
        cluster_labels = True
        if with_clusters:
            cluster_labels = self.calculate_clusters(thresholds)
        return self.network_table_meta, cluster_labels

    def calculate_clusters(self, thresholds: List[float]):
        """
        Calculate clusters for different thresholds.

        Args:
            thresholds (List[float]): List of threshold values.

        Returns:
            Tuple[Dict[str, pd.Series], pd.DataFrame]:
                - A dictionary containing cluster labels for each threshold.
                - Updated network_table_meta DataFrame with cluster labels.
        """

        cluster_labels = {}
        # Iterate over the thresholds
        for threshold in tqdm(thresholds, desc="Calculating different thresholds"):
            # Initialize cluster label
            label = 0

            # Check if threshold key exists in network_table_meta
            key = "Threshold " + format(threshold, ".2f")
            if key not in self.network_table_meta:
                self.network_table_meta[key] = self.network_table_meta.index
            graph_dict = self.calculate_connected_graphs(self.network_table, threshold)
            self.network_table_meta[key] = self.network_table_meta[key].apply(
                lambda x: self.apply_dict(x, graph_dict)
            )

            # Store the cluster labels for the current threshold in the cluster_labels dictionary

            cluster_labels[format(threshold, ".2f")] = self.network_table_meta[
                key
            ].copy()

            # Print the number of clusters for the current threshold
            num_clusters = len(
                np.unique(self.network_table_meta[key].value_counts().index)
            )
            print(f"Number of clusters for {key}: {num_clusters}")

        return cluster_labels

    def apply_dict(self, node, apply_dict):
        try:
            ret = apply_dict[node]
        except Exception as err:
            ret = -1
        return ret

    def calculate_connected_graphs(self, df, threshold):
        # Create an empty graph
        G = nx.Graph()

        # Add edges to the graph based on the correlation threshold
        filtered_df = df[df["corr"] >= threshold]
        edges = filtered_df[["to", "from"]].values.tolist()
        G.add_edges_from(edges)

        # Calculate connected components
        connected_components = nx.connected_components(G)

        # Assign labels to nodes
        node_labels = {}
        for i, component in enumerate(connected_components):
            for node in component:
                node_labels[node] = i + 1

        return node_labels
