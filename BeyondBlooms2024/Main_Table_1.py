import pandas as pd
import numpy as np

num_coefficients = 14
from BeyondBlooms2024.config import name_dict, color_dict
from BeyondBlooms2024.config_file import (
    ABUNDANCES_FILE,
    CCMN_CON_MAP_PATH,
    CON_LOUVAIN_META_PATH,
    CON_LOUVAIN_NETWORK_PATH,
    ENRICH,
    NUM_PERMUTATIONS,
    NUM_SAMPLES,
    NUM_CORES,
    METADATA_FILE,
    PRUNED_PVAL_CCMN_PATH,
    PVAL_CCMN_PATH,
    ENRICHED_META_PATH,
    RANDOM_PVAL_CCMN_PATH,
)

if __name__ == "__main__":
    df_ccm = pd.read_csv(CON_LOUVAIN_NETWORK_PATH, sep=";")
    meta = pd.read_csv(ENRICH, sep=",")
    meta_file_dict = meta[["Nodes", "cluster_names"]]
    meta_file_dict.set_index("Nodes", inplace=True)
    mfd = meta_file_dict.to_dict()["cluster_names"]
    df_ccm["from_clu"] = df_ccm["from"].map(mfd)
    df_ccm["to_clu"] = df_ccm["to"].map(mfd)
    df_ccm["edge_count"] = 1
    matrix_3b = (
        df_ccm[
            [
                "from_clu",
                "to_clu",
                "edge_count",
            ]
        ]
        .groupby(["to_clu", "from_clu"])
        .agg(np.sum)
        .reset_index()
    )
    matrix_3b = matrix_3b[matrix_3b["from_clu"] == matrix_3b["to_clu"]]
    matrix_3b["cluster_names"] = matrix_3b["from_clu"]

    print(matrix_3b)
    meta["asv_count"] = 1

    print(meta.columns)
    seasons = (
        meta.groupby("cluster_names")
        .agg({"asv_count": "sum", "MaxMonth": pd.Series.mode})
        .reset_index()
    )
    df_table_1 = pd.merge(
        left=seasons, right=matrix_3b, on="cluster_names", how="inner"
    )
    print(df_table_1)
    df_table_1.to_csv("tables/Main_Table_1_CON_Overview.csv")
