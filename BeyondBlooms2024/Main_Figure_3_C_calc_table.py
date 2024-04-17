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
    df_ccm = pd.read_csv(PRUNED_PVAL_CCMN_PATH, sep=";")
    meta = pd.read_csv(ENRICH, sep=",")
    meta_file_dict = meta[["Nodes", "cluster_names"]]
    meta_file_dict.set_index("Nodes", inplace=True)
    mfd = meta_file_dict.to_dict()["cluster_names"]
    df_ccm["from_clu"] = df_ccm["from"].map(mfd)
    df_ccm["to_clu"] = df_ccm["to"].map(mfd)
    matrix_3b = (
        df_ccm[["from_clu", "to_clu", "corr"]]
        .groupby(["to_clu", "from_clu"])
        .agg(np.mean)
        .reset_index()
    )

    # Create a matrix using pivot
    matrix_pivot = matrix_3b.pivot(index="to_clu", columns="from_clu", values="corr")
    matrix_pivot.to_csv("tables/Main_Figure_S3_D_table.csv", sep=";")
    print(matrix_pivot.head())
