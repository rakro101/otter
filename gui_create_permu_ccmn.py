""" Script for calculating the p-value substituation for the CCM Network """

import pandas as pd
from lutra.permutation import get_ccm_pvalues
from lutra.enrich import enriched_meta_table


def add_sub_pval_to_ccmn(
    df_spec,
    CCMN_CON_MAP_PATH,
    CON_LOUVAIN_META_PATH,
    CON_LOUVAIN_NETWORK_PATH,
    NUM_PERMUTATIONS,
    NUM_SAMPLES,
    NUM_CORES,
    df_env,
    PRUNED_PVAL_CCMN_PATH,
    PVAL_CCMN_PATH,
    ENRICHED_META_PATH,
    RANDOM_PVAL_CCMN_PATH,
):
    print("Calculating p-value substituation for CCMN")
    # df_spec = pd.read_csv(ABUNDANCES_FILE, sep=";", index_col=0)
    # df_spec = df_spec.T.head(50).T
    df_ccm_con = pd.read_csv(CCMN_CON_MAP_PATH, sep=";")

    df_meta = pd.read_csv(
        CON_LOUVAIN_META_PATH,
        sep=";",
    )
    df_con = pd.read_csv(
        CON_LOUVAIN_NETWORK_PATH,
        sep=";",
    )

    df_pvalues, pruned_ccm, random_df = get_ccm_pvalues(
        df_spec,
        df_ccm_con,
        num_permutations=NUM_PERMUTATIONS,
        num_samples=NUM_SAMPLES,
        num_cores=NUM_CORES,
        mod="random_df",
    )
    # df_env = pd.read_csv(METADATA_FILE, sep=";", index_col=0)
    # save to csv
    random_df.to_csv(RANDOM_PVAL_CCMN_PATH, sep=";")
    pruned_ccm.to_csv(PRUNED_PVAL_CCMN_PATH, sep=";")
    df_pvalues.to_csv(PVAL_CCMN_PATH, sep=";")

    save_meta_path = ENRICHED_META_PATH
    enrichment = True
    if enrichment:
        df_meta_out = enriched_meta_table(
            df_spec, df_meta, df_env, df_con, df_ccm_con, save_meta_path
        )
        print(df_meta_out.head())
    print("End of Calculating p-value substituation for CCMN")
