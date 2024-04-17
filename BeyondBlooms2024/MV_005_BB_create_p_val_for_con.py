""" Modelvalidation """

import pandas as pd
from lutra.MV_permutation import get_ccm_pvalues
from lutra.enrich import enriched_meta_table
from config_file import (
    ABUNDANCES_FILE,
    NUM_CORES,
    MV_PRUNED_PVAL_CCMN_PATH,
    MV_PVAL_CCMN_PATH,
    MV_CON_NETWORK_PATH,
)


def add_sub_pval_to_ccmn():
    print("Calculating p-value substituation for CON")
    df_spec = pd.read_csv(ABUNDANCES_FILE, sep=";", index_col=0)

    df_ccm_con = pd.read_csv(MV_CON_NETWORK_PATH, sep=";")

    df_pvalues, pruned_ccm = get_ccm_pvalues(
        df_spec,
        df_ccm_con,
        num_permutations=1000,
        num_samples=1000,
        num_cores=NUM_CORES,
    )

    pruned_ccm.to_csv(MV_PRUNED_PVAL_CCMN_PATH, sep=";")
    df_pvalues.to_csv(MV_PVAL_CCMN_PATH, sep=";")


if __name__ == "__main__":
    add_sub_pval_to_ccmn()
