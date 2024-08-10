""" Script for calculating the louvain clusters on the CON"""

import pandas as pd

from config_file import (ABUNDANCES_FILE, CON_ALPHA, CON_LOUVAIN_META_PATH,
                         CON_LOUVAIN_NETWORK_PATH, CON_META_PATH, CON_METHOD,
                         CON_NETWORK_PATH, CON_SYM, CON_TR, FFT_COEFFS,
                         HELLENIGER_NORM, LOUVAIN_RES, METADATA_FILE,
                         TAXA_FILE)
from lutra.louvain import (create_meta_file, filter_threshold,
                           find_fewest_cluster_number)


def test_compute_louvain():
    print("Computing louvain clusters...")
    df = pd.read_csv(
        CON_NETWORK_PATH,
        sep=";",
    )
    df_meta = pd.read_csv(
        CON_META_PATH,
        sep=";",
    )
    resolutions = [LOUVAIN_RES]  # only atm
    save_table = CON_LOUVAIN_NETWORK_PATH
    save_meta = CON_LOUVAIN_META_PATH
    df = filter_threshold(df, CON_TR, save_table)
    ret_dict, res = find_fewest_cluster_number(df, resolutions)
    create_meta_file(ret_dict, df_meta, LOUVAIN_RES, save_meta)
    print("Finished louvain clusters...")


if __name__ == "__main__":
    test_compute_louvain()
