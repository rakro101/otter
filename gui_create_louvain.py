import pandas as pd
from lutra.louvain import filter_threshold, find_fewest_cluster_number, create_meta_file


def compute_louvain(
    CON_TR,
    CON_NETWORK_PATH,
    CON_META_PATH,
    LOUVAIN_RES,
    CON_LOUVAIN_NETWORK_PATH,
    CON_LOUVAIN_META_PATH,
):
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
