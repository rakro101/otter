""" Script to map the CON and CCM together"""

import pandas as pd

from BeyondBlooms2024.config_file import (CCMN_CON_MAP_PATH, CCMN_NETWORK_PATH,
                                          CON_LOUVAIN_META_PATH,
                                          CON_LOUVAIN_NETWORK_PATH)
from lutra.mapping import map_ccm_to_con


def ccmn_con_mapping():
    print("Start Mapping CCMN CON")
    df_meta = pd.read_csv(
        CON_LOUVAIN_META_PATH,
        sep=";",
    )
    df_con = pd.read_csv(
        CON_LOUVAIN_NETWORK_PATH,
        sep=";",
    )

    df_ccm = pd.read_csv(
        CCMN_NETWORK_PATH,
        sep=";",
    )

    save_path = CCMN_CON_MAP_PATH
    map_ccm_to_con(df_meta, df_con, df_ccm, save_path)
    print("End Mapping CCMN CON")


if __name__ == "__main__":
    ccmn_con_mapping()
