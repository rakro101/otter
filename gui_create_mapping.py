import pandas as pd
from lutra.mapping import map_ccm_to_con


def ccmn_con_mapping(
    CCMN_CON_MAP_PATH,
    CON_LOUVAIN_META_PATH,
    CON_LOUVAIN_NETWORK_PATH,
    CCMN_NETWORK_PATH,
):
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
