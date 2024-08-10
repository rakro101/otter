from typing import Dict

import numpy as np
import pandas as pd

np.random.seed(42)


def apply_dict(x: str, dict_: Dict) -> int:
    """

    :param x:
    :param dict_:
    :return:
    """
    try:
        # print(x)
        ret = dict_[x]
    except:
        ret = -1
    return ret


def map_ccm_to_con(
    df_meta: pd.DataFrame,
    df_table: pd.DataFrame,
    df_ccm: pd.DataFrame,
    save_path: str,
    modus="CON",
) -> pd.DataFrame:
    """

    :param df_meta:
    :param df_table:
    :param df_ccm:
    :param save_path:
    :return:
    """
    print("Modus : " + str(modus))
    df_meta_dict = df_meta[["Nodes", "LouvainLabelD"]]
    df_meta_dict.set_index("Nodes", inplace=True)
    meta_dict = df_meta_dict.to_dict()["LouvainLabelD"]

    df_table["from_clu"] = df_table["from"].apply(lambda x: apply_dict(x, meta_dict))
    df_table["to_clu"] = df_table["to"].apply(lambda x: apply_dict(x, meta_dict))

    df_table = df_table[~df_table["from_clu"].isin([-1])]
    df_table = df_table[~df_table["to_clu"].isin([-1])]

    # Create a new DataFrame with columns switched
    switched_df = df_table.rename(columns={"from": "to", "to": "from"})

    # Concatenate the original and switched DataFrames
    df_table_2 = pd.concat([df_table, switched_df], ignore_index=True)

    df_ccm.dropna(inplace=True)
    print("Shape CCMN", df_ccm.shape)
    df_ccm["from_clu"] = df_ccm["from"].apply(lambda x: apply_dict(x, meta_dict))
    df_ccm["to_clu"] = df_ccm["to"].apply(lambda x: apply_dict(x, meta_dict))
    if modus == "CON":
        df_ccm = df_ccm[~(df_ccm["from_clu"].isin([-1]) & df_ccm["to_clu"].isin([-1]))]
        print("CON CCMN Shape:", df_ccm.shape)
    df_ccm["ccm"] = 1
    df_table_2["con"] = 1
    if modus == "CON":
        df_result = pd.merge(
            df_ccm, df_table_2[["from", "to", "con"]], on=["from", "to"], how="right"
        )
        print("CON CCMN Shape:", df_result.shape)
    else:
        df_result = pd.merge(
            df_ccm, df_table_2[["from", "to", "con"]], on=["from", "to"], how="outer"
        )
        df_result = df_result[df_result["con"] != 1]
        print("ANTI CON CCMN Shape:", df_result.shape)

    df_result.to_csv(save_path, sep=";")
    print(f"dataframe saved to {save_path}")
    return df_result
