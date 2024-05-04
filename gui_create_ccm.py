""" Script for generating the co-occurence network """

from lutra.networkz import Networkz
from lutra.transform import Transform
import pandas as pd


def create_ccmn_network(
    HELLENIGER_NORM,
    CCMN_METHOD,
    df_env,
    df_spec,
    df_taxa_info,
    CCMN_SYM,
    CCMN_NETWORK_PATH,
    CCMN_META_PATH,
):
    """Create the Convergent Cross Mapping Network"""
    # df_spec = pd.read_csv(ABUNDANCES_FILE, sep=";", index_col=0)
    if HELLENIGER_NORM == True:
        df_spec = Transform(df_spec).apply_hellinger()
    # for testing
    # df_spec = df_spec.T.head(50).T
    print(df_spec.shape)

    # df_env = pd.read_csv(METADATA_FILE, sep=";", index_col=0, decimal=",")
    print(df_env.columns)
    # df_taxa_info = pd.read_csv(TAXA_FILE, sep=";", index_col=0, decimal=",")
    calculator = Networkz(df_spec, None, df_taxa_info, method=CCMN_METHOD)
    result_df = calculator.calculate_relation_networkz()
    print(result_df.head())
    print(result_df.describe())
    calculator.reset_filtering()
    print(result_df.shape)
    meta, cluster_labels = calculator.create_meta_data(with_clusters=False)
    calculator.save_to_csv(
        CCMN_NETWORK_PATH,
        sym=CCMN_SYM,
    )
    calculator.save_to_csv(
        CCMN_META_PATH,
        mod="meta",
    )
