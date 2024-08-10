""" Script for generating the co-occurence network """

import pandas as pd

from config_file import (ABUNDANCES_FILE, CON_ALPHA, CON_META_PATH, CON_METHOD,
                         CON_NETWORK_PATH, CON_SYM, CON_TR, FFT_COEFFS,
                         HELLENIGER_NORM, METADATA_FILE, TAXA_FILE)
from lutra.networkz import Networkz
from lutra.transform import Transform


def test_create_con_network():
    print("Start CON construction")
    df_spec = pd.read_csv(ABUNDANCES_FILE, sep=";", index_col=0)
    if HELLENIGER_NORM == True:
        df_spec = Transform(df_spec).apply_hellinger()
    print(df_spec.shape)
    # for testing
    df_spec = df_spec.T.head(50).T

    df_env = pd.read_csv(METADATA_FILE, sep=";", index_col=0, decimal=";")
    print(df_env.columns)
    df_taxa_info = pd.read_csv(
        TAXA_FILE,
        sep=";",
        index_col=0,
    )
    calculator = Networkz(
        df_spec, None, df_taxa_info, method=CON_METHOD, num_coefficients=FFT_COEFFS
    )
    result_df = calculator.calculate_relation_networkz()
    print(result_df.head())
    print(result_df.describe())
    calculator.reset_filtering()
    print(result_df.shape)
    filtered_df = calculator.filter_correlations(CON_TR)
    print(filtered_df.shape)
    removed_df = calculator.remove_high_p_values(CON_ALPHA)
    print(removed_df)
    meta, cluster_labels = calculator.create_meta_data(with_clusters=False)
    print("filter:", filtered_df.shape)
    calculator.save_to_csv(
        CON_NETWORK_PATH,
        sym=CON_SYM,
    )
    calculator.save_to_csv(
        CON_META_PATH,
        mod="meta",
    )
    print("End CON construction")


if __name__ == "__main__":
    # Example usage
    test_create_con_network()
