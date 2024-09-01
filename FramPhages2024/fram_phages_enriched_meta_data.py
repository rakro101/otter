import pandas as pd
from lutra.enrich import enriched_meta_table
#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
np.random.seed(42)

if __name__ == "__main__":
    mod= "purple_snail"
    sub_dir = "data/project/fram_phages"
    num_fft = 16
    hellinger = False
    from lutra.mapping import map_ccm_to_con
    df_env = pd.read_csv(f'{sub_dir}/data/meta_info.csv', sep=";", index_col=0, decimal=",")

    chosen_method = "Pearson_FFT"
    res=1
    tr=0.70
    alpha = 0.05
    mapping = True
    enrichment = True

    df_meta = pd.read_csv(
        f"{sub_dir}/tables/Louvain_1_Pearson_FFT_Hellinger_{hellinger}__{num_fft}_complete_network_table_meta_0.7_0.05_{mod}_new.csv",
        sep=";")
    df_con = pd.read_csv(
        f"{sub_dir}/tables/Louvain_1_Pearson_FFT_Hellinger_{hellinger}__{num_fft}_complete_network_table_0.7_0.05_{mod}_new.csv",
        sep=";")

    pruned_ccm =pd.read_csv(f"{sub_dir}/tables/Pruned_CCM_CON_MAP_Network_{mod}.csv", sep=";")
    df_ccm = pruned_ccm
    df_abund = pd.read_csv(f'{sub_dir}/data/abundance.csv', sep=";")


    save_meta_path = f"{sub_dir}/tables/Enriched{res}_{chosen_method}_Hellinger_{hellinger}__{num_fft}_complete_network_table_meta_{tr}_{alpha}_{mod}_update_new.csv"
    if enrichment:
        print("Start")
        df_meta_out = enriched_meta_table(df_abund, df_meta, df_env, df_con, df_ccm, save_meta_path)
        print(df_meta_out.head())