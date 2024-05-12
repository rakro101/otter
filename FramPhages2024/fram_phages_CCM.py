import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
np.random.seed(42)
mod= "purple_snail"
sub_dir = "data/project/fram_phages"
num_fft = 16

if __name__ == "__main__":
    from lutra.miat import ConvergentCrossMappingNetwork
    from lutra.transform import Transform
    import pandas as pd
    import numpy as np

    hellinger = False
    chosen_method = "NMI"
    if chosen_method in ["Pearson_FFT", "Pearson"]:
        sym = True
    else:
        sym = False
    # Example usage
    df_spec = pd.read_csv(f'{sub_dir}/data/abundance.csv', sep=";", index_col=0)
    if hellinger == True:
        df_spec = Transform(df_spec).apply_hellinger()
    print(df_spec.shape)

    df_env = pd.read_csv(f'{sub_dir}/data/environment_info.csv', sep=";", index_col=0, decimal=",")
    print(df_env.columns)
    df_taxa_info = pd.read_csv(f'{sub_dir}/data/taxa_info.csv', sep=";", index_col=0, decimal=",")
    calculator = ConvergentCrossMappingNetwork(df_spec, None, df_taxa_info, method=chosen_method)
    result_df = calculator.calculate_cross_mapping_correlations()
    print(result_df.head())
    print(result_df.describe())
    calculator.reset_filtering()
    tr = 0.00
    print(result_df.shape)
    alpha = "no"  # 0.05
    meta, cluster_labels = calculator.create_meta_data(with_clusters=False)
    calculator.save_to_csv(
        f"{sub_dir}/tables/Hellinger_{hellinger}_{num_fft}_{chosen_method}__complete_network_table_{tr}_{alpha}_{mod}.csv",
        sym=sym)
    calculator.save_to_csv(
        f"{sub_dir}/tables/Hellinger_{hellinger}_{num_fft}_{chosen_method}__complete_network_table_meta_{tr}_{alpha}_{mod}.csv",
        mod="meta")