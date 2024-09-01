import random
import warnings
from functools import partial
from itertools import combinations
from multiprocessing import Pool

import numpy as np
import pandas as pd
from tqdm import tqdm

from lutra.ccmn import ConvergentCrossMapping as ccmn

warnings.filterwarnings("ignore")
from sklearn.utils import shuffle
from tqdm import tqdm

from lutra.transform import Transform

np.random.seed(42)


def calculate_causality(pair, data):
    i, j = pair
    L = len(data)  # length of time period to consider
    tau = 1
    d = 2
    # print(data.shape)
    ccm1 = ccmn(data.T[i], data.T[j], tau, d, L)
    ccm2 = ccmn(data.T[j], data.T[i], tau, d, L)
    corr_, p, pear = ccm1.nmi_causality()
    corr_2, p2, pear2 = ccm2.nmi_causality()

    return [(i, j, corr_, p, pear), (j, i, corr_2, p2, pear2)]


def calculate_permutation_matrix(
    df, num_permutations=1000, num_samples=1000, num_cores=20
):
    # Extracting values as a matrix
    print("Calculating permutation matrix !!!")
    matrix = df.values

    # Permuting entries 1000 times
    permuted_matrices = []  # To store permuted matrices

    for _ in range(num_permutations):
        permuted_matrix = np.random.permutation(matrix)
        permuted_matrices.append(permuted_matrix)

    # Access a particular permuted matrix (for example, the first one)

    all_results = []
    for ind, data in tqdm(enumerate(permuted_matrices), total=len(permuted_matrices)):
        column_pairs = list(combinations([h for h in range(0, df.shape[1])], 2))
        column_pairs = random.sample(column_pairs, num_samples)
        num_pairs = len(column_pairs)
        chunksize = num_pairs // (num_cores * 10)
        # Create a multiprocessing pool with 72 processes (one for each CPU core)
        with Pool(processes=num_cores) as pool:
            calculate_causality_partial = partial(calculate_causality, data=data)
            # Map pairs to processes and get the results
            results = list(
                tqdm(
                    pool.imap(
                        calculate_causality_partial, column_pairs, chunksize=chunksize
                    ),
                    total=len(column_pairs),
                    desc="Calculating correlations",
                )
            )
            correlations = [result for sublist in results for result in sublist]

        result_df = pd.DataFrame(
            correlations, columns=["from", "to", "corr", "p-value", "pearson"]
        )
        # result_df.to_csv(f"tutorial/permutations/result_{ind}.csv", sep=";")
        all_results.append(result_df)

    df_all = pd.concat(all_results, axis=0)
    # df_all.to_csv(f"tutorial/permutations/allresult_.csv", sep=";")
    return df_all


def get_pvalue(nmi, temp_shuffled):
    p_val = len(temp_shuffled[temp_shuffled > nmi]) / len(temp_shuffled)
    return p_val


def get_ccm_pvalues(
    df_spec,
    df_ccm_con,
    num_permutations=1000,
    num_samples=1000,
    num_cores=20,
    mod="no_random",
):
    df = Transform(df_spec).apply_hellinger()
    df_all = calculate_permutation_matrix(df, num_permutations, num_samples, num_cores)
    shuffled_df = df_all.copy()
    shuffle(shuffled_df["corr"], random_state=42)
    df_all = shuffled_df
    # number of con connections
    con_connections = df_ccm_con.shape[0]
    random_corr = df_all.head(con_connections)["corr"]
    random_df = df_all.head(con_connections)
    df_pvalues = df_ccm_con[["from", "to", "corr", "from_clu", "to_clu"]]
    print("Add permutation p- values")
    null_hypo = df_all.head(con_connections)
    shuffle(null_hypo["corr"], random_state=42)
    temp_shuffled = null_hypo["corr"].to_numpy()  # .head(5000)
    # df_pvalues["p-value"] = df_pvalues["corr"].apply(
    #    lambda x: get_pvalue(x, temp_shuffled)
    # )
    # df_pvalues["p-value"] = (df_pvalues["corr"].values[:,None] < temp_shuffled).mean(axis=1)
    ########
    import dask.dataframe as dd

    num_partitions = 100
    # Assuming df_pvalues is your pandas DataFrame
    ddf_pvalues = dd.from_pandas(
        df_pvalues, npartitions=num_partitions
    )  # num_partitions is the number of partitions you want

    # Define your computation function
    def compute_pvalue(chunk):
        return (chunk["corr"].values[:, None] < temp_shuffled).mean(axis=1)

    # Apply the computation function to each partition
    ddf_pvalues["p-value"] = ddf_pvalues.map_partitions(compute_pvalue)

    # Convert back to Pandas DataFrame if needed
    df_pvalues_with_pvalue = ddf_pvalues.compute()
    df_pvalues = df_pvalues_with_pvalue
    #####
    print("Added permutation p- values. Now filtering")
    pruned_ccm = df_pvalues[df_pvalues["p-value"] < 0.05]
    if mod == "random_df":
        return df_pvalues, pruned_ccm, random_df
    else:
        return df_pvalues, pruned_ccm
