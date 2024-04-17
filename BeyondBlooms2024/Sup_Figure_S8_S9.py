import pandas as pd
from scipy.stats import ks_2samp
import numpy as np

np.random.seed(42)
from sklearn.utils import shuffle

if __name__ == "__main__":
    # Extract the columns you want to compare
    from BeyondBlooms2024.config_file import (
        PVAL_CCMN_PATH,
        ANTI_CCMN_CON_MAP_PATH,
        RANDOM_PVAL_CCMN_PATH,
    )

    anti_con_corr = pd.read_csv(ANTI_CCMN_CON_MAP_PATH, sep=";")["corr"]
    ccm_con_corr = pd.read_csv(PVAL_CCMN_PATH, sep=";")["corr"]
    random_corr = pd.read_csv(RANDOM_PVAL_CCMN_PATH, sep=";")["corr"]

    # sample fixed size as min of all sizes.
    min_num = min(len(anti_con_corr), len(ccm_con_corr), len(random_corr))
    print(min_num)
    shuffle(anti_con_corr, random_state=42)
    shuffle(ccm_con_corr, random_state=42)
    shuffle(random_corr, random_state=42)

    anti_con_corr = anti_con_corr[:min_num]
    ccm_con_corr = ccm_con_corr[:min_num]
    random_corr = random_corr[:min_num]

    col1 = random_corr  # [random_corr>0]
    col2 = ccm_con_corr  # [ccm_con_corr>0]
    col3 = anti_con_corr
    # Perform the Kolmogorov-Smirnov test
    ks_statistic, p_value = ks_2samp(col1, col2)
    print("Kolmogorov-Smirnov Statistic ks(random, con_ccm):", ks_statistic)
    print("P-value:", p_value)

    ks_statistic_, p_value_ = ks_2samp(col1, col3)
    print("Kolmogorov-Smirnov Statistic ks(random, anti_con):", ks_statistic_)
    print("P-value:", p_value_)

    ks_statistic__, p_value__ = ks_2samp(col2, col3)
    print("Kolmogorov-Smirnov Statistic ks(anti_con, ccm_con):", ks_statistic__)
    print("P-value:", p_value__)

    df_distr = pd.DataFrame()
    df_distr["cols"] = [
        "random vs. con_ccm",
        "random vs. non_con_ccm",
        "non_con_ccm vs. con_ccm",
    ]
    df_distr["test"] = "Kolmogorov-Smirnov"
    df_distr["alternative"] = "two-sided"
    df_distr["test_stat"] = [ks_statistic, ks_statistic_, ks_statistic__]
    df_distr["p-value"] = [p_value, p_value_, p_value__]
    df_distr["significants stars"] = ["***", "", "***"]

    df_distr.to_csv(
        "tables/Sup_Table_S2_Kolmogorov-Smirnov_ccm_variants_ALL.csv", sep=";"
    )
    df_distr.to_latex("tables/Sup_Table_S2_Kolmogorov-Smirnov_ccm_variants_ALL.txt")

    # Print the results
    print("mean random", col1.mean())
    print("mean con ccm", col2.mean())
    print("mean anti con ", col3.mean())

    print("median random", col1.median())
    print("median con ccm", col2.median())
    print("median anti con ", col3.median())

    print("std random", col1.std())
    print("std con ccm", col2.std())
    print("std anti con ", col3.std())

    print("sum random", col1.sum())
    print("sum con ccm", col2.sum())
    print("sum anti con", col3.sum())

    print("len random", col1.count())
    print("len con ccm", col2.count())
    print("len anti con", col3.count())

    mean_list = []
    median_list = []
    std_list = []
    sum_list = []
    count_list = []
    for col in [col1, col2, col3]:
        mean_list.append(col.mean())
        median_list.append(col.median())
        std_list.append(col.std())
        sum_list.append(col.sum())
        count_list.append(col.count())

    df_samples = pd.DataFrame()
    df_samples["random"] = col1
    df_samples["con_ccm"] = col2
    df_samples["non_con_ccm"] = col2
    df_samples.to_csv("tables/Sup_Table_S1_ADD_samples_ccm_variants_ALL.csv", sep=";")
    df_samples.to_latex("tables/Sup_Table_S1_ADD_samples_ccm_variants_ALL.txt")

    df_stat = pd.DataFrame()
    df_stat["cols"] = ["random", "con_ccm", "non_con_ccm"]
    df_stat["mean"] = mean_list
    df_stat["median"] = median_list
    df_stat["std"] = std_list
    df_stat["sum"] = sum_list
    df_stat["count"] = count_list

    df_stat.to_csv("tables/Sup_Table_S1_statistics_ccm_variants_ALL.csv", sep=";")
    df_stat.to_latex("tables/Sup_Table_S1_statistics_ccm_variants_ALL.txt")

    from scipy import stats

    # Assuming you have two samples: sample1 and sample2
    # For example:
    sample1 = col1
    sample2 = col2
    sample3 = col3

    import matplotlib.pyplot as plt

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(12, 4))

    # Plot the histogram for col1 on the first subplot (ax1)
    ax1.hist(col1, bins=20, color="darkgreen")
    ax1.set_xlabel("Normalized mutual information")
    ax1.set_ylabel("Frequency")
    ax1.set_title("Random")

    # Plot the histogram for col2 on the second subplot (ax2)
    ax2.hist(col2, bins=20, color="darkgreen")
    ax2.set_xlabel("Normalized mutual information")
    ax2.set_ylabel("Frequency")
    ax2.set_title("CON_CCM")

    ax3.hist(col3, bins=20, color="darkgreen")
    ax3.set_xlabel("Normalized mutual information")
    ax3.set_ylabel("Frequency")
    ax3.set_title("NON_CON_CCM")

    # Add a common title for both subplots
    # plt.suptitle('Histograms of Normalised mutual information')

    # Adjust layout for better spacing
    plt.tight_layout()
    plt.savefig(
        "figures/Sup_Figure_S9_random_non_con_ccm_con_ccm_Nessi_Histograms_ALL.png"
    )
    # Show the plots
    plt.show()

    import matplotlib.pyplot as plt
    import numpy as np
    import scipy.stats as stats

    # Generate some sample data (replace this with your actual data)
    data = col2

    # Create a QQ plot
    # stats.probplot(data, dist="norm", plot=plt)
    # plt.title('QQ Plot - Normal Distribution Assumption')
    # plt.xlabel('Theoretical Quantiles')
    # plt.ylabel('Sample Quantiles')
    # plt.show()

    # Sort the values in the column
    sorted_values = col1.sort_values()
    # Calculate the cumulative distribution function
    cdf = sorted_values.rank(method="max") / len(sorted_values)

    sorted_values2 = col2.sort_values()
    # Calculate the cumulative distribution function
    cdf2 = sorted_values2.rank(method="max") / len(sorted_values2)

    sorted_values3 = col3.sort_values()
    # Calculate the cumulative distribution function
    cdf3 = sorted_values3.rank(method="max") / len(sorted_values3)

    # Plot the CDF
    plt.plot(sorted_values, cdf, marker="x", color="grey", label="random")
    plt.plot(sorted_values2, cdf2, marker="o", color="#cc33ff", label="con_ccm")
    plt.plot(sorted_values3, cdf3, marker="o", color="#6699ff", label="non_con_ccm")
    # plt.title('Cumulative Distribution Function (CDF) Plot of ccm variants')
    plt.xlabel("Normalized mutual information")
    plt.ylabel("Cumulative Probability")
    plt.legend()
    plt.grid(True)
    plt.savefig("figures/Sup_Figure_S9_random_non_con_ccm_con_ccm_CFD_plot_ALL.png")
    plt.show()
