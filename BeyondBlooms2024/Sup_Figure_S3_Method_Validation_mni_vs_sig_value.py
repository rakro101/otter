import matplotlib.pyplot as plt
import pandas as pd

if __name__ == "__main__":
    from BeyondBlooms2024.config_file import PVAL_CCMN_PATH

    df_con_ccm = pd.read_csv(PVAL_CCMN_PATH, sep=";")
    plt.scatter(df_con_ccm["corr"], df_con_ccm["p-value"])
    plt.axhline(y=0.05, color="r", linestyle="--", label="y = 0.05")
    plt.xlabel("normalized mutual information")
    plt.ylabel("permutation pvalue")
    plt.savefig("figures/Sup_Figure_S3_nmi_vs_significants_value_ALL.png")
    plt.show()
