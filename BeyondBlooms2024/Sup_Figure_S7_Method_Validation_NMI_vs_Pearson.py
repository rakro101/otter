import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

np.random.seed(42)

if __name__ == "__main__":
    df = pd.read_csv("data/veilleux_1.txt", sep=";")
    df["time"] = df["time (d)"] - 4.5
    df["Didinium"] = df[" prey(#ind/ml)"]
    df["Parameidcium"] = df[" predator(#ind/ml)"]
    print(df["time"].values)

    df = df.tail(62)
    print(df["time"].values)
    print(len(df["time"].values))

    # Plotting
    df.plot(
        x="time", y=["Didinium", "Parameidcium"], kind="line", color=["green", "blue"]
    )

    # Adding labels and title
    plt.xlabel("Time")
    plt.ylabel("Population")
    plt.title("Prey and Predator Populations Over Time")
    plt.savefig("figures/Sup_Figures_S7_abundances_para_vs_didi.png")
    plt.show()

    Para = df[" prey(#ind/ml)"].values
    Didi = df[" predator(#ind/ml)"].values

    df.set_index("time", inplace=True)
    print(len(df))

    from lutra.ccmn import ConvergentCrossMapping as ccmn

    L = len(df)  # length of time period to consider
    tau = 1  # 1 for paper
    d = 3  # change back to 2
    ccm1 = ccmn(Para, Didi, tau, d, L)
    ccm2 = ccmn(Didi, Para, tau, d, L)

    ccm3 = ccmn(Para, Didi, tau, d, L)
    ccm4 = ccmn(Didi, Para, tau, d, L)

    corr_, p, pear = ccm1.causality()
    print(corr_, p, pear)
    corr_2, p2, pear2 = ccm2.causality()
    print(corr_2, p2, pear2)

    corr_, p, pear = ccm3.nmi_causality()
    print(corr_, p, pear)
    corr_2, p2, pear2 = ccm4.nmi_causality()
    print(corr_2, p2, pear2)

    Lag = [lag for lag in range(9, L)]
    corr_Para = [ccmn(Para, Didi, tau, d, lag).causality()[0] for lag in range(9, L)]
    corr_Didi = [ccmn(Didi, Para, tau, d, lag).causality()[0] for lag in range(9, L)]

    nmi_Para = [ccmn(Para, Didi, tau, d, lag).nmi_causality()[0] for lag in range(9, L)]
    nmi_Didi = [ccmn(Didi, Para, tau, d, lag).nmi_causality()[0] for lag in range(9, L)]

    plt.figure(figsize=(4, 3))
    plt.plot(
        Lag, corr_Para, label="Parameidcium $\\rightarrow$ Didinium", color="green"
    )
    plt.plot(Lag, corr_Didi, label="Didinium $\\rightarrow$ Parameidcium", color="blue")
    plt.legend()
    # plt.title(" CCM of Paramecium and Didinium with increasing time-series length L.")
    plt.xlabel("L")
    plt.ylabel("Pearson Corr of CCM")
    plt.savefig("figures/Sup_Figures_S7_ccm_para_vs_didi_corr_plot.png")
    # plt.xlim(9, 31)
    # plt.ylim(0.65, 0.9)
    plt.show()

    plt.figure(figsize=(4, 3))
    plt.plot(Lag, nmi_Para, label="Parameidcium $\\rightarrow$ Didinium", color="green")
    plt.plot(Lag, nmi_Didi, label="Didinium $\\rightarrow$ Parameidcium", color="blue")
    # plt.title(" CCM of Paramecium and Didinium with increasing time-series length L.")
    plt.legend()
    plt.xlabel("L")
    plt.ylabel("NMI of CCM")
    plt.savefig("figures/Sup_Figures_S7_ccm_para_vs_didi_nmi_plot.png")
    # plt.xlim(9, 31)
    # plt.ylim(0.65, 0.9)
    plt.show()

    from scipy.stats import pearsonr

    r, p = pearsonr(corr_Para, nmi_Para)
    print("Pearson Correlation CORR NMI Para:", r, p)
    from scipy.stats import pearsonr

    r, p = pearsonr(corr_Didi, nmi_Didi)
    print("Pearson Correlation CORR NMI DIDI:", r, p)

    from scipy.stats import ttest_ind

    # Perform one-sided t-test
    t_statistic, p_value = ttest_ind(nmi_Didi, nmi_Para, alternative="greater")

    # Check the p-value
    alpha = 0.05  # significance level
    if p_value < alpha:
        print(
            "The mean of vector nmi_Didi is significantly greater than the mean of vector nmi_Para."
        )
        print(t_statistic, p_value)
    else:
        print(
            "There is not enough evidence to conclude that the mean of vector nmi_Didi is greater than the mean of vector nmi_Para."
        )

    from scipy.stats import ttest_ind

    # Perform one-sided t-test
    t_statistic, p_value = ttest_ind(corr_Didi, corr_Para, alternative="greater")

    # Check the p-value
    alpha = 0.05  # significance level
    if p_value < alpha:
        print(
            "The mean of vector Didi is significantly greater than the mean of vector Para."
        )
        print(t_statistic, p_value)
    else:
        print(
            "There is not enough evidence to conclude that the mean of vector Didi is greater than the mean of vector Para."
        )
